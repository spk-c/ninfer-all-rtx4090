#pragma once

#include <algorithm>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ninfer::serve {
namespace monitor_detail {
#ifdef _WIN32
using Module = HMODULE;
// Only the trusted Windows system directory may supply the GPU driver DLL.
inline Module open() {
    return LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
}
inline void* symbol(Module m, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(m, name));
}
inline void close(Module m) { if (m) { FreeLibrary(m); } }
#else
using Module = void*;
inline Module open() { return dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL); }
inline void* symbol(Module m, const char* name) { return dlsym(m, name); }
inline void close(Module m) { if (m) { dlclose(m); } }
#endif

// Matches the public NVML v1 ABI; no CUDA/NVML headers or import library needed.
using Device = void*;
using Result = int;
struct Utilization { unsigned int gpu, memory; };
struct Memory { unsigned long long total, free, used; };
template <class T> T function(Module module, const char* name) {
    return reinterpret_cast<T>(symbol(module, name));
}
inline std::string value(bool available, double amount) {
    return available ? std::to_string(amount) : "null";
}
} // namespace monitor_detail

// The driver library and sensor functions are optional. Never touch the
// inference CUDA context, and never make /stats depend on NVML availability.
inline std::string monitor_gpu_json(unsigned int requested_index) {
    using namespace monitor_detail;
    const unsigned int index = std::min(requested_index, 7u);
    const std::string unavailable = "{\"available\":false,\"index\":" +
                                    std::to_string(index) + "}";
    const Module module = open();
    if (!module) { return unavailable; }
    using Init = Result (*)();
    using Shutdown = Result (*)();
    using GetHandle = Result (*)(unsigned int, Device*);
    using GetUtil = Result (*)(Device, Utilization*);
    using GetTemp = Result (*)(Device, unsigned int, unsigned int*);
    using GetPower = Result (*)(Device, unsigned int*);
    using GetMemory = Result (*)(Device, Memory*);
    const auto init = function<Init>(module, "nvmlInit_v2");
    const auto shutdown = function<Shutdown>(module, "nvmlShutdown");
    auto get_handle = function<GetHandle>(module, "nvmlDeviceGetHandleByIndex_v2");
    if (!get_handle) {
        get_handle = function<GetHandle>(module, "nvmlDeviceGetHandleByIndex");
    }
    if (!init || !shutdown || !get_handle || init() != 0) {
        close(module);
        return unavailable;
    }
    Device device = nullptr;
    if (get_handle(index, &device) != 0 || !device) {
        shutdown();
        close(module);
        return unavailable;
    }
    const auto get_util = function<GetUtil>(module, "nvmlDeviceGetUtilizationRates");
    const auto get_temp = function<GetTemp>(module, "nvmlDeviceGetTemperature");
    const auto get_power = function<GetPower>(module, "nvmlDeviceGetPowerUsage");
    const auto get_memory = function<GetMemory>(module, "nvmlDeviceGetMemoryInfo");
    Utilization utilization{};
    Memory memory{};
    unsigned int temperature = 0, power_mw = 0;
    const bool util_ok = get_util && get_util(device, &utilization) == 0;
    const bool temp_ok = get_temp && get_temp(device, 0, &temperature) == 0;
    const bool power_ok = get_power && get_power(device, &power_mw) == 0;
    const bool memory_ok = get_memory && get_memory(device, &memory) == 0;
    shutdown();
    close(module);

    std::string result = "{\"available\":true,\"index\":" + std::to_string(index);
    result += ",\"utilization_pct\":" + value(util_ok, utilization.gpu);
    result += ",\"temperature_c\":" + value(temp_ok, temperature);
    result += ",\"power_w\":" + value(power_ok, power_mw / 1000.0);
    result += ",\"memory_used_bytes\":" + value(memory_ok, static_cast<double>(memory.used));
    result += ",\"memory_total_bytes\":" + value(memory_ok, static_cast<double>(memory.total));
    result += "}";
    return result;
}
} // namespace ninfer::serve
