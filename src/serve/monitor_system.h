#pragma once
#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ninfer::serve {
namespace monitor_system_detail {
struct Cpu { std::uint64_t total = 0, idle = 0; };
struct Memory { std::uint64_t total = 0, used = 0; };
inline std::optional<Cpu> cpu() {
#ifdef _WIN32
    FILETIME idle{}, kernel{}, user{};
    if (!GetSystemTimes(&idle, &kernel, &user)) return std::nullopt;
    const auto ticks = [](FILETIME t) {
        return (static_cast<std::uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
    };
    return Cpu{ticks(kernel) + ticks(user), ticks(idle)};
#elif defined(__linux__)
    std::ifstream in("/proc/stat");
    std::string line, tag;
    if (!std::getline(in, line)) return std::nullopt;
    std::istringstream row(line);
    std::uint64_t user=0,nice=0,system=0,idle=0,iowait=0,irq=0,softirq=0,steal=0;
    if (!(row >> tag >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal) || tag!="cpu") return std::nullopt;
    return Cpu{user+nice+system+idle+iowait+irq+softirq+steal,idle+iowait};
#else
    return std::nullopt;
#endif
}
inline std::optional<Memory> memory() {
#ifdef _WIN32
    MEMORYSTATUSEX m{}; m.dwLength=sizeof(m);
    if (!GlobalMemoryStatusEx(&m) || m.ullAvailPhys>m.ullTotalPhys) return std::nullopt;
    return Memory{m.ullTotalPhys,m.ullTotalPhys-m.ullAvailPhys};
#elif defined(__linux__)
    std::ifstream in("/proc/meminfo");
    std::string line, tag; std::uint64_t total=0,available=0; bool found=false;
    while(std::getline(in,line)){
        std::istringstream row(line); std::uint64_t kb=0;
        if (!(row>>tag>>kb)) continue;
        if(tag=="MemTotal:") total=kb;
        if(tag=="MemAvailable:"){available=kb;found=true;}
    }
    if(!found || !total || available>total) return std::nullopt;
    return Memory{total*1024,(total-available)*1024};
#else
    return std::nullopt;
#endif
}
inline std::optional<double> cpu_pct(Cpu old, Cpu now) {
    if(now.total<=old.total || now.idle<old.idle) return std::nullopt;
    const auto dt=now.total-old.total, di=now.idle-old.idle;
    if(di>dt) return std::nullopt;
    return 100.0*static_cast<double>(dt-di)/static_cast<double>(dt);
}
} // namespace monitor_system_detail
// System-wide, not process utilization. No subprocesses or external dependencies.
inline std::string monitor_system_json(){
    using namespace monitor_system_detail;
    using Clock=std::chrono::steady_clock;
    static std::mutex mutex;
    static std::optional<Cpu> previous;
    static std::optional<double> percent;
    static Clock::time_point sampled{};
    std::lock_guard lock(mutex);
    const auto now=Clock::now();
    if(now-sampled>=std::chrono::milliseconds(500)){
        auto current=cpu();
        percent=previous&&current?cpu_pct(*previous,*current):std::nullopt;
        previous=current; sampled=now;
    }
    const auto m=memory();
    return "{\"cpu_pct\":"+(percent?std::to_string(*percent):"null")+
        ",\"ram_total_bytes\":"+(m?std::to_string(m->total):"null")+
        ",\"ram_used_bytes\":"+(m?std::to_string(m->used):"null")+"}";
}
} // namespace ninfer::serve
