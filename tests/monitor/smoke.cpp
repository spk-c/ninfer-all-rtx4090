#include "serve/monitor_dashboard.h"
#include "serve/monitor_gpu.h"
#include "serve/monitor_system.h"
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    using namespace ninfer::serve;
    using namespace ninfer::serve::monitor_system_detail;
    auto value = cpu_pct({100,50},{200,75});
    assert(value && *value == 75.0);
    assert(!cpu_pct({100,50},{99,49}));
    assert(!cpu_pct({100,50},{101,52}));
    assert(!cpu_pct({100,50},{100,50}));
    assert(cpu_pct({100,50},{200,150}) == 0.0);
    const auto html = monitor_dashboard_html();
    assert(html.starts_with("<!doctype html>"));
    assert(html.find("Test chat") != std::string_view::npos);
    assert(html.find("monitor/system") != std::string_view::npos);
    assert(html.ends_with("</html>"));
    const auto first = monitor_system_json();
    assert(first.find("\"cpu_pct\":null") != std::string::npos);
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    const auto next = monitor_system_json();
    assert(next.find("\"ram_used_bytes\":") != std::string::npos);
    const auto gpu = monitor_gpu_json(0);
    assert(gpu.find("\"available\":") != std::string::npos);
    if (argc == 2) {
        std::ofstream out(argv[1], std::ios::binary);
        out << html;
        assert(out.good());
    }
    std::cout << "Monitor header/sensor smoke PASS\n";
}
