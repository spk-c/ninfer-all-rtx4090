#include "models/qwen3_5/program/startup_speculative_calibration.h"
#include "runtime/contract/mtp_adaptive.h"
#include <array>
#include <iostream>

int main() {
    using namespace ninfer::models::qwen3_5::detail;
    using E = GraphProfileIndex::Entry;
    const std::array entries{E{1,2,0,4095,0,0}, E{1,5,0,4095,1,1},
                             E{4,2,0,4095,2,2}, E{4,5,0,4095,3,3}};
    auto probes = startup_speculative_probes(entries, 4096, 4);
    if (probes.size() != 8) return 1;
    if (startup_speculative_probes(entries, 64, 4).size()) return 2;
    if (startup_speculative_probes(entries, 2048, 4).size() != 4) return 3;
    const std::array long_entries{E{1,5,0,65535,0,0}};
    const auto physical = startup_speculative_probes(long_entries, 65536, 1, true);
    if (physical.size() != 4 || physical.back().frontier != 32768) return 6;
    const std::array<std::uint32_t, 4> mapped{1,1,1,1};
    if (!startup_physical_backing_fits(mapped,1,513,512,1024)) return 7;
    if (startup_physical_backing_fits(mapped,2,513,1024,1024)) return 8;
    if (startup_physical_backing_fits(mapped,1,513,511,1024)) return 9;
    if (startup_physical_backing_fits(mapped,1,513,1024,512)) return 10;
    if (startup_physical_backing_fits(mapped,5,1,1024,1024)) return 11;
    const std::array<std::uint32_t, 4> retained{513,1,1,1};
    if (!startup_physical_backing_fits(retained,1,128,0,1024)) return 12;
    if (startup_physical_backing_fits(retained,2,513,512,1024)) return 13;
    ninfer::runtime::MaskedWidthController c;
    c.reset(5, 2);
    c.seed_cost(1, 2, .010, 64);
    c.seed_cost(1, 5, .100, 64);
    ninfer::runtime::MaskedWidthSignal signal;
    signal.reset(2, 5);
    const std::array<const ninfer::runtime::MaskedWidthSignal*,1> signals{&signal};
    const std::array<std::uint32_t,1> room{4096};
    unsigned width = 5;
    for (int i=0; i<8; ++i) width=c.select(signals,room,64);
    if (width != 2) return 4;
    if (c.select(signals, room, 65536) != 5) return 5;
    std::cout << "startup calibration: corners, capacity and context-isolated prior passed\n";
}
