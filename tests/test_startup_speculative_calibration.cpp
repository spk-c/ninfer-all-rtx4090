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
