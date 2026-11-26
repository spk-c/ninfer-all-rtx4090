#include "models/qwen3_5/program/graph_profile_index.h"
#include <iostream>
using ninfer::models::qwen3_5::detail::GraphProfileIndex;
int main() {
    GraphProfileIndex index;
    using E = GraphProfileIndex::Entry;
    index.build({E{2,5,0,100,3,0}, E{1,5,65,100,2,1}, E{1,5,0,64,1,2},
                 E{1,2,0,100,4,3}}, 100);
    for (unsigned f=0; f<=100; ++f) {
        if (index.select(1,5,f).profile != (f<=64 ? 2u : 1u)) return 1;
        if (index.select(2,5,f).profile != 0 || index.select(1,2,f).profile != 3) return 2;
    }
    auto rejects = [](auto&& action) {
        try { action(); } catch (const std::logic_error&) { return true; }
        return false;
    };
    if (!rejects([&]{ (void)index.select(1,5,101); }) ||
        !rejects([&]{ (void)index.select(1,3,0); }) ||
        !rejects([&]{ index.build({E{1,5,0,64,0,0},E{1,5,64,100,0,1}},100); }) ||
        !rejects([&]{ index.build({E{1,5,0,64,0,0},E{1,5,66,100,0,1}},100); }) ||
        !rejects([&]{ index.build({E{1,5,0,99,0,0}},100); })) return 3;
    std::cout << "graph profile index: coverage, boundaries, batches and windows passed\n";
}
