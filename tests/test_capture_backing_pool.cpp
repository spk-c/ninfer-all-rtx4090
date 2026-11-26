#include "models/qwen3_5/program/prefix/capture_backing_pool.h"

#include <iostream>
#include <memory>
#include <string>

using ninfer::models::qwen3_5::detail::CaptureBackingPool;

int main() {
    CaptureBackingPool<int, std::string, 2> pool;
    int allocations = 0;
    auto get = [&](int key, const std::string& value) {
        return pool.get_or_create(key, [&](const auto& v) { return v == value; },
                                  [&] {
                                      ++allocations;
                                      return std::make_shared<const std::string>(value);
                                  });
    };
    auto first = get(1, "prefix-A");
    if (get(1, "prefix-A") != first || allocations != 1) return 1;
    auto collision = get(1, "prefix-B");
    if (collision == first || allocations != 2) return 2;
    if (get(1, "prefix-B") != collision || allocations != 2) return 3;
    std::weak_ptr<const std::string> old = collision;
    collision.reset();
    if (!old.expired()) return 4;
    auto third = get(2, "prefix-C");
    if (get(2, "prefix-C") != third || allocations != 3) return 5;
    auto fourth = get(3, "prefix-D");
    if (*first != "prefix-A" || *third != "prefix-C" || *fourth != "prefix-D") return 6;
    std::cout << "capture backing pool: reuse, exact-match collision guard, weak ownership and bounded eviction passed\n";
    return 0;
}
