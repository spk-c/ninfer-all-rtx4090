// The host checks that keep a card pair off peer transfers before a probe touches its mappings
// (core/peer_preflight.h), read from a fabricated sysfs tree: the IOMMU group type, the BAR1 size
// of the second resource line, and the GeForce-only BAR1 rule.

#include "core/peer_preflight.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

constexpr std::uint64_t kGiB = 1ULL << 30U;

// A device directory with an optional IOMMU group type and a resource file whose BAR1 spans
// `bar1` bytes (none when zero).
void make_device(const fs::path& root, const std::string& address, const char* iommu_type,
                 std::uint64_t bar1) {
    const fs::path device = root / "bus/pci/devices" / address;
    fs::create_directories(device);
    if (iommu_type != nullptr) {
        fs::create_directories(device / "iommu_group");
        std::ofstream(device / "iommu_group/type") << iommu_type << '\n';
    }
    std::ofstream resource(device / "resource");
    resource << "0x00000000fb000000 0x00000000fbffffff 0x0000000000040200\n";
    if (bar1 != 0) {
        const std::uint64_t start = 0x0000038000000000ULL;
        resource << "0x" << std::hex << start << " 0x" << (start + bar1 - 1)
                 << " 0x000000000014220c\n";
    } else {
        resource << "0x0000000000000000 0x0000000000000000 0x0000000000000000\n";
    }
    resource << "0x0000000000000000 0x0000000000000000 0x0000000000000000\n";
}

ninfer::PeerDeviceFacts facts(const fs::path& root, const char* name, const char* bus_id,
                              std::uint64_t memory) {
    return ninfer::read_peer_device_facts(root.string(), name, bus_id, memory);
}

void cases(const fs::path& root) {
#ifndef _WIN32
    make_device(root, "0000:01:00.0", "identity", 32 * kGiB);
    make_device(root, "0000:02:00.0", "identity", 256ULL << 20U);
    make_device(root, "0000:3b:00.0", "DMA-FQ", 32 * kGiB);
    make_device(root, "0000:5e:00.0", nullptr, 256ULL << 20U);

    const auto rebar  = facts(root, "NVIDIA GeForce RTX 3090", "0000:01:00.0", 24 * kGiB);
    const auto small  = facts(root, "NVIDIA GeForce RTX 3090", "00000000:02:00.0", 24 * kGiB);
    const auto dma    = facts(root, "Tesla V100-PCIE-32GB", "0000:3B:00.0", 32 * kGiB);
    const auto tesla  = facts(root, "Tesla T4", "0000:5e:00.0", 16 * kGiB);
    const auto absent = facts(root, "NVIDIA GeForce RTX 4090", "0000:99:00.0", 24 * kGiB);

    check(rebar.iommu_type == "identity", "the group type is read and trimmed");
    check(rebar.bar1_bytes == 32 * kGiB, "BAR1 is the second resource line");
    check(small.bar1_bytes == (256ULL << 20U), "an eight-digit domain finds the same directory");
    check(dma.iommu_type == "DMA-FQ", "an upper-case address finds the lower-case directory");
    check(!tesla.iommu_type.has_value(), "no group directory: no IOMMU");
    check(!absent.iommu_type && !absent.bar1_bytes, "a missing device reports no facts");
#else
    // Windows has no sysfs and cannot create its colon-containing PCI directories. Exercise
    // the same production parser and refusal policy from represented file contents instead.
    const auto parse = [](const char* name, const char* address, const char* type,
                          std::uint64_t memory, std::uint64_t bar1) {
        std::ostringstream resource;
        resource << "0xfb000000 0xfbffffff 0x40200\n0x38000000000 0x" << std::hex
                 << (0x38000000000ULL + bar1 - 1) << " 0x14220c\n0x0 0x0 0x0\n";
        const std::string bytes = resource.str();
        return ninfer::parse_peer_device_facts(name, address, memory,
            type ? std::optional<std::string_view>(type) : std::nullopt, bytes);
    };
    const auto rebar = parse("NVIDIA GeForce RTX 3090", "0000:01:00.0", "identity\n",
                             24 * kGiB, 32 * kGiB);
    const auto small = parse("NVIDIA GeForce RTX 3090", "00000000:02:00.0", "identity\n",
                             24 * kGiB, 256ULL << 20);
    const auto dma = parse("Tesla V100-PCIE-32GB", "0000:3B:00.0", "DMA-FQ\n",
                           32 * kGiB, 32 * kGiB);
    const auto tesla = parse("Tesla T4", "0000:5e:00.0", nullptr, 16 * kGiB, 256ULL << 20);
    const auto absent = facts(root, "NVIDIA GeForce RTX 4090", "0000:99:00.0", 24 * kGiB);
    check(rebar.iommu_type == "identity" && rebar.bar1_bytes == 32 * kGiB,
          "sysfs contents parse without native PCI directory names");
    check(small.bar1_bytes == (256ULL << 20) && dma.iommu_type == "DMA-FQ",
          "BAR1 and translating IOMMU contents parse exactly");
    check(!tesla.iommu_type && !absent.iommu_type && !absent.bar1_bytes,
          "missing group/files remain unknown on Windows");
#endif

    const auto unused = ninfer::parse_peer_device_facts(
        "GPU", "0000:01:00.0", kGiB, "identity\r\n", "0x1000 0x1fff 0\n0 0 0\n");
    check(unused.iommu_type == "identity" && !unused.bar1_bytes,
          "an unused BAR1 remains unknown and CRLF group types are trimmed");
    const auto truncated = ninfer::parse_peer_device_facts(
        "GPU", "0000:01:00.0", kGiB, std::nullopt, "0x1000 0x1fff 0\n");
    check(!truncated.bar1_bytes && !truncated.iommu_type,
          "BAR0 must not substitute for a missing BAR1");
    const auto malformed = ninfer::parse_peer_device_facts(
        "GPU", "0000:01:00.0", kGiB, std::nullopt, "0 0 0\ninvalid resource\n");
    check(!malformed.bar1_bytes, "malformed BAR1 must not invent a size");

    check(ninfer::peer_preflight_refusal(rebar, rebar).empty(), "full BAR1, identity: probed");
    check(ninfer::peer_preflight_refusal(absent, rebar).empty(), "unknown facts: probed");
    check(ninfer::peer_preflight_refusal(tesla, tesla).empty(),
          "a small BAR1 outside GeForce: probed (mailbox mappings work there)");
    const std::string translated = ninfer::peer_preflight_refusal(rebar, dma);
    check(translated.find("DMA-FQ") != std::string::npos &&
              translated.find("iommu=pt") != std::string::npos,
          "a translating IOMMU on either card: refused, naming the fix");
    const std::string mailbox = ninfer::peer_preflight_refusal(small, rebar);
    check(mailbox.find("256 MiB") != std::string::npos &&
              mailbox.find("24,576 MiB") != std::string::npos,
          "a GeForce BAR1 below its memory: refused, with both sizes");
}

} // namespace

int run_peer_preflight_test() {
    const fs::path root =
        fs::temp_directory_path() / ("ninfer-peer-preflight-" + std::to_string(std::rand()));
    try {
        fs::remove_all(root);
        cases(root);
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);
    if (failures != 0) { return 1; }
    std::cout << "peer preflight tests passed\n";
    return 0;
}
