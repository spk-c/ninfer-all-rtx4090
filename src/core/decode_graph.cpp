#include "core/decode_graph.h"

#include "core/device_launch.h"
#include "core/device_wait.h"
#include "core/nvtx.h"

#include <algorithm>
#include <utility>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void discard_capture(cudaStream_t stream) noexcept {
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
}

void check_graph(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

cudaGraph_t active_capture(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    cudaGraph_t graph = nullptr;
    check_graph(
        cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, nullptr, nullptr, nullptr),
        "cudaStreamGetCaptureInfo");
    if (status != cudaStreamCaptureStatusActive)
        throw std::logic_error("inline graph recording requires an active parent capture");
    return graph;
}

std::vector<cudaGraphNode_t> captured_nodes(cudaGraph_t graph) {
    std::size_t count = 0;
    check_graph(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
    std::vector<cudaGraphNode_t> nodes(count);
    check_graph(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes");
    return nodes;
}

std::vector<cudaGraphNode_t> added_nodes(cudaGraph_t graph, std::vector<cudaGraphNode_t> before) {
    std::ranges::sort(before);
    auto nodes = captured_nodes(graph);
    std::erase_if(nodes, [&](auto node) { return std::ranges::binary_search(before, node); });
    return nodes;
}

bool cooperative_kernel(cudaGraphNode_t node) {
    cudaKernelNodeAttrValue attribute{};
    check_graph(
        cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributeCooperative, &attribute),
        "cudaGraphKernelNodeGetAttribute");
    return attribute.cooperative;
}

std::vector<cudaGraphNode_t> dependent_nodes(cudaGraphNode_t node) {
    std::size_t count = 0;
    check_graph(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &count),
                "cudaGraphNodeGetDependentNodes");
    std::vector<cudaGraphNode_t> nodes(count);
    if (count)
        check_graph(cudaGraphNodeGetDependentNodes(node, nodes.data(), nullptr, &count),
                    "cudaGraphNodeGetDependentNodes");
    return nodes;
}

[[noreturn]] void invalid_kernel_chain() {
    throw std::logic_error("graph kernel slice requires a non-empty single dependency chain");
}

template <class Contains>
std::vector<cudaGraphNode_t> kernel_chain(cudaGraphNode_t root, std::size_t size,
                                          Contains contains) {
    std::vector<cudaGraphNode_t> nodes;
    nodes.reserve(size);
    for (auto node = root; node;) {
        cudaGraphNodeType type;
        check_graph(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
        if (type != cudaGraphNodeTypeKernel || cooperative_kernel(node))
            throw std::logic_error("graph kernel slice requires ordinary kernel nodes");
        nodes.push_back(node);
        cudaGraphNode_t next = nullptr;
        for (auto dependent : dependent_nodes(node)) {
            if (!contains(dependent)) continue;
            if (next) invalid_kernel_chain();
            next = dependent;
        }
        node = next;
    }
    if (nodes.empty() || nodes.size() != size) invalid_kernel_chain();
    return nodes;
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;

    other.graph_ = nullptr;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    nvtx::ScopedRange capture_range(nvtx::Name::CudaGraphCapture, nvtx::Category::Graph);
    reset();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    try {
        body();
    } catch (...) {
        discard_capture(stream);
        throw;
    }

    cudaGraph_t graph = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    graph_ = graph;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

void DecodeGraphKernelPatch::capture(cudaStream_t stream, const std::function<void()>& body) {
    reset();
    DecodeGraphDefinition definition;
    definition.capture(stream, body);
    std::size_t size = 0, roots = 0;
    check_graph(cudaGraphGetNodes(definition.graph_, nullptr, &size), "cudaGraphGetNodes");
    check_graph(cudaGraphGetRootNodes(definition.graph_, nullptr, &roots), "cudaGraphGetRootNodes");
    if (roots != 1) invalid_kernel_chain();
    cudaGraphNode_t root = nullptr;
    check_graph(cudaGraphGetRootNodes(definition.graph_, &root, &roots), "cudaGraphGetRootNodes");
    const auto nodes = kernel_chain(root, size, [](auto) { return true; });
    std::vector<cudaKernelNodeParams> params;
    params.reserve(size);
    for (auto node : nodes)
        check_graph(cudaGraphKernelNodeGetParams(node, &params.emplace_back()),
                    "cudaGraphKernelNodeGetParams");
    definition_ = std::move(definition);
    params_     = std::move(params);
}

void DecodeGraphKernelPatch::reset() noexcept {
    params_.clear();
    definition_.reset();
}

void DecodeGraphKernelSlice::begin(cudaStream_t stream) {
    if (capture_) throw std::logic_error("a graph kernel slice is already recording");
    const auto graph = active_capture(stream);
    before_          = captured_nodes(graph);
    nodes_.clear();
    capture_ = graph;
    parent_ = nullptr;
}

void DecodeGraphKernelSlice::end(cudaStream_t stream) {
    const auto graph = std::exchange(capture_, nullptr);
    auto before      = std::move(before_);
    if (!graph || graph != active_capture(stream))
        throw std::logic_error("a graph kernel slice must end in its parent capture");
    auto added = added_nodes(graph, std::move(before));
    if (added.empty()) invalid_kernel_chain();
    std::ranges::sort(added);
    const auto contains = [&](auto node) { return std::ranges::binary_search(added, node); };
    // External incoming edges are irrelevant. The root has no predecessor in the slice.
    auto roots = added;
    for (auto node : added)
        for (auto dependent : dependent_nodes(node))
            if (const auto it = std::ranges::lower_bound(added, dependent);
                it != added.end() && *it == dependent)
                roots[it - added.begin()] = nullptr;
    std::erase(roots, nullptr);
    if (roots.size() != 1) invalid_kernel_chain();
    nodes_ = kernel_chain(roots.front(), added.size(), contains);
    parent_ = graph;
}

void DecodeGraphKernelSlice::capture(cudaStream_t stream, const std::function<void()>& body) {
    begin(stream);
    try {
        body();
    } catch (...) {
        capture_ = nullptr;
        before_.clear();
        throw;
    }
    end(stream);
}

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_), source_graph_(other.source_graph_) {
    other.exec_ = nullptr;
    other.source_graph_ = nullptr;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_ = other.exec_;
    source_graph_ = other.source_graph_;
    other.exec_ = nullptr;
    other.source_graph_ = nullptr;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange instantiate_range(nvtx::Name::CudaGraphInstantiate, nvtx::Category::Graph);
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();

    cudaGraphExec_t exec  = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        CUDA_CHECK(err);
    }
    exec_ = exec;
    source_graph_ = definition.graph_;
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    if (err == cudaSuccess && result.result == cudaGraphExecUpdateSuccess) {
        if (source_graph_ != definition.graph_) source_graph_ = nullptr;
        return;
    }
    // A definition whose nodes differ from the executable's cannot be installed in place. Profiles
    // share an executable when their planned launches match, but a library call can still capture
    // other kernels than the planner foresaw (cuBLAS chooses its kernels per call, which sm_70's wide
    // projections use), so the executable is instantiated from the definition instead. The failed
    // update leaves the executable as it was and its error is not sticky.
    if (err == cudaErrorGraphExecUpdateFailure) {
        (void)cudaGetLastError();
        instantiate(definition);
        return;
    }
    throw std::runtime_error("CUDA Graph executable update failed: " +
                             std::string(cudaGetErrorName(err)) + " (update result " +
                             std::to_string(static_cast<int>(result.result)) + ")");
}

void DecodeGraphExecutable::update_slice(const DecodeGraphKernelSlice& slice,
                                         const DecodeGraphKernelPatch& patch) {
    if (!ready() || !patch.ready() || !source_graph_ || source_graph_ != slice.parent_ ||
        slice.empty() || slice.size() != patch.params().size())
        throw std::logic_error("CUDA Graph kernel update requires a matching parent, slice and patch");
    nvtx::ScopedRange range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    for (std::size_t i = 0; i < slice.size(); ++i) {
        const auto err = cudaGraphExecKernelNodeSetParams(exec_, slice.nodes_[i], &patch.params()[i]);
        if (err != cudaSuccess) {
            // Never replay an executable after a partially applied patch.
            reset();
            check_graph(err, "cudaGraphExecKernelNodeSetParams");
        }
    }
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    nvtx::ScopedRange upload_range(nvtx::Name::CudaGraphUpload, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphUpload(exec_, stream));
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    // This range executes for every replay; ranges in the captured body execute only at capture.
    nvtx::ScopedRange launch_range(nvtx::Name::CudaGraphLaunch, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    CUDA_CHECK(ninfer::launch_graph(exec_, stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept {
    destroy_graph_exec(exec_);
    source_graph_ = nullptr;
}

} // namespace ninfer
