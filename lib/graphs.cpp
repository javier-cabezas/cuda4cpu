/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// CUDA graphs and stream capture.
//
// A graph owns its nodes, and nodes own their child and body graphs. An
// executable graph is a snapshot of the nodes in a topological order (stable:
// nodes that don't depend on each other keep the order they were added in),
// so the graph can change or be destroyed afterwards. Launching it runs the
// nodes one after the other, before returning, as kernel launches do.
//
// Stream capture: a capturing stream has a stream_capture with the nodes that
// the next captured work depends on. Each captured operation becomes a node
// that depends on them, and then the only dependency. Events recorded in a
// capturing stream remember the stream's dependencies, so that a stream that
// waits for them joins the capture (fork and join, as in CUDA).

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "cuda4cpu.hpp"

namespace cuda4cpu::detail {

//! The memory of a graph allocation node, as in CUDA: an address range
//! reserved when the node is added, so that the address is fixed, and
//! allocated (committed) from when the node runs to when a free node, cudaFree
//! or cudaFreeAsync frees it. Using it at any other time faults. Relaunching
//! the graph allocates it again at the same address. It outlives the graph
//! while allocated: memory the graph doesn't free is the program's to free.
struct graph_allocation {
    void *ptr;
    size_t size;
    size_t mapped;           //!< size rounded up to pages
    bool live = false;       //!< Allocated
    bool orphaned = false;   //!< No graph has the node any more
};

struct capture_session {
    cudaGraph_t graph;
    bool owns_graph;                    //!< Created by cudaStreamBeginCapture
    unsigned long long id;
    cudaStream_t origin;                //!< Where the capture began, and must end
    std::vector<cudaStream_t> streams;  //!< Streams in the capture, the origin first
    bool active = true;
};

struct stream_capture {
    std::shared_ptr<capture_session> session;
    std::vector<cudaGraphNode_t> deps;  //!< What the next captured operation depends on
};

namespace {

//
// Footprint of graph allocations (cudaDeviceGetGraphMemAttribute)
//

std::atomic<uint64_t> used_current{0}, used_high{0}, reserved_current{0}, reserved_high{0};

void raise_high(std::atomic<uint64_t> &high, uint64_t value)
{
    uint64_t seen = high.load(std::memory_order_relaxed);
    while (value > seen && !high.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
    }
}

//! Graph allocations by address, for free nodes and cudaFree. Graph nodes
//! share them through shared_ptrs whose deleter orphans them.
struct allocation_registry {
    std::mutex mutex;
    std::unordered_map<void *, std::unique_ptr<graph_allocation>> by_address;
    std::atomic<size_t> count{0};   //!< Lets cudaFree skip the lookup when there are none
};

allocation_registry &allocations()
{
    static allocation_registry registry;
    return registry;
}

//! Releases the address range of an allocation (with the registry locked)
void unmap(allocation_registry &r, graph_allocation *a)
{
    munmap(a->ptr, a->mapped);
    r.count.fetch_sub(1, std::memory_order_relaxed);
    r.by_address.erase(a->ptr);   // destroys a
}

//! Frees an allocation (with the registry locked)
void decommit(allocation_registry &r, graph_allocation *a)
{
    madvise(a->ptr, a->mapped, MADV_DONTNEED);
    mprotect(a->ptr, a->mapped, PROT_NONE);
    a->live = false;
    used_current.fetch_sub(a->size, std::memory_order_relaxed);
    reserved_current.fetch_sub(a->size, std::memory_order_relaxed);
    if (a->orphaned)
        unmap(r, a);
}

std::shared_ptr<graph_allocation> new_allocation(size_t size)
{
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
    const size_t mapped = (size + page - 1) / page * page;
    void *ptr = mmap(nullptr, mapped, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ptr == MAP_FAILED)
        return nullptr;

    auto &r = allocations();
    std::lock_guard lock(r.mutex);
    auto owned = std::make_unique<graph_allocation>(graph_allocation{ptr, size, mapped});
    graph_allocation *a = owned.get();
    r.by_address.emplace(ptr, std::move(owned));
    r.count.fetch_add(1, std::memory_order_relaxed);
    return std::shared_ptr<graph_allocation>(a, [](graph_allocation *orphan) {
        auto &reg = allocations();
        std::lock_guard l(reg.mutex);
        if (orphan->live)
            orphan->orphaned = true;   // the program frees it
        else
            unmap(reg, orphan);
    });
}

//! The graph allocation at ptr, shared with the nodes that have it
std::shared_ptr<graph_allocation> find_allocation(void *ptr, const std::vector<std::shared_ptr<graph_allocation>> &known)
{
    for (const auto &a : known) {
        if (a->ptr == ptr)
            return a;
    }
    return nullptr;
}

//! Allocates a graph allocation, when its node runs
void commit(graph_allocation &a)
{
    auto &r = allocations();
    std::lock_guard lock(r.mutex);
    if (a.live)
        return;
    mprotect(a.ptr, a.mapped, PROT_READ | PROT_WRITE);
    a.live = true;
    raise_high(used_high, used_current.fetch_add(a.size, std::memory_order_relaxed) + a.size);
    raise_high(reserved_high, reserved_current.fetch_add(a.size, std::memory_order_relaxed) + a.size);
}

//! Kernels whose address the program took with kernel_address()
struct kernel_registry {
    std::shared_mutex mutex;
    std::unordered_map<const void *, kernel_binder> binders;
};

kernel_registry &kernels()
{
    static kernel_registry registry;
    return registry;
}

}

cudaError_t free_memory(void *ptr)
{
    auto &r = allocations();
    if (ptr != nullptr && r.count.load(std::memory_order_relaxed) != 0) {
        std::lock_guard lock(r.mutex);
        auto it = r.by_address.find(ptr);
        if (it != r.by_address.end()) {
            if (!it->second->live)
                return record_error(cudaErrorInvalidValue);   // already free
            decommit(r, it->second.get());
            return cudaSuccess;
        }
    }
    std::free(ptr);
    return cudaSuccess;
}

void register_kernel(const void *func, kernel_binder binder)
{
    auto &r = kernels();
    {
        std::shared_lock lock(r.mutex);
        auto it = r.binders.find(func);
        if (it != r.binders.end() && it->second == binder)
            return;
    }
    std::unique_lock lock(r.mutex);
    r.binders[func] = binder;
}

kernel_binder find_kernel(const void *func)
{
    auto &r = kernels();
    std::shared_lock lock(r.mutex);
    auto it = r.binders.find(func);
    return it == r.binders.end() ? nullptr : it->second;
}

}

using namespace cuda4cpu;

namespace cuda4cpu::cuda_api {

struct cudaGraphNode__ {
    cudaGraph_t graph = nullptr;
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    std::vector<cudaGraphNode_t> deps;         //!< Nodes this one depends on
    std::vector<cudaGraphNode_t> dependents;   //!< Nodes that depend on this one

    // Kernel: the kernel with copies of its arguments, which kernel_params points to
    std::shared_ptr<detail::bound_kernel> kernel;
    cudaKernelNodeParams kernel_params{};
    bool cooperative = false;

    cudaMemcpy3DParms copy{};
    cudaMemsetParams fill{};
    cudaHostNodeParams host{};
    cudaEvent_t event = nullptr;
    cudaGraph_t child = nullptr;   //!< Owned

    // Memory: an allocation node's allocation, or a free node's (or, for memory
    // that isn't from a graph, the pointer that it frees)
    std::shared_ptr<detail::graph_allocation> allocation;
    cudaMemAllocNodeParams alloc_params{};
    void *free_ptr = nullptr;

    // Conditional
    std::shared_ptr<detail::graph_condition> condition;
    cudaGraphConditionalNodeType condition_type = cudaGraphCondTypeIf;
    std::vector<cudaGraph_t> bodies;   //!< Owned

    ~cudaGraphNode__();
};

struct cudaGraph__ {
    std::vector<std::unique_ptr<cudaGraphNode__>> nodes;
    std::vector<std::shared_ptr<detail::graph_condition>> conditions;
    //! Of this graph's allocation nodes, and of the graphs it is in, for free nodes
    std::vector<std::shared_ptr<detail::graph_allocation>> allocations;
};

cudaGraphNode__::~cudaGraphNode__()
{
    delete child;
    for (cudaGraph_t body : bodies)
        delete body;
}

}

namespace cuda4cpu::detail {

//! A node of an executable graph: the parameters it had when the graph was
//! instantiated (or updated)
struct exec_node {
    cudaGraphNode_t origin;   //!< The graph node it came from, which identifies it in updates
    cudaGraphNodeType type;
    std::vector<size_t> deps;   //!< Positions of the nodes it depends on, to compare topologies

    std::shared_ptr<detail::bound_kernel> kernel;
    launch_conf conf;
    bool cooperative = false;
    cudaMemcpy3DParms copy{};
    cudaMemsetParams fill{};
    cudaHostNodeParams host{};
    cudaEvent_t event = nullptr;
    std::unique_ptr<cudaGraphExec__> child;
    std::shared_ptr<detail::graph_allocation> allocation;
    void *free_ptr = nullptr;
    std::shared_ptr<detail::graph_condition> condition;
    cudaGraphConditionalNodeType condition_type = cudaGraphCondTypeIf;
    std::vector<std::unique_ptr<cudaGraphExec__>> bodies;
};

}

namespace cuda4cpu::cuda_api {

struct cudaGraphExec__ {
    std::vector<detail::exec_node> nodes;   //!< In the order they run
    std::vector<std::shared_ptr<detail::graph_condition>> conditions;   //!< Of this graph and its bodies
};

}

namespace {

using detail::exec_node;

thread_local cudaStreamCaptureMode thread_capture_mode = cudaStreamCaptureModeGlobal;
std::atomic<unsigned long long> next_capture_id{1};

bool owns(cudaGraph_t graph, cudaGraphNode_t node)
{
    return node != nullptr && node->graph == graph;
}

//! Adds node to graph, after the nodes in deps
cudaError_t add_node(cudaGraphNode_t *out, cudaGraph_t graph, const cudaGraphNode_t *deps, size_t ndeps,
                     std::unique_ptr<cudaGraphNode__> node)
{
    node->graph = graph;
    for (size_t i = 0; i < ndeps; ++i) {
        if (std::find(node->deps.begin(), node->deps.end(), deps[i]) != node->deps.end())
            continue;
        node->deps.push_back(deps[i]);
        deps[i]->dependents.push_back(node.get());
    }
    *out = node.get();
    graph->nodes.push_back(std::move(node));
    return cudaSuccess;
}

//! Checks the arguments that every cudaGraphAdd*Node takes
cudaError_t check_add(cudaGraphNode_t *out, cudaGraph_t graph, const cudaGraphNode_t *deps, size_t ndeps)
{
    if (out == nullptr || graph == nullptr || (ndeps > 0 && deps == nullptr))
        return detail::record_error(cudaErrorInvalidValue);
    for (size_t i = 0; i < ndeps; ++i) {
        if (!owns(graph, deps[i]))
            return detail::record_error(cudaErrorInvalidValue);
    }
    return cudaSuccess;
}

//! Binds a kernel node's kernel to copies of its arguments
cudaError_t bind_kernel(const cudaKernelNodeParams &p, std::shared_ptr<detail::bound_kernel> &kernel,
                        cudaKernelNodeParams &stored)
{
    if (p.extra != nullptr)
        return detail::record_error(cudaErrorNotSupported);
    if (!detail::valid_launch(launch_conf{p.cuda4cpu_gridDim, p.cuda4cpu_blockDim, p.sharedMemBytes}))
        return detail::record_error(cudaErrorInvalidConfiguration);
    const detail::kernel_binder bind = detail::find_kernel(p.func);
    if (bind == nullptr) {
        std::fprintf(stderr, "cuda4cpu: kernel node with an unknown kernel (%p): take kernel addresses with "
                             "cuda4cpu::kernel_address(kernel), which cuda4cpu-rewrite inserts for (void *)kernel\n",
                     p.func);
        return detail::record_error(cudaErrorInvalidDeviceFunction);
    }
    std::unique_ptr<detail::bound_kernel> bound = bind(p.func, p.kernelParams);
    if (bound == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    kernel = std::move(bound);
    stored = p;
    stored.kernelParams = kernel->params();
    return cudaSuccess;
}

std::unique_ptr<cudaGraphNode__> new_node(cudaGraphNodeType type)
{
    auto node = std::make_unique<cudaGraphNode__>();
    node->type = type;
    return node;
}

//! Deep copy of graph; conditions are shared with the original
cudaGraph_t clone(cudaGraph_t graph)
{
    auto *copy = new cudaGraph__;
    copy->conditions  = graph->conditions;
    copy->allocations = graph->allocations;
    std::map<cudaGraphNode_t, cudaGraphNode_t> map;
    for (const auto &n : graph->nodes) {
        auto node = std::make_unique<cudaGraphNode__>();
        node->graph          = copy;
        node->type           = n->type;
        node->kernel         = n->kernel;
        node->kernel_params  = n->kernel_params;
        node->cooperative    = n->cooperative;
        node->copy           = n->copy;
        node->fill           = n->fill;
        node->host           = n->host;
        node->event          = n->event;
        node->child          = n->child ? clone(n->child) : nullptr;
        node->allocation     = n->allocation;
        node->alloc_params   = n->alloc_params;
        node->free_ptr       = n->free_ptr;
        node->condition      = n->condition;
        node->condition_type = n->condition_type;
        for (cudaGraph_t body : n->bodies)
            node->bodies.push_back(clone(body));
        map[n.get()] = node.get();
        copy->nodes.push_back(std::move(node));
    }
    for (const auto &n : graph->nodes) {
        cudaGraphNode_t c = map[n.get()];
        for (cudaGraphNode_t d : n->deps)
            c->deps.push_back(map[d]);
        for (cudaGraphNode_t d : n->dependents)
            c->dependents.push_back(map[d]);
    }
    return copy;
}

//! The nodes of graph in the order they run, or an empty vector if the graph
//! has a cycle
std::vector<cudaGraphNode_t> topological_order(cudaGraph_t graph)
{
    std::map<cudaGraphNode_t, size_t> position, pending;
    for (size_t i = 0; i < graph->nodes.size(); ++i) {
        position[graph->nodes[i].get()] = i;
        pending[graph->nodes[i].get()] = graph->nodes[i]->deps.size();
    }
    // Ready nodes, by position: nodes that don't depend on each other run in
    // the order they were added
    std::map<size_t, cudaGraphNode_t> ready;
    for (const auto &n : graph->nodes) {
        if (n->deps.empty())
            ready[position[n.get()]] = n.get();
    }
    std::vector<cudaGraphNode_t> order;
    while (!ready.empty()) {
        cudaGraphNode_t n = ready.begin()->second;
        ready.erase(ready.begin());
        order.push_back(n);
        for (cudaGraphNode_t d : n->dependents) {
            if (--pending[d] == 0)
                ready[position[d]] = d;
        }
    }
    if (order.size() != graph->nodes.size())
        order.clear();
    return order;
}

cudaError_t instantiate(cudaGraph_t graph, std::unique_ptr<cudaGraphExec__> &out)
{
    const std::vector<cudaGraphNode_t> order = topological_order(graph);
    if (order.size() != graph->nodes.size())
        return detail::record_error(cudaErrorInvalidValue);   // a cycle

    auto exec = std::make_unique<cudaGraphExec__>();
    exec->conditions = graph->conditions;
    std::map<cudaGraphNode_t, size_t> position;
    for (cudaGraphNode_t n : order) {
        exec_node e;
        e.origin         = n;
        e.type           = n->type;
        e.kernel         = n->kernel;
        e.conf           = launch_conf{n->kernel_params.cuda4cpu_gridDim, n->kernel_params.cuda4cpu_blockDim,
                                       n->kernel_params.sharedMemBytes};
        e.cooperative    = n->cooperative;
        e.copy           = n->copy;
        e.fill           = n->fill;
        e.host           = n->host;
        e.event          = n->event;
        e.allocation     = n->allocation;
        e.free_ptr       = n->free_ptr;
        e.condition      = n->condition;
        e.condition_type = n->condition_type;
        if (n->child != nullptr) {
            if (cudaError_t err = instantiate(n->child, e.child))
                return err;
            exec->conditions.insert(exec->conditions.end(), e.child->conditions.begin(),
                                    e.child->conditions.end());
        }
        for (cudaGraph_t body : n->bodies) {
            std::unique_ptr<cudaGraphExec__> b;
            if (cudaError_t err = instantiate(body, b))
                return err;
            exec->conditions.insert(exec->conditions.end(), b->conditions.begin(), b->conditions.end());
            e.bodies.push_back(std::move(b));
        }
        for (cudaGraphNode_t d : n->deps)
            e.deps.push_back(position[d]);
        std::sort(e.deps.begin(), e.deps.end());
        position[n] = exec->nodes.size();
        exec->nodes.push_back(std::move(e));
    }
    out = std::move(exec);
    return cudaSuccess;
}

cudaError_t run(cudaGraphExec__ &exec)
{
    for (exec_node &n : exec.nodes) {
        cudaError_t err = cudaSuccess;
        switch (n.type) {
        case cudaGraphNodeTypeKernel:
            n.kernel->run(n.conf, n.cooperative);
            break;
        case cudaGraphNodeTypeMemcpy:
            err = detail::copy_3d(n.copy);
            break;
        case cudaGraphNodeTypeMemset:
            err = detail::fill(n.fill);
            break;
        case cudaGraphNodeTypeHost:
            n.host.fn(n.host.userData);
            break;
        case cudaGraphNodeTypeGraph:
            err = run(*n.child);
            break;
        case cudaGraphNodeTypeEventRecord:
            err = cudaEventRecord(n.event, nullptr);
            break;
        case cudaGraphNodeTypeMemAlloc:
            detail::commit(*n.allocation);
            break;
        case cudaGraphNodeTypeMemFree:
            err = detail::free_memory(n.free_ptr);
            break;
        case cudaGraphNodeTypeConditional: {
            auto value = [&] { return n.condition->value.load(std::memory_order_relaxed); };
            switch (n.condition_type) {
            case cudaGraphCondTypeIf:
                if (value() != 0)
                    err = run(*n.bodies[0]);
                else if (n.bodies.size() > 1)
                    err = run(*n.bodies[1]);
                break;
            case cudaGraphCondTypeWhile:
                while (err == cudaSuccess && value() != 0)
                    err = run(*n.bodies[0]);
                break;
            case cudaGraphCondTypeSwitch:
                if (value() < n.bodies.size())
                    err = run(*n.bodies[value()]);
                break;
            }
            break;
        }
        default:   // empty and event wait nodes
            break;
        }
        if (err != cudaSuccess)
            return err;
    }
    return cudaSuccess;
}

exec_node *find_exec_node(cudaGraphExec_t exec, cudaGraphNode_t node)
{
    for (exec_node &e : exec->nodes) {
        if (e.origin == node)
            return &e;
    }
    return nullptr;
}

//! Adds a node to the graph that stream is captured into
cudaError_t capture(cudaStream_t stream, std::unique_ptr<cudaGraphNode__> node)
{
    detail::stream_capture &c = *stream->capture;
    cudaGraphNode_t added = nullptr;
    add_node(&added, c.session->graph, c.deps.data(), c.deps.size(), std::move(node));
    c.deps.assign(1, added);
    return cudaSuccess;
}

//! Ends the capture of session in all its streams
void end_session(detail::capture_session &session)
{
    for (cudaStream_t s : session.streams) {
        delete s->capture;
        s->capture = nullptr;
    }
    session.streams.clear();
    session.active = false;
}

const char *type_name(cudaGraphNodeType type)
{
    switch (type) {
    case cudaGraphNodeTypeKernel:      return "kernel";
    case cudaGraphNodeTypeMemcpy:      return "memcpy";
    case cudaGraphNodeTypeMemset:      return "memset";
    case cudaGraphNodeTypeHost:        return "host";
    case cudaGraphNodeTypeGraph:       return "graph";
    case cudaGraphNodeTypeEmpty:       return "empty";
    case cudaGraphNodeTypeWaitEvent:   return "event wait";
    case cudaGraphNodeTypeEventRecord: return "event record";
    case cudaGraphNodeTypeMemAlloc:    return "mem alloc";
    case cudaGraphNodeTypeMemFree:     return "mem free";
    case cudaGraphNodeTypeConditional: return "conditional";
    default:                           return "unknown";
    }
}

void print_dot(std::ostream &out, cudaGraph_t graph, const std::string &prefix)
{
    std::map<cudaGraphNode_t, size_t> id;
    for (const auto &n : graph->nodes)
        id[n.get()] = id.size();
    for (const auto &n : graph->nodes) {
        out << "  \"" << prefix << id[n.get()] << "\" [label=\"" << type_name(n->type) << "\"];\n";
        for (cudaGraphNode_t d : n->deps)
            out << "  \"" << prefix << id[d] << "\" -> \"" << prefix << id[n.get()] << "\";\n";
        if (n->child != nullptr)
            print_dot(out, n->child, prefix + std::to_string(id[n.get()]) + ".");
        for (size_t b = 0; b < n->bodies.size(); ++b)
            print_dot(out, n->bodies[b], prefix + std::to_string(id[n.get()]) + "." + std::to_string(b) + ".");
    }
}

template <typename T>
cudaError_t list(const std::vector<T> &items, T *out, size_t *count)
{
    if (count == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    if (out == nullptr) {
        *count = items.size();
        return cudaSuccess;
    }
    for (size_t i = 0; i < *count; ++i)
        out[i] = i < items.size() ? items[i] : nullptr;
    *count = std::min(*count, items.size());
    return cudaSuccess;
}

}

namespace cuda4cpu::detail {

//
// Capture of the work issued to a stream (called from the headers)
//

cudaError_t capture_kernel(cudaStream_t stream, const launch_conf &conf, std::unique_ptr<bound_kernel> kernel,
                           bool cooperative)
{
    if (kernel == nullptr)
        return record_error(cudaErrorInvalidValue);
    auto node = new_node(cudaGraphNodeTypeKernel);
    node->kernel_params.func           = const_cast<void *>(kernel->function);
    node->kernel_params.cuda4cpu_gridDim        = conf.grid;
    node->kernel_params.cuda4cpu_blockDim       = conf.block;
    node->kernel_params.sharedMemBytes = unsigned(conf.shared_mem);
    node->kernel_params.kernelParams   = kernel->params();
    node->kernel      = std::move(kernel);
    node->cooperative = cooperative;
    return capture(stream, std::move(node));
}

cudaError_t capture_copy(cudaStream_t stream, const cudaMemcpy3DParms &params)
{
    auto node  = new_node(cudaGraphNodeTypeMemcpy);
    node->copy = params;
    return capture(stream, std::move(node));
}

cudaError_t capture_fill(cudaStream_t stream, const cudaMemsetParams &params)
{
    auto node  = new_node(cudaGraphNodeTypeMemset);
    node->fill = params;
    return capture(stream, std::move(node));
}

cudaError_t capture_host(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
    if (fn == nullptr)
        return record_error(cudaErrorInvalidValue);
    auto node  = new_node(cudaGraphNodeTypeHost);
    node->host = {fn, userData};
    return capture(stream, std::move(node));
}

cudaError_t capture_alloc(cudaStream_t stream, void **ptr, size_t size)
{
    if (ptr == nullptr || size == 0)
        return record_error(cudaErrorInvalidValue);
    cudaMemAllocNodeParams params{};
    params.bytesize = size;
    cudaGraphNode_t node = nullptr;
    detail::stream_capture &c = *stream->capture;
    if (cudaError_t err = cudaGraphAddMemAllocNode(&node, c.session->graph, c.deps.data(), c.deps.size(), &params))
        return err;
    c.deps.assign(1, node);
    *ptr = params.dptr;
    return cudaSuccess;
}

cudaError_t capture_free(cudaStream_t stream, void *ptr)
{
    cudaGraphNode_t node = nullptr;
    detail::stream_capture &c = *stream->capture;
    if (cudaError_t err = cudaGraphAddMemFreeNode(&node, c.session->graph, c.deps.data(), c.deps.size(), ptr))
        return err;
    c.deps.assign(1, node);
    return cudaSuccess;
}

cudaError_t capture_event_record(cudaEvent_t event, cudaStream_t stream)
{
    if (event == nullptr)
        return record_error(cudaErrorInvalidResourceHandle);
    event->stream       = stream;
    event->capture      = stream->capture->session;
    event->capture_deps = stream->capture->deps;
    return cudaSuccess;
}

cudaError_t capture_wait_event(cudaStream_t stream, cudaEvent_t event)
{
    const std::shared_ptr<capture_session> session = event->capture;
    if (session == nullptr || !session->active)
        return cudaSuccess;   // the event follows work that already ran
    if (stream == nullptr)
        return record_error(cudaErrorStreamCaptureImplicit);

    if (!capturing(stream)) {
        // stream joins the capture, after the event's work
        stream->capture = new stream_capture{session, event->capture_deps};
        session->streams.push_back(stream);
        return cudaSuccess;
    }
    if (stream->capture->session != session)
        return record_error(cudaErrorStreamCaptureIsolation);
    auto &deps = stream->capture->deps;
    for (cudaGraphNode_t d : event->capture_deps) {
        if (std::find(deps.begin(), deps.end(), d) == deps.end())
            deps.push_back(d);
    }
    return cudaSuccess;
}

void abandon_capture(cudaStream_t stream)
{
    const std::shared_ptr<capture_session> session = stream->capture->session;
    if (session->origin == stream) {
        end_session(*session);
        if (session->owns_graph)
            delete session->graph;
        return;
    }
    std::erase(session->streams, stream);
    delete stream->capture;
    stream->capture = nullptr;
}

}

namespace cuda4cpu::cuda_api {

//
// Graphs and nodes
//

cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int /* flags */)
{
    if (pGraph == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *pGraph = new cudaGraph__;
    return cudaSuccess;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph)
{
    if (graph == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    delete graph;
    return cudaSuccess;
}

cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph)
{
    if (pGraphClone == nullptr || originalGraph == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *pGraphClone = clone(originalGraph);
    return cudaSuccess;
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                   const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                   const cudaKernelNodeParams *pNodeParams)
{
    if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
        return err;
    if (pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    auto node = new_node(cudaGraphNodeTypeKernel);
    if (cudaError_t err = bind_kernel(*pNodeParams, node->kernel, node->kernel_params))
        return err;
    return add_node(pGraphNode, graph, pDependencies, numDependencies, std::move(node));
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams *pNodeParams)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeKernel || pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *pNodeParams = node->kernel_params;
    return cudaSuccess;
}

cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams *pNodeParams)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeKernel || pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    return bind_kernel(*pNodeParams, node->kernel, node->kernel_params);
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                   const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                   const cudaMemcpy3DParms *pCopyParams)
{
    if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
        return err;
    if (pCopyParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    if (pCopyParams->srcArray != nullptr || pCopyParams->dstArray != nullptr)
        return detail::record_error(cudaErrorNotSupported);
    auto node  = new_node(cudaGraphNodeTypeMemcpy);
    node->copy = *pCopyParams;
    return add_node(pGraphNode, graph, pDependencies, numDependencies, std::move(node));
}

cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                     const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dst,
                                     const void *src, size_t count, cudaMemcpyKind kind)
{
    const cudaMemcpy3DParms p = detail::linear_copy(dst, src, count, kind);
    return cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, &p);
}

cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms *pNodeParams)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeMemcpy || pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *pNodeParams = node->copy;
    return cudaSuccess;
}

cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms *pNodeParams)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeMemcpy || pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    node->copy = *pNodeParams;
    return cudaSuccess;
}

cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void *dst, const void *src, size_t count,
                                           cudaMemcpyKind kind)
{
    const cudaMemcpy3DParms p = detail::linear_copy(dst, src, count, kind);
    return cudaGraphMemcpyNodeSetParams(node, &p);
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                   const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                   const cudaMemsetParams *pMemsetParams)
{
    if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
        return err;
    if (pMemsetParams == nullptr ||
        (pMemsetParams->elementSize != 1 && pMemsetParams->elementSize != 2 && pMemsetParams->elementSize != 4))
        return detail::record_error(cudaErrorInvalidValue);
    auto node  = new_node(cudaGraphNodeTypeMemset);
    node->fill = *pMemsetParams;
    return add_node(pGraphNode, graph, pDependencies, numDependencies, std::move(node));
}

cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams *pNodeParams)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeMemset || pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *pNodeParams = node->fill;
    return cudaSuccess;
}

cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams *pNodeParams)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeMemset || pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    node->fill = *pNodeParams;
    return cudaSuccess;
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                 const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                 const cudaHostNodeParams *pNodeParams)
{
    if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
        return err;
    if (pNodeParams == nullptr || pNodeParams->fn == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    auto node  = new_node(cudaGraphNodeTypeHost);
    node->host = *pNodeParams;
    return add_node(pGraphNode, graph, pDependencies, numDependencies, std::move(node));
}

cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams *pNodeParams)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeHost || pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *pNodeParams = node->host;
    return cudaSuccess;
}

cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams *pNodeParams)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeHost || pNodeParams == nullptr ||
        pNodeParams->fn == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    node->host = *pNodeParams;
    return cudaSuccess;
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                  const cudaGraphNode_t *pDependencies, size_t numDependencies)
{
    if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
        return err;
    return add_node(pGraphNode, graph, pDependencies, numDependencies, new_node(cudaGraphNodeTypeEmpty));
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                       const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                       cudaGraph_t childGraph)
{
    if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
        return err;
    if (childGraph == nullptr || childGraph == graph)
        return detail::record_error(cudaErrorInvalidValue);
    auto node   = new_node(cudaGraphNodeTypeGraph);
    node->child = clone(childGraph);
    return add_node(pGraphNode, graph, pDependencies, numDependencies, std::move(node));
}

cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeGraph || pGraph == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *pGraph = node->child;
    return cudaSuccess;
}

cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                        const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                        cudaEvent_t event)
{
    if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
        return err;
    if (event == nullptr)
        return detail::record_error(cudaErrorInvalidResourceHandle);
    auto node   = new_node(cudaGraphNodeTypeEventRecord);
    node->event = event;
    return add_node(pGraphNode, graph, pDependencies, numDependencies, std::move(node));
}

cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                      const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                      cudaEvent_t event)
{
    if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
        return err;
    if (event == nullptr)
        return detail::record_error(cudaErrorInvalidResourceHandle);
    auto node   = new_node(cudaGraphNodeTypeWaitEvent);
    node->event = event;
    return add_node(pGraphNode, graph, pDependencies, numDependencies, std::move(node));
}

cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeEventRecord || event_out == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *event_out = node->event;
    return cudaSuccess;
}

cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeWaitEvent || event_out == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *event_out = node->event;
    return cudaSuccess;
}

cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                     const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                     cudaMemAllocNodeParams *nodeParams)
{
    if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
        return err;
    if (nodeParams == nullptr || nodeParams->bytesize == 0)
        return detail::record_error(cudaErrorInvalidValue);
    auto node        = new_node(cudaGraphNodeTypeMemAlloc);
    node->allocation = detail::new_allocation(nodeParams->bytesize);
    if (node->allocation == nullptr)
        return detail::record_error(cudaErrorMemoryAllocation);
    graph->allocations.push_back(node->allocation);
    nodeParams->dptr   = node->allocation->ptr;
    node->alloc_params = *nodeParams;
    return add_node(pGraphNode, graph, pDependencies, numDependencies, std::move(node));
}

cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams *params_out)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeMemAlloc || params_out == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *params_out = node->alloc_params;
    return cudaSuccess;
}

cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                    const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dptr)
{
    if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
        return err;
    if (dptr == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    auto node        = new_node(cudaGraphNodeTypeMemFree);
    node->allocation = detail::find_allocation(dptr, graph->allocations);   // keeps it mapped
    node->free_ptr   = dptr;
    return add_node(pGraphNode, graph, pDependencies, numDependencies, std::move(node));
}

cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void *dptr_out)
{
    if (node == nullptr || node->type != cudaGraphNodeTypeMemFree || dptr_out == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *static_cast<void **>(dptr_out) = node->free_ptr;
    return cudaSuccess;
}

cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle *pHandle_out, cudaGraph_t graph,
                                             unsigned int defaultLaunchValue, unsigned int flags)
{
    if (pHandle_out == nullptr || graph == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    auto condition = std::make_shared<detail::graph_condition>();
    condition->value.store(defaultLaunchValue, std::memory_order_relaxed);
    condition->default_value  = defaultLaunchValue;
    condition->assign_default = (flags & cudaGraphCondAssignDefault) != 0;
    *pHandle_out = reinterpret_cast<cudaGraphConditionalHandle>(condition.get());
    graph->conditions.push_back(std::move(condition));
    return cudaSuccess;
}

cudaError_t cudaGraphAddNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies,
                             size_t numDependencies, cudaGraphNodeParams *nodeParams)
{
    if (nodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    cudaGraphNodeParams &p = *nodeParams;
    switch (p.type) {
    case cudaGraphNodeTypeKernel:
        return cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, &p.kernel);
    case cudaGraphNodeTypeMemcpy:
        return cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, &p.memcpy.copyParams);
    case cudaGraphNodeTypeMemset:
        return cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, &p.memset);
    case cudaGraphNodeTypeHost:
        return cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, &p.host);
    case cudaGraphNodeTypeGraph:
        return cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, p.graph.graph);
    case cudaGraphNodeTypeEmpty:
        return cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies);
    case cudaGraphNodeTypeWaitEvent:
        return cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, p.eventWait.event);
    case cudaGraphNodeTypeEventRecord:
        return cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies,
                                           p.eventRecord.event);
    case cudaGraphNodeTypeMemAlloc:
        return cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, &p.alloc);
    case cudaGraphNodeTypeMemFree:
        return cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, p.free.dptr);
    case cudaGraphNodeTypeConditional: {
        if (cudaError_t err = check_add(pGraphNode, graph, pDependencies, numDependencies))
            return err;
        cudaConditionalNodeParams &c = p.conditional;
        auto *condition = reinterpret_cast<detail::graph_condition *>(c.handle);
        auto it = std::find_if(graph->conditions.begin(), graph->conditions.end(),
                               [&](const auto &h) { return h.get() == condition; });
        const bool valid_size = c.type == cudaGraphCondTypeIf      ? c.size == 1 || c.size == 2
                                : c.type == cudaGraphCondTypeWhile ? c.size == 1
                                                                   : c.size >= 1;
        if (it == graph->conditions.end() || !valid_size)
            return detail::record_error(cudaErrorInvalidValue);
        auto node            = new_node(cudaGraphNodeTypeConditional);
        node->condition      = *it;
        node->condition_type = c.type;
        for (unsigned i = 0; i < c.size; ++i) {
            auto *body       = new cudaGraph__;
            body->conditions  = graph->conditions;    // kernels in the body may set the graph's conditions
            body->allocations = graph->allocations;   // and free its allocations
            node->bodies.push_back(body);
        }
        c.phGraph_out = node->bodies.data();
        return add_node(pGraphNode, graph, pDependencies, numDependencies, std::move(node));
    }
    default:
        return detail::record_error(cudaErrorNotSupported);
    }
}

cudaError_t cudaGraphAddNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies,
                             const cudaGraphEdgeData * /* dependencyData */, size_t numDependencies,
                             cudaGraphNodeParams *nodeParams)
{
    return cudaGraphAddNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to,
                                     size_t numDependencies)
{
    if (graph == nullptr || (numDependencies > 0 && (from == nullptr || to == nullptr)))
        return detail::record_error(cudaErrorInvalidValue);
    for (size_t i = 0; i < numDependencies; ++i) {
        if (!owns(graph, from[i]) || !owns(graph, to[i]) || from[i] == to[i])
            return detail::record_error(cudaErrorInvalidValue);
    }
    for (size_t i = 0; i < numDependencies; ++i) {
        auto &deps = to[i]->deps;
        if (std::find(deps.begin(), deps.end(), from[i]) != deps.end())
            continue;
        deps.push_back(from[i]);
        from[i]->dependents.push_back(to[i]);
    }
    return cudaSuccess;
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to,
                                     const cudaGraphEdgeData * /* edgeData */, size_t numDependencies)
{
    return cudaGraphAddDependencies(graph, from, to, numDependencies);
}

cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to,
                                        size_t numDependencies)
{
    if (graph == nullptr || (numDependencies > 0 && (from == nullptr || to == nullptr)))
        return detail::record_error(cudaErrorInvalidValue);
    for (size_t i = 0; i < numDependencies; ++i) {
        if (!owns(graph, from[i]) || !owns(graph, to[i]) ||
            std::find(to[i]->deps.begin(), to[i]->deps.end(), from[i]) == to[i]->deps.end())
            return detail::record_error(cudaErrorInvalidValue);
    }
    for (size_t i = 0; i < numDependencies; ++i) {
        std::erase(to[i]->deps, from[i]);
        std::erase(from[i]->dependents, to[i]);
    }
    return cudaSuccess;
}

cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node)
{
    if (node == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    for (cudaGraphNode_t d : node->deps)
        std::erase(d->dependents, node);
    for (cudaGraphNode_t d : node->dependents)
        std::erase(d->deps, node);
    std::erase_if(node->graph->nodes, [&](const auto &n) { return n.get() == node; });
    return cudaSuccess;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes)
{
    if (graph == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    std::vector<cudaGraphNode_t> all;
    for (const auto &n : graph->nodes)
        all.push_back(n.get());
    return list(all, nodes, numNodes);
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes)
{
    if (graph == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    std::vector<cudaGraphNode_t> roots;
    for (const auto &n : graph->nodes) {
        if (n->deps.empty())
            roots.push_back(n.get());
    }
    return list(roots, pRootNodes, pNumRootNodes);
}

cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges)
{
    if (graph == nullptr || numEdges == nullptr || (from == nullptr) != (to == nullptr))
        return detail::record_error(cudaErrorInvalidValue);
    std::vector<cudaGraphNode_t> sources, targets;
    for (const auto &n : graph->nodes) {
        for (cudaGraphNode_t d : n->deps) {
            sources.push_back(d);
            targets.push_back(n.get());
        }
    }
    size_t count = *numEdges;
    list(sources, from, &count);
    return list(targets, to, numEdges);
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType *pType)
{
    if (node == nullptr || pType == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *pType = node->type;
    return cudaSuccess;
}

cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies,
                                         size_t *pNumDependencies)
{
    if (node == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    return list(node->deps, pDependencies, pNumDependencies);
}

cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes,
                                           size_t *pNumDependentNodes)
{
    if (node == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    return list(node->dependents, pDependentNodes, pNumDependentNodes);
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int /* flags */)
{
    if (graph == nullptr || path == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    std::ofstream out(path);
    if (!out)
        return detail::record_error(cudaErrorInvalidValue);
    out << "digraph cuda4cpu {\n";
    print_dot(out, graph, "");
    out << "}\n";
    return cudaSuccess;
}

//
// Executable graphs
//

cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long /* flags */)
{
    if (pGraphExec == nullptr || graph == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    std::unique_ptr<cudaGraphExec__> exec;
    if (cudaError_t err = instantiate(graph, exec))
        return err;
    *pGraphExec = exec.release();
    return cudaSuccess;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode,
                                 char *pLogBuffer, size_t bufferSize)
{
    if (pErrorNode != nullptr)
        *pErrorNode = nullptr;
    if (pLogBuffer != nullptr && bufferSize > 0)
        pLogBuffer[0] = '\0';
    return cudaGraphInstantiate(pGraphExec, graph, 0ull);
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags)
{
    return cudaGraphInstantiate(pGraphExec, graph, flags);
}

cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                           cudaGraphInstantiateParams *instantiateParams)
{
    if (instantiateParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    const cudaError_t err = cudaGraphInstantiate(pGraphExec, graph, instantiateParams->flags);
    instantiateParams->errNode_out = nullptr;
    instantiateParams->result_out  = err == cudaSuccess ? cudaGraphInstantiateSuccess : cudaGraphInstantiateError;
    return err;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec)
{
    if (graphExec == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    delete graphExec;
    return cudaSuccess;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    if (graphExec == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    if (detail::capturing(stream))
        return detail::record_error(cudaErrorStreamCaptureUnsupported);
    for (const auto &c : graphExec->conditions) {
        if (c->assign_default)
            c->value.store(c->default_value, std::memory_order_relaxed);
    }
    if (cudaError_t err = run(*graphExec))
        return detail::record_error(err);
    return cudaSuccess;
}

cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t /* stream */)
{
    return graphExec == nullptr ? detail::record_error(cudaErrorInvalidValue) : cudaSuccess;
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                cudaGraphExecUpdateResultInfo *resultInfo)
{
    if (hGraphExec == nullptr || hGraph == nullptr || resultInfo == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *resultInfo = {cudaGraphExecUpdateSuccess, nullptr, nullptr};

    std::unique_ptr<cudaGraphExec__> fresh;
    if (cudaError_t err = instantiate(hGraph, fresh))
        return err;
    auto fail = [&](cudaGraphExecUpdateResult result, cudaGraphNode_t node) {
        *resultInfo = {result, node, nullptr};
        return detail::record_error(cudaErrorGraphExecUpdateFailure);
    };
    if (fresh->nodes.size() != hGraphExec->nodes.size())
        return fail(cudaGraphExecUpdateErrorTopologyChanged, nullptr);
    for (size_t i = 0; i < fresh->nodes.size(); ++i) {
        const exec_node &a = hGraphExec->nodes[i], &b = fresh->nodes[i];
        if (a.type != b.type)
            return fail(cudaGraphExecUpdateErrorNodeTypeChanged, b.origin);
        if (a.deps != b.deps)
            return fail(cudaGraphExecUpdateErrorTopologyChanged, b.origin);
    }
    // The nodes keep their identity: the nodes of the graph they came from
    for (size_t i = 0; i < fresh->nodes.size(); ++i)
        fresh->nodes[i].origin = hGraphExec->nodes[i].origin;
    *hGraphExec = std::move(*fresh);
    return cudaSuccess;
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out,
                                cudaGraphExecUpdateResult *updateResult_out)
{
    cudaGraphExecUpdateResultInfo info{};
    const cudaError_t err = cudaGraphExecUpdate(hGraphExec, hGraph, &info);
    if (hErrorNode_out != nullptr)
        *hErrorNode_out = info.errorNode;
    if (updateResult_out != nullptr)
        *updateResult_out = err == cudaSuccess || err == cudaErrorGraphExecUpdateFailure ? info.result
                                                                                         : cudaGraphExecUpdateError;
    return err;
}

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                             const cudaKernelNodeParams *pNodeParams)
{
    exec_node *e = hGraphExec == nullptr ? nullptr : find_exec_node(hGraphExec, node);
    if (e == nullptr || e->type != cudaGraphNodeTypeKernel || pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    cudaKernelNodeParams stored{};
    if (cudaError_t err = bind_kernel(*pNodeParams, e->kernel, stored))
        return err;
    e->conf = launch_conf{stored.cuda4cpu_gridDim, stored.cuda4cpu_blockDim, stored.sharedMemBytes};
    return cudaSuccess;
}

cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                             const cudaMemcpy3DParms *pNodeParams)
{
    exec_node *e = hGraphExec == nullptr ? nullptr : find_exec_node(hGraphExec, node);
    if (e == nullptr || e->type != cudaGraphNodeTypeMemcpy || pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    e->copy = *pNodeParams;
    return cudaSuccess;
}

cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void *dst,
                                               const void *src, size_t count, cudaMemcpyKind kind)
{
    const cudaMemcpy3DParms p = detail::linear_copy(dst, src, count, kind);
    return cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, &p);
}

cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                             const cudaMemsetParams *pNodeParams)
{
    exec_node *e = hGraphExec == nullptr ? nullptr : find_exec_node(hGraphExec, node);
    if (e == nullptr || e->type != cudaGraphNodeTypeMemset || pNodeParams == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    e->fill = *pNodeParams;
    return cudaSuccess;
}

cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                           const cudaHostNodeParams *pNodeParams)
{
    exec_node *e = hGraphExec == nullptr ? nullptr : find_exec_node(hGraphExec, node);
    if (e == nullptr || e->type != cudaGraphNodeTypeHost || pNodeParams == nullptr || pNodeParams->fn == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    e->host = *pNodeParams;
    return cudaSuccess;
}

cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                                 cudaGraph_t childGraph)
{
    exec_node *e = hGraphExec == nullptr ? nullptr : find_exec_node(hGraphExec, node);
    if (e == nullptr || e->type != cudaGraphNodeTypeGraph || childGraph == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    std::unique_ptr<cudaGraphExec__> child;
    if (cudaError_t err = instantiate(childGraph, child))
        return err;
    e->child = std::move(child);
    return cudaSuccess;
}

//
// Memory of graph allocation nodes
//

cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void *value)
{
    if (device != 0)
        return detail::record_error(cudaErrorInvalidDevice);
    if (value == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    auto &out = *static_cast<uint64_t *>(value);
    switch (attr) {
    case cudaGraphMemAttrUsedMemCurrent:     out = detail::used_current.load(); break;
    case cudaGraphMemAttrUsedMemHigh:        out = detail::used_high.load(); break;
    case cudaGraphMemAttrReservedMemCurrent: out = detail::reserved_current.load(); break;
    case cudaGraphMemAttrReservedMemHigh:    out = detail::reserved_high.load(); break;
    default:                                 return detail::record_error(cudaErrorInvalidValue);
    }
    return cudaSuccess;
}

//! Only the high watermarks can be set, to 0, which resets them
cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void *value)
{
    if (device != 0)
        return detail::record_error(cudaErrorInvalidDevice);
    if (value == nullptr || *static_cast<const uint64_t *>(value) != 0)
        return detail::record_error(cudaErrorInvalidValue);
    switch (attr) {
    case cudaGraphMemAttrUsedMemHigh:     detail::used_high.store(detail::used_current.load()); break;
    case cudaGraphMemAttrReservedMemHigh: detail::reserved_high.store(detail::reserved_current.load()); break;
    default:                              return detail::record_error(cudaErrorInvalidValue);
    }
    return cudaSuccess;
}

cudaError_t cudaDeviceGraphMemTrim(int device)
{
    return device == 0 ? cudaSuccess : detail::record_error(cudaErrorInvalidDevice);
}

//
// Stream capture
//

cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph,
                                          const cudaGraphNode_t *dependencies,
                                          const cudaGraphEdgeData * /* dependencyData */, size_t numDependencies,
                                          cudaStreamCaptureMode /* mode */)
{
    if (stream == nullptr)
        return detail::record_error(cudaErrorStreamCaptureUnsupported);   // the legacy default stream
    if (detail::capturing(stream))
        return detail::record_error(cudaErrorIllegalState);
    if (graph == nullptr || (numDependencies > 0 && dependencies == nullptr))
        return detail::record_error(cudaErrorInvalidValue);
    auto session        = std::make_shared<detail::capture_session>();
    session->graph      = graph;
    session->owns_graph = false;
    session->id         = next_capture_id++;
    session->origin     = stream;
    session->streams.push_back(stream);
    stream->capture = new detail::stream_capture{session, {dependencies, dependencies + numDependencies}};
    return cudaSuccess;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
{
    if (stream == nullptr)
        return detail::record_error(cudaErrorStreamCaptureUnsupported);
    if (detail::capturing(stream))
        return detail::record_error(cudaErrorIllegalState);
    auto *graph = new cudaGraph__;
    cudaStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0, mode);
    stream->capture->session->owns_graph = true;
    return cudaSuccess;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
{
    if (!detail::capturing(stream))
        return detail::record_error(cudaErrorIllegalState);
    const std::shared_ptr<detail::capture_session> session = stream->capture->session;
    if (session->origin != stream)
        return detail::record_error(cudaErrorStreamCaptureUnmatched);
    end_session(*session);
    if (pGraph != nullptr)
        *pGraph = session->graph;
    else if (session->owns_graph)
        return detail::record_error(cudaErrorInvalidValue);
    return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus)
{
    if (pCaptureStatus == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *pCaptureStatus = detail::capturing(stream) ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus *captureStatus_out,
                                     unsigned long long *id_out, cudaGraph_t *graph_out,
                                     const cudaGraphNode_t **dependencies_out, size_t *numDependencies_out)
{
    if (captureStatus_out == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    const bool active = detail::capturing(stream);
    *captureStatus_out = active ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
    if (id_out != nullptr)
        *id_out = active ? stream->capture->session->id : 0;
    if (graph_out != nullptr)
        *graph_out = active ? stream->capture->session->graph : nullptr;
    if (dependencies_out != nullptr)
        *dependencies_out = active ? stream->capture->deps.data() : nullptr;
    if (numDependencies_out != nullptr)
        *numDependencies_out = active ? stream->capture->deps.size() : 0;
    return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus *captureStatus_out,
                                     unsigned long long *id_out, cudaGraph_t *graph_out,
                                     const cudaGraphNode_t **dependencies_out,
                                     const cudaGraphEdgeData **edgeData_out, size_t *numDependencies_out)
{
    if (edgeData_out != nullptr)
        *edgeData_out = nullptr;
    return cudaStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out, dependencies_out,
                                    numDependencies_out);
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies,
                                                size_t numDependencies, unsigned int flags)
{
    if (!detail::capturing(stream))
        return detail::record_error(cudaErrorIllegalState);
    if ((numDependencies > 0 && dependencies == nullptr) || flags > cudaStreamSetCaptureDependencies)
        return detail::record_error(cudaErrorInvalidValue);
    auto &deps = stream->capture->deps;
    if (flags == cudaStreamSetCaptureDependencies)
        deps.clear();
    for (size_t i = 0; i < numDependencies; ++i) {
        if (std::find(deps.begin(), deps.end(), dependencies[i]) == deps.end())
            deps.push_back(dependencies[i]);
    }
    return cudaSuccess;
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies,
                                                const cudaGraphEdgeData * /* dependencyData */,
                                                size_t numDependencies, unsigned int flags)
{
    return cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);
}

cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode *mode)
{
    if (mode == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    std::swap(*mode, thread_capture_mode);
    return cudaSuccess;
}

}
