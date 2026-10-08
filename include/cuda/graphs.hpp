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


// CUDA graphs: the types and functions of the runtime API. Graphs are built
// explicitly or by stream capture, and an instantiated graph runs its nodes
// one after the other, in an order that respects their dependencies. The
// functions are defined in the library (lib/graphs.cpp); kernel nodes are
// bound to their kernels in launch.hpp.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "error.hpp"
#include "memory.hpp"
#include "streams.hpp"
#include "types.hpp"

namespace cuda4cpu {

namespace detail {

//! The value behind a cudaGraphConditionalHandle
struct graph_condition {
    std::atomic<unsigned> value{0};
    unsigned default_value = 0;
    bool assign_default    = false;   //!< Set to default_value when the graph launches
};

}

inline namespace cuda_api {

enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel      = 0x00,
    cudaGraphNodeTypeMemcpy      = 0x01,
    cudaGraphNodeTypeMemset      = 0x02,
    cudaGraphNodeTypeHost        = 0x03,
    cudaGraphNodeTypeGraph       = 0x04,
    cudaGraphNodeTypeEmpty       = 0x05,
    cudaGraphNodeTypeWaitEvent   = 0x06,
    cudaGraphNodeTypeEventRecord = 0x07,
    cudaGraphNodeTypeExtSemaphoreSignal = 0x08,
    cudaGraphNodeTypeExtSemaphoreWait   = 0x09,
    cudaGraphNodeTypeMemAlloc    = 0x0a,
    cudaGraphNodeTypeMemFree     = 0x0b,
    cudaGraphNodeTypeConditional = 0x0d,
    cudaGraphNodeTypeCount
};

enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal      = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed     = 2
};

enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone        = 0,
    cudaStreamCaptureStatusActive      = 1,
    cudaStreamCaptureStatusInvalidated = 2
};

enum cudaStreamUpdateCaptureDependenciesFlags {
    cudaStreamAddCaptureDependencies = 0x0,
    cudaStreamSetCaptureDependencies = 0x1
};

enum cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess                     = 0x0,
    cudaGraphExecUpdateError                       = 0x1,
    cudaGraphExecUpdateErrorTopologyChanged        = 0x2,
    cudaGraphExecUpdateErrorNodeTypeChanged        = 0x3,
    cudaGraphExecUpdateErrorFunctionChanged        = 0x4,
    cudaGraphExecUpdateErrorParametersChanged      = 0x5,
    cudaGraphExecUpdateErrorNotSupported           = 0x6,
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 0x7,
    cudaGraphExecUpdateErrorAttributesChanged      = 0x8
};

struct cudaGraphExecUpdateResultInfo {
    cudaGraphExecUpdateResult result;
    cudaGraphNode_t errorNode;
    cudaGraphNode_t errorFromNode;
};

enum cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1,
    cudaGraphInstantiateFlagUpload           = 2,
    cudaGraphInstantiateFlagDeviceLaunch     = 4,
    cudaGraphInstantiateFlagUseNodePriority  = 8
};

enum cudaGraphInstantiateResult {
    cudaGraphInstantiateSuccess                       = 0,
    cudaGraphInstantiateError                         = 1,
    cudaGraphInstantiateInvalidStructure              = 2,
    cudaGraphInstantiateNodeOperationNotSupported     = 3,
    cudaGraphInstantiateMultipleDevicesNotSupported   = 4
};

struct cudaGraphInstantiateParams {
    unsigned long long flags;
    cudaStream_t uploadStream;
    cudaGraphNode_t errNode_out;
    cudaGraphInstantiateResult result_out;
};

enum cudaGraphMemAttributeType {
    cudaGraphMemAttrUsedMemCurrent     = 0,
    cudaGraphMemAttrUsedMemHigh        = 1,
    cudaGraphMemAttrReservedMemCurrent = 2,
    cudaGraphMemAttrReservedMemHigh    = 3
};

enum cudaGraphDebugDotFlags {
    cudaGraphDebugDotFlagsVerbose = 1 << 0
};

enum cudaGraphDependencyType {
    cudaGraphDependencyTypeDefault      = 0,
    cudaGraphDependencyTypeProgrammatic = 1
};

//! Edges are always default, full dependencies
struct cudaGraphEdgeData {
    unsigned char from_port;
    unsigned char to_port;
    unsigned char type;
    unsigned char reserved[5];
};

//! gridDim and blockDim are macros for the built-in variables, so the fields
//! have other names: cuda4cpu-rewrite turns params.gridDim into
//! params.cuda4cpu_gridDim (and likewise for blockDim)
struct cudaKernelNodeParams {
    void *func;
    dim3 cuda4cpu_gridDim;
    dim3 cuda4cpu_blockDim;
    unsigned int sharedMemBytes;
    void **kernelParams;   //!< One pointer per kernel parameter; the values are copied
    void **extra;          //!< Not supported
};
using cudaKernelNodeParamsV2 = cudaKernelNodeParams;

struct cudaHostNodeParams {
    cudaHostFn_t fn;
    void *userData;
};
using cudaHostNodeParamsV2 = cudaHostNodeParams;

using cudaMemsetParamsV2 = cudaMemsetParams;

struct cudaMemcpyNodeParams {
    int flags;
    int reserved[3];
    cudaMemcpy3DParms copyParams;
};

enum cudaMemAllocationType {
    cudaMemAllocationTypeInvalid = 0x0,
    cudaMemAllocationTypePinned  = 0x1,
    cudaMemAllocationTypeMax     = 0x7fffffff
};

enum cudaMemAllocationHandleType {
    cudaMemHandleTypeNone                = 0x0,
    cudaMemHandleTypePosixFileDescriptor = 0x1,
    cudaMemHandleTypeWin32               = 0x2,
    cudaMemHandleTypeWin32Kmt            = 0x4,
    cudaMemHandleTypeFabric              = 0x8
};

enum cudaMemAccessFlags {
    cudaMemAccessFlagsProtNone      = 0,
    cudaMemAccessFlagsProtRead      = 1,
    cudaMemAccessFlagsProtReadWrite = 3
};

struct cudaMemAccessDesc {
    cudaMemLocation location;
    cudaMemAccessFlags flags;
};

struct cudaMemPoolProps {
    cudaMemAllocationType allocType;
    cudaMemAllocationHandleType handleTypes;
    cudaMemLocation location;
    void *win32SecurityAttributes;
    size_t maxSize;
    unsigned short usage;
    unsigned char reserved[54];
};

struct cudaMemAllocNodeParams {
    cudaMemPoolProps poolProps;
    const cudaMemAccessDesc *accessDescs;
    size_t accessDescCount;
    size_t bytesize;
    void *dptr;   //!< Set when the node is added: the address is fixed from then on
};
using cudaMemAllocNodeParamsV2 = cudaMemAllocNodeParams;

struct cudaMemFreeNodeParams {
    void *dptr;
};

struct cudaChildGraphNodeParams {
    cudaGraph_t graph;
};

struct cudaEventRecordNodeParams {
    cudaEvent_t event;
};

struct cudaEventWaitNodeParams {
    cudaEvent_t event;
};

//
// Conditional nodes: kernels set the condition with cudaGraphSetConditional
//

using cudaGraphConditionalHandle = unsigned long long;

enum cudaGraphConditionalHandleFlags {
    cudaGraphCondAssignDefault = 1
};

enum cudaGraphConditionalNodeType {
    cudaGraphCondTypeIf     = 0,   //!< Body 0 if the condition is nonzero, else body 1 (if size is 2)
    cudaGraphCondTypeWhile  = 1,   //!< Body 0 while the condition is nonzero, checked before each run
    cudaGraphCondTypeSwitch = 2    //!< Body n for condition n, if n < size
};

struct cudaConditionalNodeParams {
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalNodeType type;
    unsigned int size;
    cudaGraph_t *phGraph_out;   //!< Set when the node is added: its empty body graphs, owned by the node
};

struct cudaGraphNodeParams {
    cudaGraphNodeType type;
    int reserved0[3];
    union {
        long long reserved1[29];
        cudaKernelNodeParamsV2 kernel;
        cudaMemcpyNodeParams memcpy;
        cudaMemsetParamsV2 memset;
        cudaHostNodeParamsV2 host;
        cudaChildGraphNodeParams graph;
        cudaEventWaitNodeParams eventWait;
        cudaEventRecordNodeParams eventRecord;
        cudaMemAllocNodeParamsV2 alloc;
        cudaMemFreeNodeParams free;
        cudaConditionalNodeParams conditional;
    };
    long long reserved2;
};

//! Called by a kernel to set the condition of a conditional node
inline void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value)
{
    reinterpret_cast<detail::graph_condition *>(handle)->value.store(value, std::memory_order_relaxed);
}

//
// Graphs and nodes
//

cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags = 0);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph);

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                   const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                   const cudaKernelNodeParams *pNodeParams);
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams *pNodeParams);
cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams *pNodeParams);

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                   const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                   const cudaMemcpy3DParms *pCopyParams);
cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                     const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dst,
                                     const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms *pNodeParams);
cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms *pNodeParams);
cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void *dst, const void *src, size_t count,
                                           cudaMemcpyKind kind);

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                   const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                   const cudaMemsetParams *pMemsetParams);
cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams *pNodeParams);
cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams *pNodeParams);

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                 const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                 const cudaHostNodeParams *pNodeParams);
cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams *pNodeParams);
cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams *pNodeParams);

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                  const cudaGraphNode_t *pDependencies, size_t numDependencies);

//! The child graph is cloned into the node
cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                       const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                       cudaGraph_t childGraph);
cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph);

cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                        const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                        cudaEvent_t event);
cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                      const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                      cudaEvent_t event);
cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out);
cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out);

//! The memory is allocated when the node is added, and nodeParams->dptr is
//! set; the graph's footprint counts it while it is allocated, from when the
//! node runs to when a free node or cudaFree frees it
cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                     const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                     cudaMemAllocNodeParams *nodeParams);
cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams *params_out);
cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                    const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dptr);
cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void *dptr_out);

cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle *pHandle_out, cudaGraph_t graph,
                                             unsigned int defaultLaunchValue = 0, unsigned int flags = 0);

//! Any node type, from a cudaGraphNodeParams
cudaError_t cudaGraphAddNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies,
                             size_t numDependencies, cudaGraphNodeParams *nodeParams);
cudaError_t cudaGraphAddNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies,
                             const cudaGraphEdgeData *dependencyData, size_t numDependencies,
                             cudaGraphNodeParams *nodeParams);

cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to,
                                     size_t numDependencies);
cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to,
                                     const cudaGraphEdgeData *edgeData, size_t numDependencies);
cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to,
                                        size_t numDependencies);
cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node);

// Queries. With a null array, they return the count only.
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes);
cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes);
cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType *pType);
cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies,
                                         size_t *pNumDependencies);
cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes,
                                           size_t *pNumDependentNodes);

//! Writes the graph in Graphviz format
cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags);

//
// Executable graphs
//

cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags = 0);
//! The form of CUDA 11
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode,
                                 char *pLogBuffer, size_t bufferSize);
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                          unsigned long long flags = 0);
cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                           cudaGraphInstantiateParams *instantiateParams);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);

//! Runs every node of the graph before returning, as kernels do
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);

//! Takes the parameters of a graph with the same topology
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                cudaGraphExecUpdateResultInfo *resultInfo);
//! The form of CUDA 11
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out,
                                cudaGraphExecUpdateResult *updateResult_out);

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                             const cudaKernelNodeParams *pNodeParams);
cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                             const cudaMemcpy3DParms *pNodeParams);
cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void *dst,
                                               const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                             const cudaMemsetParams *pNodeParams);
cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                           const cudaHostNodeParams *pNodeParams);
cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                                 cudaGraph_t childGraph);

//
// Memory of graph allocation nodes
//

cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void *value);
cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void *value);
cudaError_t cudaDeviceGraphMemTrim(int device);

//
// Stream capture
//

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal);
cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph,
                                          const cudaGraphNode_t *dependencies,
                                          const cudaGraphEdgeData *dependencyData, size_t numDependencies,
                                          cudaStreamCaptureMode mode);
//! pGraph may be null when the capture was into an existing graph
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus);
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus *captureStatus_out,
                                     unsigned long long *id_out = nullptr, cudaGraph_t *graph_out = nullptr,
                                     const cudaGraphNode_t **dependencies_out = nullptr,
                                     size_t *numDependencies_out = nullptr);
//! The form of CUDA 12.3, with edge data (always null)
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus *captureStatus_out,
                                     unsigned long long *id_out, cudaGraph_t *graph_out,
                                     const cudaGraphNode_t **dependencies_out,
                                     const cudaGraphEdgeData **edgeData_out, size_t *numDependencies_out);
cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies,
                                                size_t numDependencies, unsigned int flags = 0);
cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies,
                                                const cudaGraphEdgeData *dependencyData, size_t numDependencies,
                                                unsigned int flags = 0);
cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode *mode);

}

}
