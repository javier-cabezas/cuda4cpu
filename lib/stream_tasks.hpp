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

// Work that the libraries (cuBLAS, cuFFT, ...) run in a stream's order: now,
// since kernels and copies run when they're issued, or, while the stream is
// being captured, as a host node of the graph, which owns it (and so do the
// executable graphs made from it: lib/graphs.cpp's capture_host_owned).
#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "cuda_runtime.h"

namespace cuda4cpu::detail {

cudaError_t capture_host_owned(cudaStream_t stream, cudaHostFn_t fn, std::shared_ptr<void> data);

template <typename F>
cudaError_t run_on_stream(cudaStream_t stream, F &&f)
{
    if (capturing(stream)) {
        auto job = std::make_shared<std::function<void()>>(std::forward<F>(f));
        auto call = [](void *data) { (*static_cast<std::function<void()> *>(data))(); };
        return capture_host_owned(stream, call, job);
    }
    f();
    return cudaSuccess;
}

} // namespace cuda4cpu::detail
