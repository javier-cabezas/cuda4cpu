/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
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


#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "cuda4cpu.hpp"

using namespace cuda4cpu;

int main()
{
    using namespace std::chrono_literals;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, nullptr);
    std::this_thread::sleep_for(50ms);
    cudaEventRecord(end, nullptr);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // cudaEventElapsedTime reports milliseconds
    if (ms < 50.f || ms > 5000.f) {
        std::cout << "cudaEventElapsedTime: expected ~50ms, got " << ms << "ms\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
