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


//
// Helpers shared by the CUDA samples: timing and result checking
//

#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>

namespace sample {

//! Runs f once and returns the elapsed wall-clock time in milliseconds
template <typename F>
double time_ms(F &&f)
{
    auto start = std::chrono::steady_clock::now();
    f();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline void report(const char *name, double cuda_ms, double host_ms)
{
    std::cout << name << ": cuda4cpu " << cuda_ms << " ms, host reference " << host_ms << " ms\n";
}

//! Counts the elements of got that differ from expected by more than a relative tolerance
template <typename T, typename U>
size_t count_mismatches(const T *got, const U *expected, size_t n, double rel_tol = 0.0)
{
    size_t errors = 0;
    for (size_t i = 0; i < n; ++i) {
        double g = double(got[i]), e = double(expected[i]);
        if (std::abs(g - e) > rel_tol * std::max(1.0, std::abs(e))) {
            if (errors++ < 10)
                std::cout << "  [" << i << "] = " << g << ", expected " << e << "\n";
        }
    }
    return errors;
}

inline int finish(size_t errors)
{
    if (errors > 0) {
        std::cout << errors << " mismatches\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

}
