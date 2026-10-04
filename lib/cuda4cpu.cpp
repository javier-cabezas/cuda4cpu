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

#include <cstdlib>

#ifdef CUDA4CPU_HAVE_NUMA
#include <numa.h>
#endif

#include "cuda4cpu.hpp"

namespace cuda4cpu {

// static variables
system system::sys_;
thread_local thread_block *
thread_block::Current_ = nullptr;

system::system()
{
    cpus_ = omp_get_num_procs();
    omp_set_num_threads(cpus_);

#ifdef CUDA4CPU_HAVE_NUMA
    if (numa_available() < 0) {
        init_single_node();
        return;
    }

    nodes_ = numa_num_configured_nodes();

    for (int i = 0; i < nodes_; ++i) {
        bitmask *mask = numa_allocate_cpumask();

        numa_node_to_cpus(i, mask);
        std::vector<int> node2cpus;
        for (int j = 0; j < cpus_; ++j) {
            if (numa_bitmask_isbitset(mask, j)) {
                cpu2node_[j] = i;
                node2cpus.push_back(j);
            }
        }
        node2cpus_[i] = node2cpus;

        numa_free_cpumask(mask);
    }
#else
    init_single_node();
#endif
}

void system::init_single_node()
{
    nodes_ = 1;

    std::vector<int> node2cpus;
    for (int j = 0; j < cpus_; ++j) {
        cpu2node_[j] = 0;
        node2cpus.push_back(j);
    }
    node2cpus_[0] = node2cpus;
}

}
