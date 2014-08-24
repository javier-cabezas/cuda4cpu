/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas <javier.cabezas@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
 */

#include <cstdlib>

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

    int ret = numa_available();
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
}

}
