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
// All-pairs gravitational n-body step, as in the CUDA samples' "nbody"
// (GPU Gems 3, chapter 31): each block stages tiles of body positions in
// dynamic shared memory and accumulates the accelerations they cause.
//

#include <random>
#include <vector>

#include "cuda4cpu.hpp"
#include "sample.hpp"

using namespace cuda4cpu;

#define EPS2 0.01f

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
    float3 r;
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
    float invDist = rsqrtf(distSqr);
    float invDistCube = invDist * invDist * invDist;
    float s = bj.w * invDistCube;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__global__ void integrateBodies(float4 *newPos, float4 *newVel,
                                const float4 *oldPos, const float4 *oldVel,
                                float deltaTime, int numBodies)
{
    float4 *shPosition = dynamic_shared<float4>();   // CUDA: extern __shared__ float4 shPosition[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 pos = oldPos[i];
    float3 acc = make_float3(0.f, 0.f, 0.f);

    for (int tile = 0; tile < int(gridDim.x); ++tile) {
        shPosition[threadIdx.x] = oldPos[tile * blockDim.x + threadIdx.x];
        __syncthreads();

        for (unsigned j = 0; j < blockDim.x; ++j)
            acc = bodyBodyInteraction(pos, shPosition[j], acc);
        __syncthreads();
    }

    float4 vel = oldVel[i];
    vel.x += acc.x * deltaTime;
    vel.y += acc.y * deltaTime;
    vel.z += acc.z * deltaTime;

    pos.x += vel.x * deltaTime;
    pos.y += vel.y * deltaTime;
    pos.z += vel.z * deltaTime;

    newPos[i] = pos;
    newVel[i] = vel;
}

int main()
{
    const int n = 4096;
    const unsigned threads = 256;
    const float dt = 0.01f;

    std::vector<float4> pos(n), vel(n), new_pos(n), new_vel(n), gold_pos(n), gold_vel(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i = 0; i < n; ++i) {
        pos[i] = make_float4(dist(gen), dist(gen), dist(gen), 1.f);
        vel[i] = make_float4(dist(gen), dist(gen), dist(gen), 0.f);
    }

    double ms = sample::time_ms([&] {
        launch(integrateBodies, n / threads, threads, threads * sizeof(float4))
            .call(new_pos.data(), new_vel.data(), pos.data(), vel.data(), dt, n);
    });

    // Same interaction order as the kernel, so the results match closely
    double host_ms = sample::time_ms([&] {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            float3 acc = make_float3(0.f, 0.f, 0.f);
            for (int j = 0; j < n; ++j)
                acc = bodyBodyInteraction(pos[i], pos[j], acc);
            float4 v = vel[i], p = pos[i];
            v.x += acc.x * dt; v.y += acc.y * dt; v.z += acc.z * dt;
            p.x += v.x * dt;   p.y += v.y * dt;   p.z += v.z * dt;
            gold_pos[i] = p;
            gold_vel[i] = v;
        }
    });

    sample::report("nbody", ms, host_ms);

    std::vector<float> got, expected;
    for (int i = 0; i < n; ++i) {
        for (const float4 *v : {&new_pos[i], &new_vel[i]}) { got.insert(got.end(), {v->x, v->y, v->z, v->w}); }
        for (const float4 *v : {&gold_pos[i], &gold_vel[i]}) { expected.insert(expected.end(), {v->x, v->y, v->z, v->w}); }
    }
    return sample::finish(sample::count_mismatches(got.data(), expected.data(), got.size(), 1e-4));
}
