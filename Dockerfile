# CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
#
# Copyright (C) 2014 Javier Cabezas
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# cuda4cpu installed in /usr/local, with GCC 14 and CMake to build programs.
# By default it runs the MCP server on stdio:
#
#     docker run -i --rm ghcr.io/javier-cabezas/cuda4cpu
#     docker run --rm -v "$PWD:/src" -w /src ghcr.io/javier-cabezas/cuda4cpu cuda4cpu-nvcc -O2 app.cu -o app

FROM ubuntu:24.04 AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++-14 cmake ninja-build python3 libnuma-dev \
    && rm -rf /var/lib/apt/lists/*
COPY . /src
RUN cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 \
        -DCUDA4CPU_BUILD_TESTS=OFF -DCUDA4CPU_BUILD_BENCHMARKS=OFF \
    && cmake --build /build \
    && cmake --install /build --prefix /opt/cuda4cpu

FROM ubuntu:24.04
# g++-14 brings the sanitizers' runtimes, which the MCP server's check uses
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++-14 cmake make python3 libnuma1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/cuda4cpu /usr/local
RUN ldconfig
ENV CXX=g++-14
LABEL org.opencontainers.image.source="https://github.com/javier-cabezas/cuda4cpu" \
      org.opencontainers.image.description="cuda4cpu: CUDA and HIP programs on the CPU, and an MCP server that compiles, runs and checks them" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.modelcontextprotocol.server.name="io.github.javier-cabezas/cuda4cpu"
CMD ["cuda4cpu-mcp"]
