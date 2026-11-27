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

// rocFFT's API on cuda4cpu's FFTs: plans made from a description (data
// layouts with lengths and strides fastest dimension first, offsets, a
// scale factor) and executed with an execution info (stream, work buffer,
// load and store callbacks). The functions are in the library, with C
// linkage. Interleaved arrays only: planar ones, fields and bricks (several
// GPUs) and plan caches aren't provided.
#pragma once

#include <cstddef>

extern "C" {

struct rocfft_plan_t;
typedef struct rocfft_plan_t *rocfft_plan;
struct rocfft_plan_description_t;
typedef struct rocfft_plan_description_t *rocfft_plan_description;
struct rocfft_execution_info_t;
typedef struct rocfft_execution_info_t *rocfft_execution_info;

typedef enum rocfft_status_e {
    rocfft_status_success,
    rocfft_status_failure,
    rocfft_status_invalid_arg_value,
    rocfft_status_invalid_dimensions,
    rocfft_status_invalid_array_type,
    rocfft_status_invalid_strides,
    rocfft_status_invalid_distance,
    rocfft_status_invalid_offset,
    rocfft_status_invalid_work_buffer
} rocfft_status;

typedef enum rocfft_transform_type_e {
    rocfft_transform_type_complex_forward,
    rocfft_transform_type_complex_inverse,
    rocfft_transform_type_real_forward,
    rocfft_transform_type_real_inverse
} rocfft_transform_type;

typedef enum rocfft_precision_e { rocfft_precision_single, rocfft_precision_double, rocfft_precision_half } rocfft_precision;

typedef enum rocfft_result_placement_e { rocfft_placement_inplace, rocfft_placement_notinplace } rocfft_result_placement;

typedef enum rocfft_array_type_e {
    rocfft_array_type_complex_interleaved,
    rocfft_array_type_complex_planar,
    rocfft_array_type_real,
    rocfft_array_type_hermitian_interleaved,
    rocfft_array_type_hermitian_planar,
    rocfft_array_type_unset
} rocfft_array_type;

rocfft_status rocfft_setup(void);
rocfft_status rocfft_cleanup(void);
rocfft_status rocfft_get_version_string(char *buf, size_t len);

rocfft_status rocfft_plan_create(rocfft_plan *plan, rocfft_result_placement placement,
                                 rocfft_transform_type transform_type, rocfft_precision precision, size_t dimensions,
                                 const size_t *lengths, size_t number_of_transforms,
                                 const rocfft_plan_description description);
rocfft_status rocfft_execute(const rocfft_plan plan, void *in_buffer[], void *out_buffer[], rocfft_execution_info info);
rocfft_status rocfft_plan_destroy(rocfft_plan plan);
rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan, size_t *size_in_bytes);
rocfft_status rocfft_plan_get_print(const rocfft_plan plan);

rocfft_status rocfft_plan_description_create(rocfft_plan_description *description);
rocfft_status rocfft_plan_description_destroy(rocfft_plan_description description);
rocfft_status rocfft_plan_description_set_scale_factor(rocfft_plan_description description, const double scale_factor);
rocfft_status rocfft_plan_description_set_data_layout(rocfft_plan_description description,
                                                      const rocfft_array_type in_array_type,
                                                      const rocfft_array_type out_array_type, const size_t *in_offsets,
                                                      const size_t *out_offsets, const size_t in_strides_size,
                                                      const size_t *in_strides, const size_t in_distance,
                                                      const size_t out_strides_size, const size_t *out_strides,
                                                      const size_t out_distance);

rocfft_status rocfft_execution_info_create(rocfft_execution_info *info);
rocfft_status rocfft_execution_info_destroy(rocfft_execution_info info);
rocfft_status rocfft_execution_info_set_work_buffer(rocfft_execution_info info, void *work_buffer,
                                                    const size_t size_in_bytes);
rocfft_status rocfft_execution_info_set_stream(rocfft_execution_info info, void *stream);
rocfft_status rocfft_execution_info_set_load_callback(rocfft_execution_info info, void **cb_functions, void **cb_data,
                                                      size_t shared_mem_bytes);
rocfft_status rocfft_execution_info_set_store_callback(rocfft_execution_info info, void **cb_functions,
                                                       void **cb_data, size_t shared_mem_bytes);

} // extern "C"
