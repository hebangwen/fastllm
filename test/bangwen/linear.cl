// [1, m] * [k, m] => [1, k]
// A: [1, m]
// B: [k, m]
// C: [1, k]
// bias: [1, k]
// scales: [k, 1]
// mins: [k, 1]

// ALG1: 每一个线程计算一个 VV, 一共 K 个线程
__kernel void GemvFloatInt4NoZero(__global float *A, __global uchar *B,
                                  __global float *C, __global float *bias,
                                  __global float *scales, __global float *mins,
                                  const int m, const int k) {
  int tid = get_local_id(1);
  int tsz = get_local_size(1);

  int gid = get_global_id(0);
  __local float sdata[256];

  sdata[tid] = 0.0f;
  float minv = mins[gid] / scales[gid];
  for (int i = tid; i < (m >> 1); i += tsz) {
    uchar now = B[gid * (m >> 1) + i];
    sdata[tid] += (A[i * 2] * (minv + (now >> 4))) + (A[i * 2 + 1] * (minv + (now & 15)));
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (tid < 128) {
    sdata[tid] += sdata[tid + 128];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if (tid < 64) {
    sdata[tid] += sdata[tid + 64];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (tid < 64) {
    if (tid < 32) sdata[tid] += sdata[tid + 32];
    if (tid < 16) sdata[tid] += sdata[tid + 16];
    if (tid < 8) sdata[tid] += sdata[tid + 8];
    if (tid < 4) sdata[tid] += sdata[tid + 4];
    if (tid < 2) sdata[tid] += sdata[tid + 2];
    if (tid < 1) sdata[tid] += sdata[tid + 1];
  }

  // barrier(CLK_LOCAL_MEM_FENCE);

  if (tid == 0) {
    C[gid] = sdata[0] * scales[gid] + bias[gid];
  }
}

__kernel void GemvFloatInt4NoZero3(__global float *A, __global uchar *B,
                                  __global float *C, __global float *bias,
                                  __global float *scales, __global float *mins,
                                  const int m, const int k) {
  int tid = get_local_id(1);
  int tsz = get_local_size(1);

  int gid = get_global_id(0);
  __local float sdata[256];

  sdata[tid] = 0.0f;
  float minv = mins[gid] / scales[gid];

  // Loop unrolling
  for (int i = tid; i < (m >> 2); i += tsz) {
    uchar now = B[gid * (m >> 2) + i];
    sdata[tid] += (A[i * 4] * (minv + (now >> 4))) + (A[i * 4 + 1] * (minv + (now & 15)));
    now = B[gid * (m >> 2) + i + 1];
    sdata[tid] += (A[i * 4 + 2] * (minv + (now >> 4))) + (A[i * 4 + 3] * (minv + (now & 15)));
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Optimized reduction operation
  for (int s = tsz / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    C[gid] = sdata[0] * scales[gid] + bias[gid];
  }
}


__kernel void GemvFloatInt4NoZero4(__global float *A, __global uchar *B,
                                  __global float *C, __global float *bias,
                                  __global float *scales, __global float *mins,
                                  const int m, const int k) {
  int tid = get_local_id(1);
  int tsz = get_local_size(1);

  int gid = get_global_id(0);
  __local float sdata[256];
  __local float adata[256];
  __local uchar bdata[256];

  sdata[tid] = 0.0f;
  float minv = mins[gid] / scales[gid];

  for (int i = tid; i < (m >> 1); i += tsz) {
    adata[tid] = A[i * 2];
    bdata[tid] = B[gid * (m >> 1) + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    uchar now = bdata[tid];
    sdata[tid] += (adata[tid] * (minv + (now >> 4))) + (adata[tid + 1] * (minv + (now & 15)));

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Optimized reduction operation
  for (int s = tsz / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    C[gid] = sdata[0] * scales[gid] + bias[gid];
  }
}

// (int4) [k, m] -> (float) [k//4, m, 4]
__kernel void TransformUnquantWeight(__global uchar *weight, __global float *output,
                                     __global float *scales, __global float *mins, 
                                     const int k, const int m) {
  int gid0 = get_global_id(0);  // k>>2
  int gid1 = get_global_id(1);  // m>>1

  int input_channel = k >> 2;
  int input_height = 4, input_width = m >> 1;
  int output_height = m, output_width = 4;

  int input_offset = gid0 * input_height * input_width + gid1;
  int input_stride = input_width;
  int quant_value_idx = gid0 << 2;
  int output_offset = gid0 * output_height * output_width + (gid1 << 3);  // 每个线程计算 8 个结果, 所以左移 3 位

  uint4 data;
  data.s0 = weight[input_offset];
  data.s1 = weight[input_offset + input_stride];
  data.s2 = weight[input_offset + input_stride * 2];
  data.s3 = weight[input_offset + input_stride * 3];

  float4 scale4 = vload4(0, scales + quant_value_idx);
  float4 min4 = vload4(0, mins + quant_value_idx);
  float4 value = convert_float4(data >> 4) * scale4 + min4;
  vstore4(value, 0, output + output_offset);
  value = convert_float4(data & 15) * scale4 + min4;
  vstore4(value, 0, output + output_offset + 4);

}

// gemv 的 conv 1x1 实现
// conv_1x1: [1, m, 1, 1] (c) [k//4, m, 4, 1, 1] => [k//4, 4] => [k]
__kernel void conv2d_1x1(__global float *input, __global float *weight, 
                         __global float *output, __global float *bias, 
                         const int ic, const int ih, const int iw,
                         const int oc, const int oh, const int ow,
                         const int fh, const int fw) {
  int gid0 = get_global_id(0);    // k//4 (oc)
  int weight_offset = gid0 * fh * fw;

  float4 w0, w1, w2, w3;
  float4 in0, out0 = vload4(0, bias + (gid0 << 2));

  for (int i = 0; i < ic; i += 4) {
    in0 = vload4(0, input + i);

    w0 = vload4(0, weight + weight_offset);
    w1 = vload4(0, weight + weight_offset + 4);
    w2 = vload4(0, weight + weight_offset + 8);
    w3 = vload4(0, weight + weight_offset + 12);
    
    out0 = mad((float4) in0.s0, w0, out0);
    out0 = mad((float4) in0.s1, w1, out0);
    out0 = mad((float4) in0.s2, w2, out0);
    out0 = mad((float4) in0.s3, w3, out0);

    weight_offset += 16;
  }

  int output_offset = gid0 << 2;
  vstore4(out0, 0, output + output_offset);
}


__kernel void GemvConv1x1Impl(
  __global float *input, __global uchar *weight, __global float *output,
  __global float *bias, __global float *scales, __global float *mins,
  const int k, const int m
) {
  int gid0 = get_global_id(0);

  // [k, m] => [k//4, 4, m]
  int weight_width = m >> 1;
  int weight_offset = gid0 * 4 * weight_width;
  int output_offset = gid0 << 2;
  float4 scale4 = vload4(0, scales + output_offset);
  float4 min4 = vload4(0, mins + output_offset);
  float4 minv = min4 / scale4;

  float2 in0;
  float4 out0 = vload4(0, bias + output_offset);
  uint4 w0;

  for (int i = 0; i < weight_width; i++) {
    w0.s0 = weight[weight_offset];
    w0.s1 = weight[weight_offset + weight_width];
    w0.s2 = weight[weight_offset + m];
    w0.s3 = weight[weight_offset + m + weight_width];
    in0 = vload2(0, input + (i << 1));

    out0 = mad((float4) in0.s0, convert_float4(w0 >> 4) + minv, out0);
    out0 = mad((float4) in0.s1, convert_float4(w0 & 15) + minv, out0);

    weight_offset++;
  }  

  out0 *= scale4;
  vstore4(out0, 0, output + output_offset);
}


__kernel void GemvFloatInt4NoZero2(__global float *A, __global uchar *B,
                                  __global float *C, __global float *bias,
                                  __global float *scales, __global float *mins,
                                  const int m, const int k) {
  int tid = get_local_id(1);
  int tsz = get_local_size(1);

  int gid = get_global_id(0);
  __local float sdata[LOCAL_SIZE];

  sdata[tid] = 0.0f;
  float minv = mins[gid] / scales[gid];
  for (int i = tid; i < (m >> 2); i += tsz) {
    uchar2 now = vload2(gid * (m >> 1) + (i << 1), B);
    float4 A_float = vload4(0, A + (i << 2));
    float4 B_float = minv + (float4)(now.x >> 4, now.x & 0xf, now.y >> 4, now.y & 0xf);
    sdata[tid] += dot(A_float, B_float);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (tid < 64) {
    if (tid < 32) sdata[tid] += sdata[tid + 32];
    if (tid < 16) sdata[tid] += sdata[tid + 16];
    if (tid < 8) sdata[tid] += sdata[tid + 8];
    if (tid < 4) sdata[tid] += sdata[tid + 4];
    if (tid < 2) sdata[tid] += sdata[tid + 2];
    if (tid < 1) sdata[tid] += sdata[tid + 1];
  }

  // barrier(CLK_LOCAL_MEM_FENCE);

  if (tid == 0) {
    C[gid] = sdata[0] * scales[gid] + bias[gid];
  }
}


__kernel void gemv_quantized(__global uchar* A, __global float* x, __global float* y, int k, int m, __global float* scale, __global float* min) {
    int gid = get_global_id(0);
    if (gid < k) {
      float8 min_value = (float8)(min[gid], min[gid], min[gid], min[gid], min[gid], min[gid], min[gid], min[gid]);
      float8 scale_value = (float8)(scale[gid], scale[gid], scale[gid], scale[gid], scale[gid], scale[gid], scale[gid], scale[gid]);
      float8 minv = min_value / scale_value;
      float dot_product = 0.0f;
      for (int j = 0; j < m; j += 8) {
        uchar4 A_int = vload4(gid * (m >> 1) + (j >> 1), A);
        float8 A_float = minv + 
                          (float8)(A_int.s0 >> 4, A_int.s0 & 0xF, A_int.s1 >> 4, A_int.s1 & 0xF, 
                                   A_int.s2 >> 4, A_int.s2 & 0xF, A_int.s3 >> 4, A_int.s3 & 0xF);
        float8 x_float = vload8(j, x);
        dot_product += dot(A_float.lo, x_float.lo) + dot(A_float.hi, x_float.hi);
      }
      y[gid] = dot_product * scale[gid];
    }
}
// 调用代码
// cl::Kernel kernel{program, "gemv_quantized"};
// int idx = 0;
// ret |= kernel.setArg(idx++, bufferA);
// ret |= kernel.setArg(idx++, bufferX);
// ret |= kernel.setArg(idx++, bufferY);
// ret |= kernel.setArg(idx++, m);
// ret |= kernel.setArg(idx++, n);
// ret |= kernel.setArg(idx++, bufferScale);
// ret |= kernel.setArg(idx++, bufferMin);
// ret |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(m));


// __kernel void gemv_kernel(
//     __global uchar* matrix,
//     __global float* vector,
//     __global float* result,
//     __global float* bias,
//     int k,
//     int m,
//     __global float* scale,
//     __global float* min
// ) {
//     int gid = get_global_id(0);
//     int lid = get_local_id(0);

//     // Local memory to store a tile of the matrix
//     __local uchar localMatrixTile[LOCAL_SIZE][LOCAL_SIZE];

//     // Load the matrix tile into local memory
//     for (int i = lid; i < m; i += LOCAL_SIZE) {
//         localMatrixTile[lid][i] = matrix[gid * m + i];
//     }

//     barrier(CLK_LOCAL_MEM_FENCE);

//     // Compute the result using the matrix tile, vector, and bias
//     float sum = 0.0f;
//     for (int i = 0; i < m; ++i) {
//         uchar matrix_value = localMatrixTile[lid][i];
//         float matrix_float = scale[gid] * (float)matrix_value + min[gid];
//         sum += matrix_float * vector[i];
//     }

//     // Use work-group reduction to compute the final result
//     sum = work_group_reduce_add(sum);

//     // Write the result to global memory
//     if (lid == 0) {
//         result[gid] = sum + bias[gid];
//     }
// }
