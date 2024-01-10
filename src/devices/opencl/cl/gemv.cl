
// [1, m] * [k//4, 4, m] => [1, k]
__kernel void GemvConv1x1Impl(
  const int gws0,
  const int gws1,
  __global float *input, 
  __global uint *weight,
  __global float *output,
#ifdef HAS_BIAS
  __global float *bias, 
#endif
  __global float *scales, 
  __global float *mins,
  const int k, 
  const int m
) {
  int gid0 = get_global_id(0);
  int gid1 = get_global_id(1);
  if (gid1 >= gws1) {
    return;
  }

  // [k, m] => [k//4, 4, m]
  int weight_width = m >> 3;
  int weight_offset = gid0 * 4 * weight_width;
  int output_offset = gid0 << 2;
  float4 scale4 = vload4(0, scales + output_offset);
  float4 min4 = vload4(0, mins + output_offset);
  float4 minv = min4 / scale4;

  float4 in0, in1;
#ifdef HAS_BIAS
  float4 out0 = vload4(0, bias + output_offset);
#else
  float4 out0 = 0;
#endif
  uint4 w0;

  for (int i = 0; i < weight_width; i++) {
    w0.s0 = weight[weight_offset];
    w0.s1 = weight[weight_offset + weight_width];
    w0.s2 = weight[weight_offset + weight_width * 2];
    w0.s3 = weight[weight_offset + weight_width * 3];
    in0 = vload4(0, input + (i << 3));
    in1 = vload4(0, input + (i << 3) + 4);

    out0 = mad((float4) in0.s0, convert_float4(w0 >> 28) + minv, out0);
    out0 = mad((float4) in0.s1, convert_float4((w0 >> 24) & 15) + minv, out0);
    out0 = mad((float4) in0.s2, convert_float4((w0 >> 20) & 15) + minv, out0);
    out0 = mad((float4) in0.s3, convert_float4((w0 >> 16) & 15) + minv, out0);
    out0 = mad((float4) in1.s0, convert_float4((w0 >> 12) & 15) + minv, out0);
    out0 = mad((float4) in1.s1, convert_float4((w0 >> 8) & 15) + minv, out0);
    out0 = mad((float4) in1.s2, convert_float4((w0 >> 4) & 15) + minv, out0);
    out0 = mad((float4) in1.s3, convert_float4(w0 & 15) + minv, out0);

    weight_offset++;
  }  

  out0 *= scale4;
  vstore4(out0, 0, output + output_offset);
}
