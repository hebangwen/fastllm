
// [1, m] * [k//4, 4, m] => [1, k]
__kernel void GemvConv1x1Impl(
  __global float *input, 
  __global uchar *weight, 
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

  // [k, m] => [k//4, 4, m]
  int weight_width = m >> 1;
  int weight_offset = gid0 * 4 * weight_width;
  int output_offset = gid0 << 2;
  float4 scale4 = vload4(0, scales + output_offset);
  float4 min4 = vload4(0, mins + output_offset);
  float4 minv = min4 / scale4;

  float2 in0;
#ifdef HAS_BIAS
  float4 out0 = vload4(0, bias + output_offset);
#else
  float4 out0 = 0;
#endif
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

  // if (gid0 == 0) {
  //   printf("%v4lf\n", out0);
  // }
}
