#ifndef __COLOR_GRADIENT_H__
#define __COLOR_GRADIENT_H__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda_error_handler.cuh"

#define GAMMA 0.43f

float4* colors_device_buffer;
float* distances_device_buffer;
float4* colors_linear_device_buffer;
float* brightnesses_device_buffer;

__device__ int n_colors_device;
__device__ float4* colors_device;
__device__ float* distances_device;
__device__ float4* colors_linear_device;
__device__ float* brightnesses_device;

__device__ __host__ float to_sRGB(float x) {
    return x <= 0.0031308f ? 12.92f * x : (1.055f * (powf(x, (1.0f / 2.4f)))) - 0.055f;
}

__device__ __host__ float from_sRGB(float x) {
    x /= 255.0f;
    if (x <= 0.04045f)
        return x / 12.92f;
    else
        return powf(((x + 0.055f) / 1.055f), 2.4f);
}

__device__ __host__ float lerp(float a, float b, float fraction) {
    return a * (1.0 - fraction) + b * fraction;
}

__device__ __host__ float sum(float4 v) {
    return v.x + v.y + v.z;
}

__device__ __host__ float pow_gamma(float x) {
    return powf(x, 1 / GAMMA);
}

__device__ __host__ float4 all_channels(float4 c, float f(float)) {
    c.x = f(c.x);
    c.y = f(c.y);
    c.z = f(c.z);
    return c;
}

__device__ __host__ float4 all_channels2(float4 color1, float4 color2, float fraction, float f(float, float, float)) {
    color1.x = f(color1.x, color2.x, fraction);
    color1.y = f(color1.y, color2.y, fraction);
    color1.z = f(color1.z, color2.z, fraction);
    return color1;
}

__device__ int get_index(float x) {
    if (x < distances_device[0])
        return -1;

    else if (x >= distances_device[n_colors_device - 1])
        return n_colors_device;

    for (int i = 0; i < n_colors_device - 1; i++)
        if (x >= distances_device[i] && x < distances_device[i + 1])
            return i;
}

__device__ float get_fraction(float x, int index) {
    return (x - distances_device[index]) / (distances_device[index + 1] - distances_device[index]);
}

__device__ float4 get_color(float x) {
    int index = get_index(x);
    if (index == -1)
        return all_channels(colors_linear_device[0], to_sRGB);
    else if (index == n_colors_device)
        return all_channels(colors_linear_device[n_colors_device - 1], to_sRGB);

    float fraction = get_fraction(x, index);
    //float4 intensity = all_channels2(brightnesses_device[index], brightnesses_device[index + 1], fraction, lerp);
    float intensity = lerp(brightnesses_device[index], brightnesses_device[index + 1], fraction);
    intensity = pow_gamma(intensity);
    float4 color = all_channels2(colors_linear_device[index], colors_linear_device[index + 1], fraction, lerp);
    float color_sum = sum(color);
    if (color_sum != 0) {
        color.x *= (intensity / color_sum);
        color.y *= (intensity / color_sum);
        color.z *= (intensity / color_sum);
    }
    color = all_channels(color, to_sRGB);
    return color;

}


void gradient_init(int n, float4 colors[], float distances[], CudaErrorHandler ceh, unsigned int error_key) {
    ceh.checkCudaStatus(cudaMemcpyToSymbol(n_colors_device, &n, sizeof(int), 0, cudaMemcpyHostToDevice), error_key);

    malloc_pointer<float4>(colors_device, colors_device_buffer, n, ceh, error_key);
    ceh.checkCudaStatus(cudaMemcpy(colors_device_buffer, colors, n * sizeof(float4), cudaMemcpyHostToDevice), error_key);

    malloc_pointer<float>(distances_device, distances_device_buffer, n, ceh, error_key);
    ceh.checkCudaStatus(cudaMemcpy(distances_device_buffer, distances, n * sizeof(float), cudaMemcpyHostToDevice), error_key);

    float4* colors_buffer = new float4[n];
    float* brightnesses_buffer = new float[n];
    for (int i = 0; i < n; i++) {
        colors_buffer[i] = all_channels(colors[i], from_sRGB);
        brightnesses_buffer[i] = powf(sum(colors_buffer[i]), GAMMA);
        //std::cout << brightnesses_device_buffer[i] << std::endl;
    }

    malloc_pointer<float4>(colors_linear_device, colors_linear_device_buffer, n, ceh, error_key);
    ceh.checkCudaStatus(cudaMemcpy(colors_linear_device_buffer, colors_buffer, n * sizeof(float4), cudaMemcpyHostToDevice), error_key);

    malloc_pointer<float>(brightnesses_device, brightnesses_device_buffer, n, ceh, error_key);
    ceh.checkCudaStatus(cudaMemcpy(brightnesses_device_buffer, brightnesses_buffer, n * sizeof(float), cudaMemcpyHostToDevice), error_key);

}

#endif