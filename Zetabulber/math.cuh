#ifndef __MATH_DEVICE__
#define __MATH_DEVICE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "params.cuh"

__host__ __device__ void copyVec(float vec[], float vec_copy[], int num = 3) {
    for (int i = 0; i < num; ++i) { vec_copy[i] = vec[i]; }
}

__host__ __device__ void copyVec(float vec[], double vec_copy[], int num = 3) {
    for (int i = 0; i < num; ++i) { vec_copy[i] = vec[i]; }
}

__host__ __device__  float myClamp(float val) {
    if (val <= 0.0f)
        val = 0.0f;
    val += (0.25f)/(1 + powf(2.0f * val, 5));
    //if (val >= 0.0f) { return val; }
    //if (val <= 0.0f) { return 0.5f; }
    return val;
}

__host__ __device__ float dotVec3s(float vec1[], float vec2[]) {
    return (vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]);
}

__host__ __device__ float magVec(float vec[]) {
    return sqrtf(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

__host__ __device__ double magVec(double vec[]) {
    return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

__host__ __device__ void subVec3s(float vec1[], float vec2[], float result[]) {
    result[0] = vec1[0] - vec2[0];
    result[1] = vec1[1] - vec2[1];
    result[2] = vec1[2] - vec2[2];
}

__host__ __device__ void addVec3s(float vec1[], float vec2[], float result[]) {
    result[0] = vec1[0] + vec2[0];
    result[1] = vec1[1] + vec2[1];
    result[2] = vec1[2] + vec2[2];
}

__host__ __device__ void addVec3s(float vec1[], float vec2[], double result[]) {
    result[0] = vec1[0] + vec2[0];
    result[1] = vec1[1] + vec2[1];
    result[2] = vec1[2] + vec2[2];
}

__host__ __device__ void multNumVec3(float val, float vec[], float result[]) {
    result[0] = val * vec[0];
    result[1] = val * vec[1];
    result[2] = val * vec[2];
}

__host__ __device__ void moveVec3(float vec[], float dir[], float step, float result[]) {
    result[0] = vec[0] + step * dir[0];
    result[1] = vec[1] + step * dir[1];
    result[2] = vec[2] + step * dir[2];
}

__host__ __device__ void moveVec3(float vec[], float dir[], float step) {
    vec[0] += step * dir[0];
    vec[1] += step * dir[1];
    vec[2] += step * dir[2];
}

__host__ __device__ void normVec3(float vec[]) {
    float vec_mag = magVec(vec);
    if (vec_mag == 0) vec_mag = 1; // just return the original vector
    vec[0] = vec[0] / vec_mag;
    vec[1] = vec[1] / vec_mag;
    vec[2] = vec[2] / vec_mag;
}

__host__ __device__ void multMatrixVec3(float matrix[], float vec3[], float result[]) {
    result[0] = matrix[0] * vec3[0] + matrix[1] * vec3[1] + matrix[2] * vec3[2];
    result[1] = matrix[3] * vec3[0] + matrix[4] * vec3[1] + matrix[5] * vec3[2];
    result[2] = matrix[6] * vec3[0] + matrix[7] * vec3[1] + matrix[8] * vec3[2];
}

__host__ __device__ void crossVec3s(float vec1[], float vec2[], float result[]) {
    result[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    result[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
    result[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
}

__host__ __device__ void rotateVec3Ver(float vec[], float angle) {
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    float rotation_matrix[9] = {
        1, 0, 0,
        0, cos_angle, -sin_angle,
        0, sin_angle, cos_angle
    };
    float rot_vec[3];
    multMatrixVec3(rotation_matrix, vec, rot_vec);
    vec[0] = rot_vec[0];
    vec[1] = rot_vec[1];
    vec[2] = rot_vec[2];
}

__host__ __device__ void rotateVec3Hor(float vec[], float angle) {
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    float rotation_matrix[9] = {
        cos_angle,  0, sin_angle,
        0, 1, 0,
        -sin_angle, 0, cos_angle
    };
    float rot_vec[3];
    multMatrixVec3(rotation_matrix, vec, rot_vec);
    vec[0] = rot_vec[0];
    vec[1] = rot_vec[1];
    vec[2] = rot_vec[2];
}

__host__ __device__ void rotateVec3Z(float vec[], float angle) {
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    float rotation_matrix[9] = {
        cos_angle, -sin_angle, 0,
        sin_angle, cos_angle, 0,
        0, 0, 1,
    };
    float rot_vec[3];
    multMatrixVec3(rotation_matrix, vec, rot_vec);
    vec[0] = rot_vec[0];
    vec[1] = rot_vec[1];
    vec[2] = rot_vec[2];
}

__host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

inline __device__ float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline __device__ float4 operator*(float a, float4 b) { return make_float4(a * b.x, a * b.y, a * b.z, a * b.w); }
inline __device__ float4 operator*(float4 a, float b) { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }

#endif