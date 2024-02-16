#ifndef __COMPLEX_DEVICE__
#define __COMPLEX_DEVICE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "params.cuh"


__device__ void complex_sqrt(double value[], double result[]) {
    double re_s = value[0];
    double im_s = value[1];

    double r = sqrt(re_s * re_s + im_s * im_s);
    double b = sqrt((re_s + r) * (re_s + r) + im_s * im_s);

    result[0] = sqrt(r) * (re_s + r) / b;
    result[1] = sqrt(r) * im_s / b;
}

__device__ void complex_pow(double a[], double b[], double result[]) {
    double re_1 = a[0];
    double im_1 = a[1];

    double re_2 = b[0];
    double im_2 = b[1];

    double theta = atan(im_1 / re_1);
    double r2 = re_1 * re_1 + im_1 * im_1;
    double B = re_2 * theta + im_2 * log(r2) / 2;
    double A = pow(r2, re_2 / 2) * exp(-im_2 * theta);

    result[0] = A * cos(B);
    result[1] = A * sin(B);
}

__device__ void complex_div(double a[], double b[], double result[]) {
    double re_1 = a[0];
    double im_1 = a[1];

    double re_2 = b[0];
    double im_2 = b[1];

    double r2 = re_2 * re_2 + im_2 * im_2;
    result[0] = (re_1 * re_2 + im_1 * im_2) / r2;
    result[1] = (im_1 * re_2 - re_1 * im_2) / r2;
}

__device__ void complex_mult(double a[], double b[], double result[]) {
    result[0] = a[0] * b[0] - a[1] * b[1];
    result[1] = a[0] * b[1] + a[1] * b[0];
}

__device__ void complex_mult(double a[], double b, double result[]) {
    result[0] = a[0] * b;
    result[1] = a[1] * b;
}

__device__ void complex_sqrtf(float value[], float result[]) {
    float re_s = value[0];
    float im_s = value[1];

    float r = sqrtf(re_s * re_s + im_s * im_s);
    float b = sqrtf((re_s + r) * (re_s + r) + im_s * im_s);

    result[0] = sqrtf(r) * (re_s + r) / b;
    result[1] = sqrtf(r) * im_s / b;
}

__device__ void complex_powf(float a[], float b[], float result[]) {
    float re_1 = a[0];
    float im_1 = a[1];

    float re_2 = b[0];
    float im_2 = b[1];

    float theta = atanf(im_1 / re_1);
    float r2 = re_1 * re_1 + im_1 * im_1;
    float B = re_2 * theta + im_2 * logf(r2) / 2.0f;
    float eTheta = 0.0f;
    //if (-im_2 * theta > 7)
    //    eTheta = 0;
    //else
        eTheta = expf(-im_2 * theta);
    float A = powf(r2, re_2 / 2.0f) * eTheta;

    result[0] = A * cos(B);
    result[1] = A * sin(B);
}

__device__ void complex_divf(float a[], float b[], float result[]) {
    float re_1 = a[0];
    float im_1 = a[1];

    float re_2 = b[0];
    float im_2 = b[1];

    float r2 = re_2 * re_2 + im_2 * im_2;
    result[0] = (re_1 * re_2 + im_1 * im_2) / r2;
    result[1] = (im_1 * re_2 - re_1 * im_2) / r2;
}

__device__ void complex_multf(float a[], float b[], float result[]) {
    float re = a[0] * b[0] - a[1] * b[1];
    float im = a[0] * b[1] + a[1] * b[0];

    result[0] = re;
    result[1] = im;
}

__device__ void complex_multf(float a[], float b, float result[]) {
    result[0] = a[0] * b;
    result[1] = a[1] * b;
}

__device__ void complex_addf(float a[], float b[], float result[]) {
    result[0] = a[0] + b[0];
    result[1] = a[1] + b[1];
}

__device__ void complex_add(double a[], double b[], double result[]) {
    result[0] = a[0] + b[0];
    result[1] = a[1] + b[1];
}

__device__ float complex_magf(float a[]) {
    return sqrtf(a[0] * a[0] + a[1] * a[1]);
}

__device__ double complex_mag(double a[]) {
    return sqrt(a[0] * a[0] + a[1] * a[1]);
}

#endif