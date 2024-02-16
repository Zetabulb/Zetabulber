#ifndef __ZETA_DEVICE__
#define __ZETA_DEVICE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "complex.cuh"
#include "cuda_error_handler.cuh"
#include <cuda/std/complex>

/// <summary>
/// CUDA implementation of the modified Borwein's efficient algorithm for calculating the Riemann zeta function
/// This file contains two implementations one using float type variables and one using double type variables
/// The float implementation is less accurate but much faster than the double implementation
/// The float implementation functions have "_f" in their names
/// To perform computations using this algorithm the coefficients psi must be initialized using functions zeta_init_host or zeta_init_host_f
/// Then functions zeta_split and zeta_split_f can be called from the device to get Riemann zeta values
/// This algorithm is best suited for computing multiple Riemann zeta function values at once
/// The following functions use arrays of size 2 to store complex numbers
/// </summary>


/// <summary>
/// Buffers for coefficients psi
/// </summary>
float* coefficients_device_buffer_f;
__device__ float* coefficients_device_f;

double* coefficients_device_buffer;
__device__ double* coefficients_device;
__device__ int n_device;

__device__ double _reflect_p[9] = { 0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7 };

/// <summary>
/// Reflection function used to expand the domain of the Riemann zeta function to sigma < 0.5
/// </summary>
/// <param name="value">input value complex number</param>
/// <param name="result">result value complex number</param>
/// <returns></returns>
__device__ void gamma_reflect_splitf(float value[], float result[]) {
    float _reflect_sqrt_2_pi = sqrtf(2.0f * PI);
    float _reflect_exp_65 = expf(6.5f);
    float _reflect_exp_10 = 2.0f * PI * exp(1.0f);

    double re_gamma = _reflect_p[0];
    double im_gamma = 0.0;

    for (int i = 1; i < 9; i++) {
        double re_z = value[0] + i - 1.0f;
        double im_z = value[1];
        double A = _reflect_p[i] / (re_z * re_z + im_z * im_z);

        re_gamma += A * re_z;
        im_gamma -= A * im_z;
    }

    float B = sqrtf(2.0f * PI);
    float C = 2.0f * PI * expf(1.0f);
    float D[2] = { value[0] + 6.5, value[1]};
    float D1[2] = { 0.0f, 0.0f };
    complex_sqrtf(D, D1);

    float E[2] = { (value[0] + 6.5) / C, value[1] / C };
    float F[2] = { 0.0f, 0.0f };
    complex_powf(E, value, F);

    result[0] = re_gamma;
    result[1] = im_gamma;
    if (abs(value[1]) < 50) {
        float G[2] = { 0.0f, 1.0f };
        float G1[2] = { 0.0f, 0.0f };
        complex_powf(G, value, G1);
        float H[2] = { 0.0f, -1.0f };
        float H1[2] = { 0.0f, 0.0f };
        complex_powf(H, value, H1);
        float D2[2] = { 0.0f, 0.0f };
        complex_multf(D1, expf(6.5f), D2);
        complex_addf(G1, H1, D);

        complex_multf(result, B, result);
        complex_multf(result, F, result);
        complex_multf(result, D, result);
        complex_divf(result, D2, result);
    }
    else {
        float F[2] = {0.0f, copysignf(1.0f, -value[1])};
        float G[2] = { 0.0f, 0.0f };
        complex_multf(F, E, G);
        float H[2] = { 0.0f, 0.0f };
        complex_powf(G, value, H);

        complex_multf(result, B, result);
        complex_divf(result, D, result);
        complex_divf(result, D1, result);
        complex_multf(result, H, result);
    }
}

/// <summary>
/// Modified Borwein's efficient algorithm
/// </summary>
/// <param name="value">input value complex number</param>
/// <param name="result">output value complex number</param>
/// <returns></returns>
__device__ void zeta_split_f(float value[], float result[]) {
    bool reflect = false;
    bool invert = false;

    float re_z = 0.0f;
    float im_z = 0.0f;

    float re_s = value[0];
    float im_s = value[1];

    if (re_s < 0.5f) {
        re_s = 1.0f - re_s;
        im_s = -im_s;
        reflect = true;
    }

    if (im_s < 0.0f) {
        im_s = -im_s;
        invert = true;
    }

    float pow2 = powf(2.0f, 1.0f - re_s);
    float omega = 1.0f - powf(2.0f, 2.0f - re_s) * cosf(im_s * logf(2.0f)) + powf(4.0f, 1.0f - re_s);

    for (int k = 0; k < n_device - 1; k++) {
        int p = k % 2 == 0 ? 1.0f : -1.0f;
        float A = p * coefficients_device_f[k] / powf(k + 1.0f, re_s);
        float B1 = im_s * logf(k + 1.0f);
        float B2 = im_s * logf((k + 1.0f) / 2.0f);

        re_z += A * (cosf(B1) - pow2 * cosf(B2));
        im_z += A * (sinf(B1) - pow2 * sinf(B2));
    }

    re_z /= omega;
    im_z /= -omega;


    if (reflect) {
        float r[2] = { 0, 0 };
        float s[2] = { re_s, im_s };
        gamma_reflect_splitf(s, r);
        float z[2] = { re_z, im_z };
        float r2[2] = { 0, 0 };
        complex_multf(z, r, r2);
        re_z = r2[0];
        im_z = r2[1];
    }

    if (invert)
        im_z = -im_z;

    result[0] = re_z;
    result[1] = im_z;
}

/// <summary>
/// Equivalent to gamma_reflect_splitf
/// </summary>
/// <param name="value"></param>
/// <param name="result"></param>
/// <returns></returns>
__device__ void gamma_reflect_split(double value[], double result[]) {
    double _reflect_sqrt_2_pi = sqrt(2.0 * PI);
    double _reflect_exp_65 = exp(6.5);
    double _reflect_exp_10 = 2.0 * PI * exp(1.0);

    double re_gamma = _reflect_p[0];
    double im_gamma = 0.0;

    for (int i = 1; i < 9; i++) {
        double re_z = value[0] + i - 1.0;
        double im_z = value[1];
        double A = _reflect_p[i] / (re_z * re_z + im_z * im_z);

        re_gamma += A * re_z;
        im_gamma -= A * im_z;
    }

    double B = sqrt(2.0 * PI);
    double C = 2.0 * PI * exp(1.0f);
    double D[2] = { value[0] + 6.5, value[1] };
    double D1[2] = { 0.0, 0.0 };
    complex_sqrt(D, D1);

    double E[2] = { (value[0] + 6.5) / C, value[1] / C };
    double F[2] = { 0.0f, 0.0f };
    complex_pow(E, value, F);

    result[0] = re_gamma;
    result[1] = im_gamma;
    if (abs(value[1]) < 50) {
        double G[2] = { 0.0f, 1.0f };
        double G1[2] = { 0.0f, 0.0f };
        complex_pow(G, value, G1);
        double H[2] = { 0.0f, -1.0f };
        double H1[2] = { 0.0f, 0.0f };
        complex_pow(H, value, H1);
        double D2[2] = { 0.0f, 0.0f };
        complex_mult(D1, exp(6.5), D2);
        complex_add(G1, H1, D);

        complex_mult(result, B, result);
        complex_mult(result, F, result);
        complex_mult(result, D, result);
        complex_div(result, D2, result);
    }
    else {
        double F[2] = { 0.0f, copysignf(1.0f, -value[1]) };
        double G[2] = { 0.0f, 0.0f };
        complex_mult(F, E, G);
        double H[2] = { 0.0f, 0.0f };
        complex_pow(G, value, H);

        complex_mult(result, B, result);
        complex_div(result, D, result);
        complex_div(result, D1, result);
        complex_mult(result, H, result);
    }
}

/// <summary>
/// Equivalent to zeta_split_f
/// </summary>
/// <param name="value"></param>
/// <param name="result"></param>
/// <returns></returns>
__device__ void zeta_split(double value[], double result[]) {
    bool reflect = false;
    bool invert = false;

    double re_z = 0.0;
    double im_z = 0.0;

    double re_s = value[0];
    double im_s = value[1];

    if (re_s < 0.5) {
        re_s = 1.0 - re_s;
        im_s = -im_s;
        reflect = true;
    }

    if (im_s < 0.0) {
        im_s = -im_s;
        invert = true;
    }

    double pow2 = pow(2.0, 1.0 - re_s);
    double omega = 1.0 - pow(2.0, 2.0 - re_s) * cos(im_s * log(2.0)) + pow(4.0, 1.0 - re_s);

    for (int k = 0; k < n_device - 1; k++) {
        int p = k % 2 == 0 ? 1 : -1;
        double A = p * coefficients_device[k] / pow(k + 1.0, re_s);
        double B1 = im_s * log(k + 1.0);
        double B2 = im_s * log((k + 1.0) / 2.0);

        re_z += A * (cos(B1) - pow2 * cos(B2));
        im_z += A * (sin(B1) - pow2 * sin(B2));
    }

    re_z /= omega;
    im_z /= -omega;

    if (reflect) {
        double r[2] = { 0, 0 };
        double s[2] = { re_s, im_s };
        gamma_reflect_split(s, r);
        double z[2] = { re_z, im_z };
        double r2[2] = { 0, 0 };
        complex_mult(z, r, r2);
        re_z = r2[0];
        im_z = r2[1];
    }

    if (invert)
        im_z = -im_z;

    result[0] = re_z;
    result[1] = im_z;
}

/// <summary>
/// Emipirically evaluated minimum number n of coefficients required to compute the Riemann zeta function with d decimal digis of accuracy
/// Only gives reliable results for the double implementation of this algorithm
/// </summary>
/// <param name="t">maximum imaginary part of an argument for which the Riemann zeta function should be computed</param>
/// <param name="d">decimal digits of accuracy with which the Riemann zeta function should be computed</param>
/// <returns></returns>
int nEMB(float t, int d) {
    double a = 0.451;
    double b = 1.407 * std::sqrt(d) - 0.245;
    double c = 0.371 * d + 0.195;
    return ceil(a * t + b * std::sqrt(t) + c);
}


/// <summary>
/// Coefficients psi computation
/// </summary>
/// <param name="n">number of coefficients, usually given by the function nEMB</param>
/// <returns></returns>
float* cMBf(int n) {
    int K = std::floor((float)n / std::sqrt(2));
    float* T = new float[n + 1];
    T[0] = -std::log(n);
    for (int k = 1; k <= n; k++)
        T[k] = T[k - 1] + std::log(n - k + 1) + std::log(n + k - 1) - std::log(2 * k - 1) - std::log(2 * k);
    float* SR = new float[n + 1];
    SR[0] = std::exp(T[0] - T[K] - K * std::log(4));
    for (int k = 1; k <= n; k++)
        SR[k] = SR[k - 1] + std::exp(T[k] - T[K] + std::log(4) * (k - K));
    float* cnk = new float[n];
    for (int k = 0; k < n; k++)
        cnk[k] = (1 - SR[k] / SR[n]);
    delete[] T;
    delete[] SR;
    return cnk;
}

/// <summary>
/// Equivalent to cMBf
/// </summary>
/// <param name="n"></param>
/// <returns></returns>
double* cMB(int n) {
    int K = std::floor((double)n / std::sqrt(2));
    double* T = new double[n + 1];
    T[0] = -std::log(n);
    for (int k = 1; k <= n; k++)
        T[k] = T[k - 1] + std::log(n - k + 1) + std::log(n + k - 1) - std::log(2 * k - 1) - std::log(2 * k);
    double* SR = new double[n + 1];
    SR[0] = std::exp(T[0] - T[K] - K * std::log(4));
    for (int k = 1; k <= n; k++)
        SR[k] = SR[k - 1] + std::exp(T[k] - T[K] + std::log(4) * (k - K));
    double* cnk = new double[n];
    for (int k = 0; k < n; k++)
        cnk[k] = (1 - SR[k] / SR[n]);
    delete[] T;
    delete[] SR;
    return cnk;
}


/// <summary>
/// Function used allocate a buffer in the device.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="pointer"></param>
/// <param name="buffer"></param>
/// <param name="size"></param>
/// <param name="ceh"></param>
/// <param name="error_key"></param>
template <typename T>
void malloc_pointer(T*& pointer, T*& buffer, unsigned int size, CudaErrorHandler ceh, unsigned int error_key) {
    ceh.checkCudaStatus(cudaMalloc((void**)&buffer, size * sizeof(T)), error_key);
    ceh.checkCudaStatus(cudaMemcpyToSymbol(pointer, &buffer, sizeof(T*)), error_key);
}

/// <summary>
/// Coefficient psi initialisation function. Must be called before calling zeta_split_f.
/// Note that since the initialisation process is linear it is usuallly more efficient for it to be performed on the CPU. 
/// </summary>
/// <param name="max_t">maximum imaginary part of an argument for which the Riemann zeta function should be computed</param>
/// <param name="d"></param>
/// <param name="ceh">Instance of the CudaErrorHandler. Error handling can be removed or changed if needed.</param>
/// <param name="error_key">Error key of the CudaErrorHandler.</param>
void zeta_init_host_f(float max_t, int d, CudaErrorHandler ceh, unsigned int error_key) {
    int n = nEMB(max_t, d);
    float* c_buffer = cMBf(n);
    ceh.checkCudaStatus(cudaMemcpyToSymbol(n_device, &n, sizeof(int), 0, cudaMemcpyHostToDevice), error_key);

    malloc_pointer<float>(coefficients_device_f, coefficients_device_buffer_f, n, ceh, error_key);
    ceh.checkCudaStatus(cudaMemcpy(coefficients_device_buffer_f, c_buffer, n * sizeof(float), cudaMemcpyHostToDevice), error_key);

    delete[] c_buffer;
}

/// <summary>
/// Equivalent to zeta_init_host_f
/// </summary>
/// <param name="max_t"></param>
/// <param name="d"></param>
/// <param name="ceh"></param>
/// <param name="error_key"></param>
void zeta_init_host(float max_t, int d, CudaErrorHandler ceh, unsigned int error_key) {
    int n = nEMB(max_t, d);
    double* c_buffer = cMB(n);
    ceh.checkCudaStatus(cudaMemcpyToSymbol(n_device, &n, sizeof(int), 0, cudaMemcpyHostToDevice), error_key);

    malloc_pointer<double>(coefficients_device, coefficients_device_buffer, n, ceh, error_key);
    ceh.checkCudaStatus(cudaMemcpy(coefficients_device_buffer, c_buffer, n * sizeof(double), cudaMemcpyHostToDevice), error_key);

    delete[] c_buffer;
}

void zeta_destroy(CudaErrorHandler ceh, unsigned int error_key) {
    ceh.checkCudaStatus(cudaFree(coefficients_device_buffer_f), error_key);
}

#endif