#ifndef __VECTOR3__H__
#define __VECTOR3__H__

namespace Vector3 {
    __host__ __device__ inline float magnitude(float3 v)
    {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    __host__ __device__ inline float3 normalize(float3 v) {
        float mag = magnitude(v);
        if (mag == 0)
            return v;
        return make_float3(v.x / mag, v.y / mag, v.z / mag);
    }

    __host__ __device__ inline float3 subtract(float3 a, float3 b)
    {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __host__ __device__ inline float3 add(float3 a, float3 b)
    {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __host__ __device__ inline float dot(float3 a, float3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ inline float3 multiply(float3 a, float val)
    {
        return make_float3(a.x * val, a.y * val, a.z * val);
    }

    __host__ __device__ inline float3 move(float3 v, float3 direction, float step)
    {
        return make_float3(v.x + step * direction.x, v.y + step * direction.y, v.z + step * direction.z);
    }

    __host__ __device__ inline float3 cross(float3 a, float3 b)
    {
        return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }

    __host__ __device__ inline void rotate_vertical(float3& v, float angle)
    {
        float cos_angle = cosf(angle);
        float sin_angle = sinf(angle);
        float3 row1 = { 1.0f, 0.0f, 0.0f };
        float3 row2 = { 0.0f, cos_angle, -sin_angle };
        float3 row3 = { 0.0f, sin_angle, cos_angle };
        v.x = dot(v, row1);
        v.y = dot(v, row2);
        v.z = dot(v, row3);
        //return make_float3( dot(v, row1), dot(v, row2), dot(v, row3) );
    }

    __host__ __device__ inline void rotate_horizontal(float3& v, float angle)
    {
        float cos_angle = cosf(angle);
        float sin_angle = sinf(angle);
        float3 row1 = { cos_angle, 0.0f, sin_angle };
        float3 row2 = { 0.0f, 1.0f, 0.0f };
        float3 row3 = { -sin_angle, 0.0f, cos_angle };
        v.x = dot(v, row1);
        v.y = dot(v, row2);
        v.z = dot(v, row3);
        //return vector3{ dot(v, row1), dot(v, row2), dot(v, row3) };
    }
}

#endif