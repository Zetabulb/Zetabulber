#ifndef __PARAMS__
#define __PARAMS__

#include "camera.cuh"

#define HEIGHT 720
#define WIDTH 1280
// raymarching properties
#define MAX_STEPS 1000
#define MAX_DIST  10.0
#define SURF_DIST 0.000002
#define EPSILON   0.001

struct kernel_params {
    float power;
    float angle_horizontal;
    float angle_vertical;
    float camera_position[3];
    float camera_target[3];
    float vector_to_screen[3];
    float global_up[3];
    float light_position[3];
    float view_matrix[9];
    float zeta_pos[2];
    bool sky_box;
    bool zeta_enabled;
    bool power_enabled;
};

class SceneParams
{
private:
    float _light_position[3] = { 0.0f, 50.0f, 0.0f };
    float _power = 8.0f;
    //1
    //float _zeta_position[2] = {1.5f, 5.0f};
    //2
    //float _zeta_position[2] = {1.5f, 500.0f};
    //3
    //float _zeta_position[2] = {1.0f,163.161f};
    // 16 decent
    //float _zeta_position[2] = { 1.0f, 0.0f };
    float _zeta_position[2] = { 1.5f, 30.424876125f };
    //float _zeta_position[2] = { 1.5f, 75.65f };

    float _zeta_movement_speed = 0.005f;
    float _power_speed = 0.01f;
    bool _sky_box = true;
    bool _im_movement_enabled = true;
    bool _re_movement_enabled = true;
    bool _zeta_enabled = true;
    bool _power_enabled = true;
public:
    Camera camera;

    SceneParams():
        camera(HEIGHT, WIDTH)
    {
        
    }

    void increasePower() {
        if (_power_enabled)
            _power += 0.01f;
    }

    void decreasePower() {
        if (_power_enabled)
            _power -= 0.01f;
    }

    void minusReZeta() {
        if (_re_movement_enabled)
            _zeta_position[0] -= _zeta_movement_speed;
    }

    void plusReZeta() {
        if (_re_movement_enabled)
            _zeta_position[0] += _zeta_movement_speed;
    }

    void minusImZeta() {
        if (_im_movement_enabled)
            _zeta_position[1] -= _zeta_movement_speed;
    }
    void plusImZeta() {
        if (_im_movement_enabled)
            _zeta_position[1] += _zeta_movement_speed;
    }

    float* zeta_position() {
        return _zeta_position;
    }

    void set_zeta_position(float* position) {
        _zeta_position[0] = position[0];
        _zeta_position[1] = position[1];
    }

    void toggle_sky_box() {
        _sky_box = !_sky_box;
    }

    void toggle_re_movement(bool enabled) {
        _re_movement_enabled = enabled;
    }

    void toggle_im_movement(bool enabled) {
        _im_movement_enabled = enabled;
    }

    void toggle_zeta(bool enabled) {
        _zeta_enabled = enabled;
    }

    void toggle_power(bool enabled) {
        _power_enabled = enabled;
    }

    void set_power(float power) {
        _power = power;
    }


    kernel_params getKernelParams() {
        return {
            _power,
            camera.angleHorizontal(),
            camera.angleVertical(),
            {camera.position()[0], camera.position()[1], camera.position()[2]},
            {camera.targetPosition()[0], camera.targetPosition()[1], camera.targetPosition()[2]},
            {camera.vectorToScreen()[0], camera.vectorToScreen()[1], camera.vectorToScreen()[2]},
            {camera.globalUp()[0], camera.globalUp()[1], camera.globalUp()[2]},
            {camera.position()[0], _light_position[1], camera.position()[2]},
            {camera.getViewMatrix()[0], camera.getViewMatrix()[1], camera.getViewMatrix()[2],
            camera.getViewMatrix()[3], camera.getViewMatrix()[4], camera.getViewMatrix()[5],
            camera.getViewMatrix()[6], camera.getViewMatrix()[7], camera.getViewMatrix()[8],},
            {_zeta_position[0], _zeta_position[1]},
            {_sky_box},
            _zeta_enabled,
            _power_enabled
        };
    }
private: 
    float deg2rad(float deg) {
        return deg * PI / 180.0f;
    }
};

#endif // !__PARAMS__
