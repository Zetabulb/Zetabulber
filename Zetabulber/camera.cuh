#ifndef __CAMERA__
#define __CAMERA__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math.cuh"

#define PI 3.14159265f

class Camera 
{
private:
	float _fov = deg2rad(45.0f);

	//1
	//float _position[3] = { 0.0f, 1.0f, 3.0f };
	//float _rotation[3] = { 0.16f, 0.0f, -1.0f };
	//float _vertical_rotation = 0.0f;

	//2
	//float _position[3] = { 0.0f, 1.0f, 2.0f };
	//float _rotation[3] = { 0.16f, 0.0f, -1.0f };
	//float _vertical_rotation = -0.150f;

	//3
	float _position[3] = { -0.809405f, 0.3f, 1.81017f };
	float _rotation[3] = { 0.54222f, 0.0f, -0.965913f };
	float _vertical_rotation = 0.011f;

	float _target_position[3] = {0.0f, 0.0f, 0.0f};
	float _vector_to_screen[3] = { 0.0f, 0.0f, 0.0f };
	float _global_up[3] = { 0.0f, 10000.0f, 0.0f };
	float _camera_movement_speed = 0.05f;
	float _camera_rotation_speed = 0.001f;

	float _angle_vertical = deg2rad(0.0f);
	float _angle_horizontal = deg2rad(0.0f);

private:
	void updateTargetPosition() {
		_target_position[0] = _position[0] + _rotation[0];
		_target_position[1] = _position[1] + _vertical_rotation;
		_target_position[2] = _position[2] + _rotation[2];
		//std::cout << _rotation[0] << " " << _rotation[1] << " " << _rotation[2] << " " << _vertical_rotation << std::endl;
	}

	float deg2rad(float deg) {
		return deg * PI / 180.0f;
	}

public:
	Camera() {
	
	}

	Camera(unsigned int width, unsigned int height)
	{
		_vector_to_screen[0] = floorf(width / 2.0f);
		_vector_to_screen[1] = floorf(height / 2.0f);
		_vector_to_screen[2] = height / tanf(_fov);
		updateTargetPosition();
	}


	__host__ __device__ float* vectorToScreen() {
		return _vector_to_screen;
	}

	__host__ __device__ float* position() {
		return _position;
	}

	__host__ __device__ float* targetPosition() {
		return _target_position;
	}

	__host__ __device__ float* globalUp() {
		_global_up[0] = _position[0];
		_global_up[2] = _position[2];
		return _global_up;
	}

	__host__ __device__ float angleVertical() {
		return _angle_vertical;
	}
		
	__host__ __device__ float angleHorizontal() {
		return _angle_horizontal;
	}

	__host__ __device__ float* getViewMatrix() {
		float z_axis[3];
		subVec3s(_target_position, _position, z_axis);
		normVec3(z_axis);
		float x_axis[3];
		crossVec3s(z_axis, _global_up, x_axis);
		normVec3(x_axis);
		float y_axis[3];
		crossVec3s(z_axis, x_axis, y_axis);
		float v_matrix[9] = {
			x_axis[0], y_axis[0], z_axis[0],
			x_axis[1], y_axis[1], z_axis[1],
			x_axis[2], y_axis[2], z_axis[2]
		};
		return v_matrix;
	}

	void moveLeft() {
		float direction[3] = { -_rotation[2], 0.0f, _rotation[0] };
		moveVec3(_position, direction, _camera_movement_speed);
		updateTargetPosition();
	}

	void moveRight() {
		float direction[3] = { _rotation[2], 0.0f, -_rotation[0] };
		moveVec3(_position, direction, _camera_movement_speed);
		updateTargetPosition();
	}

	void moveDown() {
		_position[1] -= _camera_movement_speed;
		updateTargetPosition();
	}

	void moveUp() {
		_position[1] += _camera_movement_speed;
		updateTargetPosition();
	}

	void moveForward() {
		float direction[3] = { _rotation[0], 0.0f, _rotation[2] };
		moveVec3(_position, direction, _camera_movement_speed);
		updateTargetPosition();
	}

	void moveBackward() {
		float direction[3] = { _rotation[0], 0.0f, _rotation[2] };
		moveVec3(_position, direction, -_camera_movement_speed);
		updateTargetPosition();
	}

	void rotateLeft() {
		updateTargetPosition();
	}

	void rotateRight() {
		updateTargetPosition();
	}

	void rotateUp() {
		updateTargetPosition();
	}
	void rotateDown() {
		updateTargetPosition();
	}

	void rotateHorizontal(double times) {
		rotateVec3Hor(_rotation, times * _camera_rotation_speed);
		updateTargetPosition();
	}

	void rotateVertical(double times) {
		float new_rotation = _vertical_rotation + times * _camera_rotation_speed;
		if (times < 0)
			_vertical_rotation = max(-1.0f, new_rotation);		
		
		if (times >= 0)
			_vertical_rotation = min(1.0f, new_rotation);

		updateTargetPosition();
	}

	void setPosition(float* position) {
		_position[0] = position[0];
		_position[1] = position[1];
		_position[2] = position[2];
		updateTargetPosition();
	}

	void setRotation(float* horizontalRotation, float verticalRotation) {
		_rotation[0] = horizontalRotation[0];
		_rotation[1] = horizontalRotation[1];
		_rotation[2] = horizontalRotation[2];
		_vertical_rotation = verticalRotation;
		updateTargetPosition();
	}
};

#endif