#ifndef __RAYMARCHER_PARAMS__
#define __RAYMARCHER_PARAMS__

#include "camera.cuh"
#include <GL/glew.h>

class RayMarcherParams 
{
private:
	unsigned int _width = 0;
	unsigned int _height = 0;
	//vector3 _camera_position = { 10, 1, 1 };
	//vector3 _camera_direction = { 0, 0, 0 };


	//double _fov = 90;
	//double _camera_angle_vertical = 90;
	//double _camera_angle_horizontal = 90;
public:
	//Camera camera;

	RayMarcherParams() {}

	inline unsigned int getWidth() { return _width; }
	inline void setWidth(unsigned int width) { _width = width; }

	inline unsigned int getHeight() { return _height; }
	inline void setHeight(unsigned int height) { _height = height; }

	//inline double getFOV() { return _fov; }
	//inline void setFOV(double fov) { _fov = fov; }

	//inline double getCameraAngleVertical() { return _camera_angle_vertical; }
	//inline void setCameraAngleVerticalRadians(double angle) { _camera_angle_vertical = angle; }
	//inline void setCameraAngleVertivalDegrees(double angle) { _camera_angle_vertical = angle * PI / 180.0; }

	//inline double getCameraAngleHorizontal() { return _camera_angle_horizontal; }
	//inline void setCameraAngleHorizontalRadians(double angle) { _camera_angle_horizontal = angle; }
	//inline void setCameraAngleHorizontalDegrees(double angle) { _camera_angle_horizontal = angle * PI / 180.0; }
};

#endif // !__RAYMARCHER_PARAMS__
