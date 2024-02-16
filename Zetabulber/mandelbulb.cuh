#ifndef __MANDELBULB_DEVICE__
#define __MANDELBULB_DEVICE__

#include <cuda_runtime.h>
#include "device_launch_parameters.h"


__device__ float SDFPlane(float p[]) {
	float tmp_pos[3] = {p[0], p[2]};
	float zeta_res[2] = { 0.0f, 0.0f };
	zeta_split_f(tmp_pos, zeta_res);

	float n[3] = {0.0f, 1.0f, 0.0f};
	float h = zeta_res[0];

	return (dotVec3s(p, n) + h);
}

__device__ float SDFSphere(float ray_pos[])
{
	float tmp_pos[3];
	copyVec(ray_pos, tmp_pos);
	return magVec(tmp_pos) - 1.0f;
}

__device__ double round_to(double value, double precision) {
	return round(value / precision) * precision;
}

__device__ float DEZetabulb(float p[], kernel_params params) {
	float tmp_pos[3];
	copyVec(p, tmp_pos);
	
	if (params.zeta_enabled) {
		tmp_pos[0] = p[0] + params.zeta_pos[0];
		tmp_pos[1] = p[1] + params.zeta_pos[1];

		float zeta_res[2] = { 0.0f, 0.0f };
		zeta_split_f(tmp_pos, zeta_res);

		tmp_pos[0] = zeta_res[0];
		tmp_pos[1] = zeta_res[1];
		tmp_pos[2] = p[2];
	}
	
	float cart_pos[3];
	float dr = 1.0f;
	float r;
	float theta;
	float phi;
	float zr;
	for (int tmp_iter = 0; tmp_iter < 15; tmp_iter++) {
		r = magVec(tmp_pos);
		if (r > 2.0f) { break; }
		// approximate the distance differential

		float powr = powf(r, params.power - 1.0f);

		dr = params.power * powr * dr + 1.0f;
		// calculate fractal surface
		// convert to polar coordinates
		theta = params.power * acosf(tmp_pos[2] / r);
		phi = params.power * atan2f(tmp_pos[1], tmp_pos[0]);
		zr = r * powr;
		// convert back to cartesian coordinated

		float sin_theta = sinf(theta);
		cart_pos[0] = zr * sin_theta * cosf(phi);
		cart_pos[1] = zr * sin_theta * sinf(phi);
		cart_pos[2] = zr * cosf(theta);
		addVec3s(p, cart_pos, tmp_pos);
	}

	return 0.5f * logf(r) * r / dr;
}

__device__ void getNormal(float ray_pos[], float surf_normal[], kernel_params params) {
	float epsilon_x[3] = { EPSILON, 0, 0 };
	float epsilon_y[3] = { 0, EPSILON, 0 };
	float epsilon_z[3] = { 0, 0, EPSILON };
	float ray_perb_x1[3];
	float ray_perb_y1[3];
	float ray_perb_z1[3];
	addVec3s(ray_pos, epsilon_x, ray_perb_x1);
	addVec3s(ray_pos, epsilon_y, ray_perb_y1);
	addVec3s(ray_pos, epsilon_z, ray_perb_z1);
	float ray_perb_x2[3];
	float ray_perb_y2[3];
	float ray_perb_z2[3];
	subVec3s(ray_pos, epsilon_x, ray_perb_x2);
	subVec3s(ray_pos, epsilon_y, ray_perb_y2);
	subVec3s(ray_pos, epsilon_z, ray_perb_z2);
	surf_normal[0] = DEZetabulb(ray_perb_x1, params) - DEZetabulb(ray_perb_x2, params);
	surf_normal[1] = DEZetabulb(ray_perb_y1, params) - DEZetabulb(ray_perb_y2, params);
	surf_normal[2] = DEZetabulb(ray_perb_z1, params) - DEZetabulb(ray_perb_z2, params);
	//surf_normal[0] = SDFPlane(ray_perb_x1) - SDFPlane(ray_perb_x2);
	//surf_normal[1] = SDFPlane(ray_perb_y1) - SDFPlane(ray_perb_y2);
	//surf_normal[2] = SDFPlane(ray_perb_z1) - SDFPlane(ray_perb_z2);
	normVec3(surf_normal);
}

__device__ float getLight(float ray_pos[], kernel_params params) {
	float light_pos[3];
	copyVec(params.light_position, light_pos);
	// rotate light source along with camera
	rotateVec3Hor(light_pos, params.angle_horizontal);
	rotateVec3Ver(light_pos, params.angle_vertical);
	// measure angle of intesection of light with surface
	float light2surface_angle[3];
	subVec3s(light_pos, ray_pos, light2surface_angle);
	normVec3(light2surface_angle);
	// measure angle of surface normal
	float surface_normal[3];
	getNormal(ray_pos, surface_normal, params);
	// calculate how intense light consentration is at point on surface
	return myClamp(dotVec3s(surface_normal, light2surface_angle));
}

#endif