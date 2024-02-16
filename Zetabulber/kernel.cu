
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda_error_handler.cuh"
#include <cuda_gl_interop.h>
#include "params.cuh"
#include "vector3.cuh"
#include "math.cuh"
#include "complex.cuh"
#include "zeta.cuh"
#include "mandelbulb.cuh"
#include "color_gradient.cuh"
#include "test_scenarios.cuh"
#include <cuda/std/complex>

GLuint viewGLTexture;
cudaGraphicsResource_t viewCudaResource;

GLuint viewGLTexture_pp;
cudaGraphicsResource_t viewCudaResource_pp;

float4* ss_buffer_h = new float4[WIDTH * HEIGHT];
float4* ss_buffer;

void initialize(int width, int height) {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &viewGLTexture);

    glBindTexture(GL_TEXTURE_2D, viewGLTexture);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

SceneParams sp;

//__host__ __device__ float* view_matrix(SceneParams sceneParams) {
//    // working with arrays : https://stackoverflow.com/questions/31180470/declaring-a-double-array-in-c-using-brackets-or-asterisk
//    // basis vectors : https://www.3dgep.com/understanding-the-view-matrix/
//    // forward basis vector
//    float cam2obj[3];
//    subVec3s(sceneParams.TARGET_POS, sceneParams.CAM_POS, cam2obj);
//    normVec3(cam2obj);
//    // right basis vector
//    float x_axis[3];
//    crossVec3s(cam2obj, sceneParams.GLOBAL_UP, x_axis);
//    normVec3(x_axis);
//    // up basis vector
//    float y_axis[3];
//    crossVec3s(x_axis, cam2obj, y_axis);
//    // store view matrix as array
//    float v_matrix[9] = {
//        x_axis[0],  x_axis[1],  x_axis[2],
//        y_axis[0],  y_axis[1],  y_axis[2],
//        cam2obj[0], cam2obj[1], cam2obj[2]
//    };
//    return v_matrix;
//}

__device__ float DESphereVector(float3 ray_pos)
{
    return Vector3::magnitude(ray_pos) - 1.0f;
}

__global__ void
call_zeta(float re, float im) {
    float val[2] = { re, im};
    float res[2] = { 0.0f, 0.0f};
    zeta_split_f(val, res);
    printf("float %f, %f\n", res[0], res[1]);
}

__global__ void
call_zeta(double re, double im) {
    double val[2] = { re, im };
    double res[2] = { 0.0f, 0.0f };
    zeta_split(val, res);
    printf("double %f, %f\n", res[0], res[1]);
}

__global__ void
compare_zeta(double re, double im, double step, double* result) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    float valf[2] = { re, im + step * x };
    float resf[2] = { 0.0f, 0.0f };
    zeta_split_f(valf, resf);

    double vald[2] = { re, im + step * x };
    double resd[2] = { 0.0, 0.0 };
    zeta_split(vald, resd);

    double diff[2] = { resd[0] - resf[0], resd[1] - resf[1] };
    double mag = complex_mag(diff);
    result[x] = mag;
}

__global__ void
plot_zeta_diff(double re, double re_step, int re_pts, double im, double im_step, int im_pts, double* diff_matrix) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z > re_pts * im_pts) return;

    int x = z % re_pts;
    int y = z / im_pts;

    float valf[2] = { re + x * re_step, im + y * im_step };
    float resf[2] = { 0.0f, 0.0f };
    zeta_split_f(valf, resf);

    double vald[2] = { re + x * re_step, im + y * im_step };
    double resd[2] = { 0.0, 0.0 };
    zeta_split(vald, resd);

    double diff[2] = { resd[0] - resf[0], resd[1] - resf[1] };
    double mag = complex_mag(diff);
    diff_matrix[z] = mag;
}

__global__ void 
rayMarchingKernel(cudaSurfaceObject_t target, kernel_params params, float4* ss_buffer) {
    //unsigned int width = 1080;
    //unsigned int height = 720;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;


    // compute vector from camera to object
    // in world coordinates
    float vec2pixel[3] = { (float)x, (float)y, 0.0f };
    float ray_dir_world[3];
    subVec3s(params.vector_to_screen, vec2pixel, ray_dir_world);
    normVec3(ray_dir_world);
    // in camera (view matrix) coordinates
    float ray_dir_cam[3];
    multMatrixVec3(params.view_matrix, ray_dir_world, ray_dir_cam);
    float ray_dist = 0.0f;
    float ray_pos[3];
    float total_distance = 0;
    float4 pixel = make_float4(0, 0, 0, 0);
    for (int i = 0; i < MAX_STEPS; i++) {
        moveVec3(params.camera_position, ray_dir_cam, total_distance, ray_pos);
        rotateVec3Ver(ray_pos, params.angle_vertical);
        rotateVec3Hor(ray_pos, params.angle_horizontal);
        float distance = DEZetabulb(ray_pos, params);
        //float distance = SDFPlane(ray_pos);
        if (distance < SURF_DIST) {
            float light = getLight(ray_pos, params);
            //float light = 1.0f;

            //moveVec3(params.camera_position, ray_dir_cam, total_distance + distance, ray_pos);
            float distance_from_center = magVec(ray_pos);
            pixel = get_color(distance_from_center);
            pixel.x *= light;
            pixel.y *= light;
            pixel.z *= light;
            break;
        }

        if (total_distance > MAX_DIST) {
            if (params.sky_box)
                pixel = make_float4(0.35f, 0.68f, 0.83f, 0.0f);
            else
                pixel = make_float4(255, 255, 255, 0);
            break;
        }

        total_distance += distance;
    }
    //distance += getLight(ray_back_step, params);
    //uchar4 data;
    //if (total_distance + 0.001 > MAX_DIST) {
    //    data = make_uchar4(0, 255, 255, 57);
    //}
    //else {
    //    //int value = 175;
    //    int value = (int)((MAX_DIST - total_distance) / MAX_DIST * 175.0f);
    //    data = make_uchar4(value, value, value, 1);
    //}
    //surf2Dwrite(data, target, x * sizeof(uchar4), y);
    surf2Dwrite<float4>(pixel, target, (int)sizeof(float4) * (x), (y), cudaBoundaryModeClamp);
    ss_buffer[y * WIDTH + x] = pixel;
}

std::string get_timestamp() {
    std::time_t time = std::time({});
    char timeString[std::size("yyyymmdd_hhmmss")];
    std::strftime(std::data(timeString), std::size(timeString),
        "%y%m%d_%H%M%S", std::gmtime(&time));
    return timeString;
}

void screenshot() {
    cudaMemcpy(ss_buffer_h, ss_buffer, sizeof(float4) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    std::string timestamp = get_timestamp();
    std::cout << "Saving screenshot to Screenshots\\" << timestamp << ".ppm" << std::endl;
    std::ofstream image("screenshots/" + timestamp + ".ppm");

    if (image.is_open()) {
        image << "P3\n" << WIDTH << " " << HEIGHT << "\n" << 255 << "\n";
        for (int x = HEIGHT - 1; x >= 0; x--) {
            for (int y = 0; y < WIDTH; y++) {
                float4 value = ss_buffer_h[x * WIDTH + y];
                image << (int)(value.x * 255) << " " << (int)(value.y * 255) << " " << (int)(value.z * 255) << "\n";
            }
        }
        image.close();
    }
    else {
        std::cout << "ERROR: Unable to open distance txt file.\n";
        throw std::runtime_error("ERROR: Unable to open distance txt file.\n");
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    //return;

    bool keyPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);

    if (key == GLFW_KEY_P && keyPressed) {
        screenshot();
    }

    //return;
    // camera movement
    if (key == GLFW_KEY_D && keyPressed) {
        sp.camera.moveRight();
    }
    if (key == GLFW_KEY_A && keyPressed) {
        sp.camera.moveLeft();
    }
    if (key == GLFW_KEY_SPACE && keyPressed) {
        sp.camera.moveUp();
    }
    if (key == GLFW_KEY_LEFT_SHIFT && keyPressed) {
        sp.camera.moveDown();
    }  
    if (key == GLFW_KEY_W && keyPressed) {
        sp.camera.moveForward();
    }
    if (key == GLFW_KEY_S && keyPressed) {
        sp.camera.moveBackward();
    }
    // camera rotations 
    if (key == GLFW_KEY_Q && keyPressed) {
        sp.camera.rotateLeft();
    }
    if (key == GLFW_KEY_E && keyPressed) {
        sp.camera.rotateRight();
    }
    if (key == GLFW_KEY_Z && keyPressed) {
        sp.camera.rotateUp();
    }
    if (key == GLFW_KEY_C && keyPressed) {
        sp.camera.rotateDown();
    }

    // zeta plate movement
    if (key == GLFW_KEY_I && keyPressed) {
        sp.plusImZeta();
    }
    if (key == GLFW_KEY_K && keyPressed) {
        sp.minusImZeta();
    }
    if (key == GLFW_KEY_L && keyPressed) {
        sp.plusReZeta();
    }
    if (key == GLFW_KEY_J && keyPressed) {
        sp.minusReZeta();
    }
    // power change

    if (key == GLFW_KEY_KP_ADD && keyPressed) {
        sp.increasePower();
    }
    if (key == GLFW_KEY_KP_SUBTRACT && keyPressed) {
        sp.decreasePower();
    }

    if (key == GLFW_KEY_B && keyPressed) {
        sp.toggle_sky_box();
    }

    float* pos = sp.camera.position();
    float* zeta_pos = sp.zeta_position();
    //float* rot = s.
    std::cout << "camera_position (x, y, z): (" << pos[0] << "; " << pos[1] << "; " << pos[2] << "), zeta_position (re, im): (" << zeta_pos[0] << "; " << zeta_pos[1] << ")" << std::endl;
}

__global__ void filter_fxaa2(cudaSurfaceObject_t data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int edge = 0;
    if (x >= WIDTH - edge || y >= HEIGHT - edge || x <= edge || y <= edge) return;

    float4 out_color;

    const float FXAA_SPAN_MAX = 8.0f;
    const float FXAA_REDUCE_MUL = 1.0f / 8.0f;
    const float FXAA_REDUCE_MIN = (1.0f / 128.0f);

    int u = x;
    int v = y;

    float4 rgbNW, rgbNE, rgbSW, rgbSE, rgbM;

    surf2Dread<float4>(&rgbNW, data, (int)sizeof(float4) * (u - 1), (v - 1), cudaBoundaryModeClamp);
    surf2Dread<float4>(&rgbNE, data, (int)sizeof(float4) * (u + 1), (v - 1), cudaBoundaryModeClamp);
    surf2Dread<float4>(&rgbSW, data, (int)sizeof(float4) * (u - 1), (v + 1), cudaBoundaryModeClamp);
    surf2Dread<float4>(&rgbSE, data, (int)sizeof(float4) * (u + 1), (v + 1), cudaBoundaryModeClamp);
    surf2Dread<float4>(&rgbM, data, (int)sizeof(float4) * (u), (v), cudaBoundaryModeClamp);

    const float4 luma = make_float4(0.299f, 0.587f, 0.114f, 0.0f);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM = dot(rgbM, luma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25f * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

    float rcpDirMin = 1.0f / (min(abs(dir.x), abs(dir.y)) + dirReduce);


    //float2 test = dir * rcpDirMin;
    float2 test = dir;
    test.x *= rcpDirMin;
    test.y *= rcpDirMin;
    //dir = clamp(test, -FXAA_SPAN_MAX, FXAA_SPAN_MAX);
    dir.x = clamp(dir.x, -FXAA_SPAN_MAX, FXAA_SPAN_MAX);
    dir.y = clamp(dir.y, -FXAA_SPAN_MAX, FXAA_SPAN_MAX);

    //float4 a1, a2, b1, b2;
    //a1 = surf2Dread<float4>(data, (int)sizeof(float4) * (int)(u + dir.x * (1.0f / 3.0f - 0.5f)), (int)(v + dir.y * (1.0f / 3.0f - 0.5f)), cudaBoundaryModeClamp);

    float4 rgbA = (1.0f / 2.0f) * (
        surf2Dread<float4>(data, (int)sizeof(float4) * (int)(u + dir.x * (1.0f / 3.0f - 0.5f)), (int)(v + dir.y * (1.0f / 3.0f - 0.5f)), cudaBoundaryModeClamp) +
        surf2Dread<float4>(data, (int)sizeof(float4) * (int)(u + dir.x * (2.0f / 3.0f - 0.5f)), (int)(v + dir.y * (2.0f / 3.0f - 0.5f)), cudaBoundaryModeClamp));
    float4 rgbB = rgbA * (1.0f / 2.0f) + (1.0f / 4.0f) * (
        surf2Dread<float4>(data, (int)sizeof(float4) * (int)(u + dir.x * (0.0f / 3.0f - 0.5f)), (int)(v + dir.y * (0.0f / 3.0f - 0.5f)), cudaBoundaryModeClamp) +
        surf2Dread<float4>(data, (int)sizeof(float4) * (int)(u + dir.x * (3.0f / 3.0f - 0.5f)), (int)(v + dir.y * (3.0f / 3.0f - 0.5f)), cudaBoundaryModeClamp));
    float lumaB = dot(rgbB, luma);

    if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
        out_color = rgbA;
    }
    else {
        out_color = rgbB;
    }

    surf2Dwrite<float4>(make_float4(out_color.x, out_color.y, out_color.z, 0.0f), data, (int)sizeof(float4) * (x), (y), cudaBoundaryModeClamp);
}

double previous_xpos = 0;
double previous_ypos = 0;

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    //return;
    sp.camera.rotateHorizontal(xpos - previous_xpos);
    sp.camera.rotateVertical(previous_ypos - ypos);
    previous_xpos = xpos;
    previous_ypos = ypos;
    //std::cout << xpos << " " << ypos << " " << previous_xpos << " " << previous_ypos << std::endl;
}

int main()
{
    if (!glfwInit()) { 
        return -1;
    }

    unsigned int block_size = 32;

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Zetabulber", NULL, NULL);


    if (!window) { 
        glfwTerminate();
        return -2;
    }

    glfwMakeContextCurrent(window);

    if (GLEW_OK != glewInit())
    {
        return -3;
    }

    glfwSetKeyCallback(window, key_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (glfwRawMouseMotionSupported())
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    CudaErrorHandler _cudaErrorHandler;
    unsigned int _error_key = _cudaErrorHandler.subscribe(NULL);

    dim3 threads(block_size, 8);
    dim3 blocks(WIDTH / threads.x + 1, HEIGHT / threads.y + 1);

    unsigned int _thread_count = 32;
    unsigned int _block_count = WIDTH * HEIGHT / _thread_count + 1;

    _cudaErrorHandler.checkCudaStatus(cudaSetDevice(0), _error_key);
    _cudaErrorHandler.checkCudaStatus(cudaGetLastError(), _error_key);
    cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost | cudaDeviceLmemResizeToMax);

    int nbFrames = 0;

    initialize(WIDTH, HEIGHT);
    int i = 0;
    _cudaErrorHandler.checkCudaStatus(cudaGraphicsMapResources(1, &viewCudaResource), _error_key);

    int d = 10;
    float max_t = 50.0f;
    
    zeta_init_host_f(max_t, d, _cudaErrorHandler, _error_key);

    float a = 0.12345678910;
    //printf("%f\n", a);
    //printf("\n%f, %f\n", pos[0], pos[1]);
    //return;
    //sp.c_array = /*c_device*/;
    const int n_colors = 5;
    //float4 colors[n_colors] = {
    //    make_float4(41.0f, 78.0f, 175.0f, 0.0f),
    //    make_float4(41.0f, 78.0f, 125.0f, 0.0f),
    //    make_float4(41.0f, 253.0f, 46.0f, 0.0f),
    //    make_float4(201.0f, 119.0f, 10.0f, 0.0f),
    //    make_float4(232.0f, 45.0f, 69.0f, 0.0f),
    //};

    //purple
    //float4 colors[n_colors] = {
    //    make_float4(41.0f, 3.0f, 94.0f, 0.0f),
    //    make_float4(83.0f, 11.0f, 184.0f, 0.0f),
    //    make_float4(199.0f, 20.0f, 77.0f, 0.0f),
    //    make_float4(237, 12, 94, 0.0f),
    //    make_float4(110, 134, 240, 0.0f),
    //};

    //orange
    //float4 colors[n_colors] = {
    //make_float4(83, 6, 145, 0.0f),
    //make_float4(194, 12, 204, 0.0f),
    //make_float4(181, 28, 11, 0.0f),
    //make_float4(252, 86, 3, 0.0f),
    //make_float4(252, 202, 3, 0.0f),
    //};

    //green
    float4 colors[n_colors] = {
    make_float4(210, 252, 223, 0.0f),
    make_float4(210, 252, 223, 0.0f),
    make_float4(166, 245, 190, 0.0f),
    //make_float4(80, 143, 99, 0.0f),

    make_float4(11, 156, 112, 0.0f),
    make_float4(2, 46, 33, 0.0f),
    };    
    
    //blue
    //float4 colors[n_colors] = {
    //make_float4(107, 73, 52, 0.0f),
    //make_float4(235, 204, 120, 0.0f),
    //make_float4(128, 163, 191, 0.0f),

    //make_float4(71, 73, 125, 0.0f),
    //make_float4(72, 247, 247, 0.0f),
    //};


    //float distances[n_colors] = {
    //    0.65, 0.7, 0.75, 0.85, 1.05
    //};

    float distances[n_colors] = {
    0.1, 0.3, 0.75, 0.85, 1.05
    };
    gradient_init(n_colors, colors, distances, _cudaErrorHandler, _error_key);

    int start_input;
    //std::cin >> start_input;

    //float4* ss_buffer = new float4[WIDTH * HEIGHT];
    _cudaErrorHandler.checkCudaStatus(cudaMalloc(&ss_buffer, sizeof(float4)* WIDTH* HEIGHT), _error_key);

    bool isRunning = false;

    double lastTime = glfwGetTime();
    int frames = 0;
    while (!glfwWindowShouldClose(window)) { 
        // fps counter
        double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) { 
            std::cout << nbFrames << " fps" << std::endl;
            nbFrames = 0;
            lastTime += 1.0;
        }

        cudaArray_t viewCudaArray;
        _cudaErrorHandler.checkCudaStatus(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0), _error_key);
        cudaResourceDesc viewCudaArrayResourceDesc;
        viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
        viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
        cudaSurfaceObject_t viewCudaSurfaceObject;
        _cudaErrorHandler.checkCudaStatus(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc), _error_key);

        kernel_params params = sp.getKernelParams();

        rayMarchingKernel << <blocks, threads >> > (viewCudaSurfaceObject, params, ss_buffer);
        filter_fxaa2 << <blocks, threads >> > (viewCudaSurfaceObject);

        _cudaErrorHandler.checkCudaStatus(cudaGetLastError(), _error_key);

        _cudaErrorHandler.checkCudaStatus(cudaDestroySurfaceObject(viewCudaSurfaceObject), _error_key);
        _cudaErrorHandler.checkCudaStatus(cudaStreamSynchronize(0), _error_key);

        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, viewGLTexture);
        {
            glBegin(GL_QUADS);
            {
                glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
                glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
                glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
                glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
            }
            glEnd();
        }
        glBindTexture(GL_TEXTURE_2D, 0);

        glFinish();

        glfwSwapBuffers(window);
        glfwPollEvents();

        frames++;
    }
    zeta_destroy(_cudaErrorHandler, _error_key);
    _cudaErrorHandler.checkCudaStatus(cudaGraphicsUnmapResources(1, &viewCudaResource), _error_key);
    glfwTerminate();
    return 0;
}