#ifndef __CUDA_ERROR_HANDLER__
#define __CUDA_ERROR_HANDLER__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <functional>
#include <map>
#include <iostream>

class CudaErrorHandler 
{
private:
	unsigned int currentKey = 0;
	std::map<unsigned int, std::function<void()>> functionMap;

	unsigned int getKey()
	{
		return currentKey++;
	}

	void invoke(unsigned int key)
	{
		auto it = functionMap.find(key);
		if (it != functionMap.end())
			it->second();
	}
public: 
	unsigned int subscribe(std::function<void()> function)
	{
		unsigned int key = getKey();
		functionMap.emplace(key, function);
		return key;
	}

	void unsubscribe(unsigned int key)
	{
		functionMap.erase(key);
	}

	void checkCudaStatus(cudaError_t cudaStatus, unsigned int key, bool abort = true)
	{
		if (cudaStatus != cudaSuccess) {
			std::cout << "Cuda error: " << cudaGetErrorString(cudaStatus) << std::endl;
			invoke(key);
			if (abort)
				std::exit(cudaStatus);
		}
	}
};

#endif