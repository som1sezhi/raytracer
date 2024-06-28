#pragma once

#include <random>
#include <glm/glm.hpp>
#include <curand_kernel.h>

__host__ __device__
inline bool nearZero(const glm::vec3& v)
{
	constexpr float eps = 1e-10f;
	glm::vec3 absV = glm::abs(v);
	return absV.x < eps && absV.y < eps && absV.z < eps;
}

// Convert a gamma-transformed color to a linear color for gamma = 2.
__host__ __device__
inline glm::vec3 gammaToLinear(const glm::vec3& color)
{
	return color * color;
}

// Convert a linear color to a gamma-transformed color for gamma = 2.
__host__ __device__
inline glm::vec3 linearToGamma(const glm::vec3& color)
{
	glm::vec3 c = glm::max(color, glm::vec3(0.0));
	return glm::sqrt(c);
}

// Generate a random value in the interval [0, 1).
__host__ __device__
inline float randomFloat(curandState* state)
{
#ifdef __CUDA_ARCH__
	float x = curand_uniform((curandState*) state);
	return x == 1.0f ? 0.0f : x;
#else
	static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	static std::mt19937 generator;
	return distribution(generator);
#endif
}

__host__ __device__
inline glm::vec3 randomVec(curandState* state)
{
	return {
		randomFloat(state),
		randomFloat(state),
		randomFloat(state)
	};
}

__host__ __device__
inline float randomNormalDist(curandState* state)
{
	// https://stackoverflow.com/a/6178290 ported to C++
	float theta = glm::tau<float>() * randomFloat(state);
	float rho = glm::sqrt(-2.0f * glm::log(1.0f - randomFloat(state)));
	return rho * glm::cos(theta);
}

__host__ __device__
inline glm::vec3 randomUnitVec(curandState* state)
{
#ifdef __CUDA_ARCH__
	glm::vec3 v = {
		randomNormalDist(state),
		randomNormalDist(state),
		randomNormalDist(state)
	};
#else
	glm::vec3 v = randomVec(state) * 2.0f - 1.0f;
	while (glm::dot(v, v) > 1.0f)
		v = randomVec(state) * 2.0f - 1.0f;
#endif
	
	if (v == glm::vec3(0.0f))
		return { 1.0f, 0.0f ,0.0f };
	return glm::normalize(v);
}

__host__ __device__
inline glm::vec3 randomOnHemisphere(const glm::vec3& normal, curandState* state)
{
	glm::vec3 v = randomUnitVec(state);
	if (glm::dot(v, normal) > 0.0f)
		return v;
	return -v;
}