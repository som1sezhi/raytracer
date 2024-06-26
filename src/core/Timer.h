#pragma once

#include <chrono>

class Timer
{
public:
	Timer();
	void Start();
	float GetElapsedSecs() const;
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTime;
};