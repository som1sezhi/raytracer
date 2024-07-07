#include "Timer.h"

using namespace std::chrono;

Timer::Timer()
{
    Start();
}

void Timer::Start()
{
    m_StartTime = high_resolution_clock::now();
}

float Timer::GetElapsedSecs() const
{
    auto now = high_resolution_clock::now();
    auto ns = duration_cast<nanoseconds>(now - m_StartTime).count();
    return ns * 1e-9f;
}
