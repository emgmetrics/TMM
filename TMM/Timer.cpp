#include "Timer.h"
const LARGE_INTEGER LargeIntZero = {0, 0};

Timer::Timer():
m_startFrame(LargeIntZero),
m_accumulateTime(0)
{
    QueryPerformanceFrequency(&m_frequency);
}

Timer::~Timer()
{
}

void Timer::Start()
{
    QueryPerformanceCounter(&m_startFrame);
	//m_accumulateTime = 0;
}

void Timer::Stop()
{
    LARGE_INTEGER endFrame;
    QueryPerformanceCounter(&endFrame);
    m_accumulateTime+= static_cast<double>(endFrame.QuadPart - m_startFrame.QuadPart) 
			/ static_cast<double>(m_frequency.QuadPart);

}

double Timer::GetTime()
{
    return m_accumulateTime;
}

void Timer::Reset()
{
	m_accumulateTime = 0;
}