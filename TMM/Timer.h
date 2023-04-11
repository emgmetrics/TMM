#ifndef _TIMER_H
#define _TIMER_H
#include <windows.h>
class Timer
{
public:
    Timer();
    ~Timer();
    void Start();
    void Stop();
    double GetTime();
	void Reset();
private:
    LARGE_INTEGER m_frequency;
    LARGE_INTEGER m_startFrame;
    double m_accumulateTime;
};

#endif //_TIMER_H