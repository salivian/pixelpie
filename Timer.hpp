#ifndef __TIMER__
#define __TIMER__

#if defined _WIN32 || defined _WIN64

#include <windows.h>

class Timer{
 private:
  LARGE_INTEGER freq_,start_,stop_;

 public:
  void start(){
    QueryPerformanceCounter(&start_);
  }

  double stop(){
    QueryPerformanceCounter(&stop_);
    QueryPerformanceFrequency(&freq_);
    return ((stop_.QuadPart - start_.QuadPart) * 1.0 / freq_.QuadPart);
  }
};

#else //*nix Timer

class Timer{
 private:
  struct timespec start_, stop_;

 public:
  double timeDiff(const timespec& starttime,const timespec& stoptime){
    return (stoptime.tv_sec - starttime.tv_sec)
        +(stoptime.tv_nsec - starttime.tv_nsec)*1e-9;
  }
  
  void start(){
    clock_gettime(CLOCK_MONOTONIC,&start_);
  }
  
  double stop(){
    clock_gettime(CLOCK_MONOTONIC,&stop_);
    return timeDiff(start_,stop_);
  }
};
  
#endif

#endif
