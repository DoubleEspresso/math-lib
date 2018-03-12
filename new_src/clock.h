#ifndef MATHLIB_CLOCK_H
#define MATHLIB_CLOCK_H

#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

class Clock {
  struct timeval strt, ed;
  const char * tag;
 public:
 Clock(const char * t, bool start_now=true) : tag(t) { if (start_now) start(); }
  ~Clock() { }
  void start() { gettimeofday(&strt, NULL); }
  void stop() { gettimeofday(&ed, NULL); }
  double seconds() { return (double)(ed.tv_sec - strt.tv_sec); }
  double useconds() { return (double)(ed.tv_usec - strt.tv_usec); }
  double ms() { return ((seconds()) * 1000 + useconds() / 1000); }
  double elapsed_sec() { stop(); double secs = seconds(); start(); return secs; }
  double elapsed_ms() { stop(); double msecs = ms(); start(); return msecs; } 
  void finished() {
    stop(); double msecs = ms();
    printf("..%s finished in %4.2f(ms)\n", tag, msecs);
  }
};

#endif
