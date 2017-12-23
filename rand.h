#ifndef _RAND_H
#define _RAND_H

#include <random>

template<typename T>
class MT19937 {
  std::mt19937 * engine;
  std::uniform_real_distribution<T> * dist;
 public:
  MT19937(T min, T max) {
    auto const seed = std::random_device()();
    engine = new std::mt19937(seed);
    dist = new std::uniform_real_distribution<T>(min, max);
  }
  ~MT19937() {
    if (dist) { delete dist; dist = 0; }
    if (engine) { delete engine; engine = 0; }
  }
  T next() { return (*dist)(*engine); }
};
#endif

