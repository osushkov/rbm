
#include "Util.hpp"
#include <cmath>
#include <cstdlib>

float Util::RandInterval(float s, float e) { return s + (e - s) * (rand() / (float)RAND_MAX); }

float Util::GaussianSample(float mean, float sd) {
  // Taken from GSL Library Gaussian random distribution.
  float x, y, r2;

  do {
    // choose x,y in uniform square (-1,-1) to (+1,+1)
    x = -1.0f + 2.0f * RandInterval(0.0f, 1.0f);
    y = -1.0f + 2.0f * RandInterval(0.0f, 1.0f);

    // see if it is in the unit circle
    r2 = x * x + y * y;
  } while (r2 > 1.0f || r2 < 0.0001f);

  // Box-Muller transform
  return mean + sd * y * sqrtf(-2.0f * logf(r2) / r2);
}
