#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"

class RBM {
public:
  RBM(unsigned visibleSize, unsigned hiddenSize);
  virtual ~RBM();

  Vector ComputeHidden(const Vector &visible);
  Vector ComputeVisible(const Vector &hidden);

  void ApplyUpdate(const Matrix &weightsDelta);

private:
  struct RBMImpl;
  uptr<RBMImpl> impl;
};
