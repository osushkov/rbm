#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"

class RBM {
public:
  RBM(unsigned visibleSize, unsigned hiddenSize);
  virtual ~RBM();

  Matrix ComputeHidden(const Matrix &visibleBatch);
  Matrix ComputeVisible(const Matrix &hiddenBatch);

  float Energy(const Vector &visible, const Vector &hidden);

  void ApplyUpdate(const Matrix &weightsDelta);

  unsigned NumVisibleUnits(void) const;
  unsigned NumHiddenUnits(void) const;
  Matrix GetWeights(void) const;

private:
  struct RBMImpl;
  uptr<RBMImpl> impl;
};
