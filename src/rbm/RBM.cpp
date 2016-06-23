
#include "RBM.hpp"

struct RBM::RBMImpl {

  RBMImpl(unsigned visibleSize, unsigned hiddenSize) {}

  Vector ComputeHidden(const Vector &visible) { return Vector(); }

  Vector ComputeVisible(const Vector &hidden) { return Vector(); }

  void ApplyUpdate(const Matrix &weightsDelta) {}
};

RBM::RBM(unsigned visibleSize, unsigned hiddenSize) : impl(new RBMImpl(visibleSize, hiddenSize)) {}
RBM::~RBM() = default;

Vector RBM::ComputeHidden(const Vector &visible) { return impl->ComputeHidden(visible); }

Vector RBM::ComputeVisible(const Vector &hidden) { return impl->ComputeVisible(hidden); }

void RBM::ApplyUpdate(const Matrix &weightsDelta) { impl->ApplyUpdate(weightsDelta); }
