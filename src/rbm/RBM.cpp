
#include "RBM.hpp"
#include "../common/Util.hpp"
#include <cmath>

struct RBM::RBMImpl {
  // Top left corner of weights matrix is the W matrix in RBM.
  // the rightmost column is b, the hidden unit bias.
  // the bottom row is a, the visible unit bias.

  Matrix weights;          // for computing visible -> hidden activation.
  Matrix transposeWeights; // for computing hidden -> visible activation.

  unsigned visibleSize;
  unsigned hiddenSize;

  RBMImpl(unsigned visibleSize, unsigned hiddenSize)
      : visibleSize(visibleSize), hiddenSize(hiddenSize) {
    assert(visibleSize > 0 && hiddenSize > 0);
    initialiseWeights(visibleSize, hiddenSize);
  }

  Matrix ComputeHidden(const Matrix &visibleBatch) {
    Matrix biased = getBatchMatrixWithBias(visibleBatch);
    Matrix activations = weights * biased;
    assert(activations.rows() == hiddenSize + 1);
    assert(activations.cols() == visibleBatch.cols());

    Matrix hiddenValues(hiddenSize, visibleBatch.cols());
    for (int c = 0; c < hiddenValues.cols(); c++) {
      for (int r = 0; r < hiddenValues.rows(); r++) {
        float sample = Util::RandInterval(0.0f, 1.0f);
        hiddenValues(r, c) = sample < activationProbability(activations(r, c)) ? 1.0f : 0.0f;
      }
    }
    return hiddenValues;
  }

  Matrix ComputeVisible(const Matrix &hiddenBatch) {
    Matrix biased = getBatchMatrixWithBias(hiddenBatch);
    Matrix activations = transposeWeights * biased;
    assert(activations.rows() == visibleSize + 1);
    assert(activations.cols() == hiddenBatch.cols());

    Matrix visibleValues(visibleSize, hiddenBatch.cols());
    for (int c = 0; c < visibleValues.cols(); c++) {
      for (int r = 0; r < visibleValues.rows(); r++) {
        float sample = Util::RandInterval(0.0f, 1.0f);
        visibleValues(r, c) = sample < activationProbability(activations(r, c)) ? 1.0f : 0.0f;
      }
    }
    return visibleValues;
  }

  void ApplyUpdate(const Matrix &weightsDelta) {
    assert(weightsDelta.rows() == weights.rows());
    assert(weightsDelta.cols() == weights.cols());
    weights += weightsDelta;
    transposeWeights = weights.transpose();
  }

  float activationProbability(float in) { return 1.0f / (1.0f + expf(-in)); }

  Matrix getBatchMatrixWithBias(const Matrix &noBias) {
    Matrix result(noBias.rows() + 1, noBias.cols());
    for (int c = 0; c < noBias.cols(); c++) {
      result(noBias.rows(), c) = 1.0f;
    }
    result.topRightCorner(noBias.rows(), noBias.cols()) = noBias;
    return result;
  }

  void initialiseWeights(unsigned visibleSize, unsigned hiddenSize) {
    weights = Matrix(hiddenSize + 1, visibleSize + 1);
    weights.fill(0.0f);

    float initRange = 0.1f;
    // float initRange = 1.0f / sqrtf(weights.cols());

    for (int r = 0; r < weights.rows(); r++) {
      for (int c = 0; c < weights.cols(); c++) {
        weights(r, c) = Util::RandInterval(-initRange, initRange);
      }
    }

    transposeWeights = weights.transpose();
  }
};

RBM::RBM(unsigned visibleSize, unsigned hiddenSize) : impl(new RBMImpl(visibleSize, hiddenSize)) {}
RBM::~RBM() = default;

Matrix RBM::ComputeHidden(const Matrix &visibleBatch) { return impl->ComputeHidden(visibleBatch); }
Matrix RBM::ComputeVisible(const Matrix &hiddenBatch) { return impl->ComputeVisible(hiddenBatch); }

void RBM::ApplyUpdate(const Matrix &weightsDelta) { impl->ApplyUpdate(weightsDelta); }

unsigned RBM::NumVisibleUnits(void) const { return impl->visibleSize; }
unsigned RBM::NumHiddenUnits(void) const { return impl->hiddenSize; }

Matrix RBM::GetWeights(void) const { return impl->weights; }
