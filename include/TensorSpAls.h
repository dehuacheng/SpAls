#pragma once
class TensorCP_SPALS;

#include "pRNG.h"
#include "utils.h"
#include "CPDecomp.h"
#include "TensorData.h"
#include "TensorDataSpAls.h"

#include <vector>
#include <chrono>
#include <omp.h>
#include <memory>

using namespace std;

class TensorCP_SPALS : public TensorCP_ALS
{
  public:
    TensorCP_SPALS(const TensorDataSpAls &_data, shared_ptr<CPDecomp> &_cpd);
    virtual int updateFactor(const unsigned factorId);

  protected:
    // cmf for each factor
};