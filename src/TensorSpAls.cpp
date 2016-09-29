#include "utils.h"
#include "TensorAls.h"
#include "TensorSpAls.h"
#include "CPDecomp.h"
#include "TensorData.h"
#include "TensorDataSpAls.h"
#include "SpAlsLinalg.h"

#include <memory>
#include <vector>
#include <chrono>

using namespace std;

TensorCP_SPALS::TensorCP_SPALS(const TensorDataSpAls &_data, shared_ptr<CPDecomp> &_cpd) : TensorCP_ALS(_data, _cpd)
{
}

int TensorCP_SPALS::updateFactor(const unsigned factorId)
{
    return 0;
}
