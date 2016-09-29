#pragma once

#include "utils.h"
class TensorData;
#include "TensorAls.h"

#include <vector>

using namespace std;

class TensorData
{
    friend class TensorCP_ALS;

  public:
    TensorData() : normT(-1), verbose(1), ro_loc(loc), ro_val(val), ro_nnz(nnz), ro_dims(dims) {}
    TensorData(const char *filename);

    virtual void toFile(const char *filename);
    virtual void fromFile(const char *filename);
    void printDataStats();
    void printData();

    T normData() const;

    const vector<vector<size_t>> &ro_loc;
    const vector<T> &ro_val;
    const size_t &ro_nnz;
    const vector<size_t> &ro_dims;

    unsigned verbose;

  protected:
    vector<vector<size_t>> loc;
    vector<T> val;
    vector<size_t> dims;
    T normT;

    size_t nnz;
    const int MAXDIM = 100;
};
