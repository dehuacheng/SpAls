#pragma once
class TensorData;
class TensorDataSpAls;

#include "utils.h"
#include "TensorAls.h"
// #include "TensorSpAls.h"

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

class TensorDataSpAls : public TensorData
{
    friend class TensorCP_SpALS;

  public:
    TensorDataSpAls() : TensorData(), isSorted(false), isCmfReady(false){};
    TensorDataSpAls(const char *filename);

    void toFile(const char *filename) override;
    void fromFile(const char *filename) override;
    void printData(int did = -1);

    unsigned verbose;
    void sortIndexes();
    void findEntryFromFactor(const size_t factorId, const vector<size_t> &ps, int &start, int &end) const;

    vector<vector<size_t>> sortArgs;

  protected:
    vector<double> dataCmf;
    bool isSorted;
    bool isCmfReady;

    void sortIndex(const int notFrom, size_t *s);

    int findGreaterThan(const vector<double> &cmf, const int first, const int last, double p);

    void findIndexLoc(
        const vector<size_t> &froms,
        //const int from1, const int from2,
        const vector<size_t> &is,
        //, const int i2,
        const vector<size_t> &s,
        int &start,
        int &end) const;

    int findIndexLocBiSearchStart(
        const vector<size_t> &froms,
        const vector<size_t> &pivot,
        const vector<size_t> &s,
        int start,
        int end) const;

    inline bool EqThan(const vector<size_t> &rpivot, const vector<size_t> &lpivot, const vector<size_t> &froms) const
    {
        for (int i = 0; i < froms.size(); i++)
        {
            if (lpivot[froms[i]] == rpivot[froms[i]])
            {
                continue;
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    inline bool lessThan(const vector<size_t> &rpivot, const vector<size_t> &lpivot, const vector<size_t> &froms) const
    {
        for (int i = 0; i < froms.size(); i++)
        {
            if (lpivot[froms[i]] == rpivot[froms[i]])
            {
                continue;
            }
            else
            {
                return lpivot[froms[i]] < rpivot[froms[i]];
            }
        }
        return false;
    }

    class myclass
    {
      public:
        // myclass(const vector<vector<size_t>> &_loc, const int _from1, const int _from2) :loc(_loc), from1(_from1), from2(_from2) {}
        // const int from1;
        // const int from2;

        myclass(const vector<vector<size_t>> &_loc, const vector<size_t> _froms) : loc(_loc), froms(_froms) {}
        const vector<size_t> froms;
        const vector<vector<size_t>> &loc;
        bool operator()(int i, int j)
        {
            for (int did = 0; did < froms.size(); did++)
            {
                int currFrom = froms[did];
                if (loc[j][currFrom] == loc[i][currFrom] && did + 1 < froms.size())
                {
                    continue;
                }
                else
                {
                    return loc[j][currFrom] > loc[i][currFrom];
                }
            }
        }
    };
};
