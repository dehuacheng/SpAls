#include "utils.h"
#include "SpAlsLinalg.h"
#include "TensorAls.h"
#include "CPDecomp.h"
#include "TensorData.h"

#include <memory>
#include <vector>
#include <chrono>

using namespace std;

TensorCP_ALS::TensorCP_ALS(const TensorData &_data, shared_ptr<CPDecomp> &_cpd) : currIter(0), data(_data), cpd(_cpd), rank(_cpd->rank)
{
    gramABpdt = vector<vector<vector<T>>>(data.dims.size(), vector<vector<T>>(rank, vector<T>(rank)));
    gramABpdtInv = vector<vector<vector<T>>>(data.dims.size(), vector<vector<T>>(rank, vector<T>(rank)));
}

chrono::time_point<chrono::system_clock> TensorCP_ALS::logBeforIter(const unsigned factorId)
{
    factorIter.push_back(factorId);

    return chrono::system_clock::now();
}

void TensorCP_ALS::setErrorRecordInterval(const size_t _interval)
{
    RecordError = _interval;
}

void TensorCP_ALS::setVerbose(const int _verbose)
{
    verbose = _verbose;
}

void TensorCP_ALS::logAfterIter(const unsigned factorId, chrono::time_point<chrono::system_clock> &stepStartTime)
{
    // log after iteration
    if (RecordError)
    {
        if (currIter % RecordError == 0)
        {
            absErrorIter.push_back(pair<size_t, T>(currIter, Linalg::Fnorm2Diff(data, *cpd)));
            rltErrorIter.push_back(pair<size_t, T>(currIter, absErrorIter.back().second / data.normData()));
            if (verbose)
            {
                cout << "Iteration:\t" << currIter << "\t Factor:" << factorId << "\n\tAbs. Err.:"
                     << absErrorIter.back().second << "\tRel. Err.:" << rltErrorIter.back().second << endl;
            }
        }
    }
    //record time
    auto stepEndTime = chrono::system_clock::now();
    chrono::duration<double> stepTime(0);
    stepTime += stepEndTime - stepStartTime;
    secIter.push_back(stepTime.count());
}

int TensorCP_ALS::updateFactor(const unsigned factorId)
{
    // log before iteration
    auto stepStartTime = logBeforIter(factorId);

    // clear the
    cpd->lambdas = vector<T>(rank, 1.0);
    SpAlsUtils::reset(cpd->factors[factorId]);
    cpd->isFactorNormalized[factorId] = false;
    cpd->isGramUpdated[factorId] = false;
    cpd->isGramInvUpdated[factorId] = false;

    //get inverse gram matrix
    prepareGramInv(factorId);
    cpd->isFactorNormalized[factorId] = false;
    cpd->isGramUpdated[factorId] = false;
    cpd->isGramInvUpdated[factorId] = false;

    vector<size_t> froms = SpAlsUtils::getFroms(factorId, cpd->dims.size());
    for (size_t i = 0; i < data.nnz; i++)
    {
        updateEntry(factorId, froms, i, gramABpdtInv[factorId], 1.0);
    }

    //postprocessing of the new factor
    if (verbose > 2)
    {
        cout << "print Factor Matrix Before Normalization:\n";
        SpAlsUtils::printMatrix(cpd->ro_factors[factorId]);
    }
    cpd->normalizeFactor(factorId);
    if (verbose > 2)
    {
        cout << "print Factor Matrix After Normalization:\n";
        SpAlsUtils::printMatrix(cpd->ro_factors[factorId]);
    }

    cpd->updateGram(factorId);
    if (verbose > 2)
    {
        cout << "\t\tgram matrix for factor after update:" << factorId << endl;
        SpAlsUtils::printMatrix(cpd->getGramMtx(factorId));
    }
    logAfterIter(factorId, stepStartTime);
    currIter++;
    return 0;
}

void TensorCP_ALS::prepareGramInv(const size_t factorId)
{
    vector<size_t> froms = SpAlsUtils::getFroms(factorId, cpd->dims.size());

    auto &gramAB = gramABpdt[factorId];
    auto &gramABInv = gramABpdtInv[factorId];

    SpAlsUtils::reset(gramAB);
    SpAlsUtils::reset(gramABInv);
    for (size_t i = 0; i < rank; i++)
    {
        for (size_t j = 0; j < rank; j++)
        {
            gramAB[i][j] = 1.0;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < rank; i++)
    {
        for (auto &fid : froms)
        {
            auto &gramMtx = cpd->getGramMtx(fid);
            for (size_t j = 0; j < rank; j++)
            {
                gramAB[i][j] *= gramMtx[i][j];
            }
        }
    }
    SpAlsUtils::invert(gramAB, gramABInv, rank);
    if (verbose > 2)
    {
        cout << "print Gram Matrix:\n";
        SpAlsUtils::printMatrix(gramAB);
        for (auto &fid : froms)
        {
            cout << "\t\tgram matrix for factor:" << fid << endl;
            SpAlsUtils::printMatrix(cpd->getGramMtx(fid));
        }
        cout << "print Inverse Gram Matrix:\n";
        SpAlsUtils::printMatrix(gramABInv);
    }
}

void TensorCP_ALS::updateEntry(const unsigned factorId, const vector<size_t> &froms,
                               const size_t i, const vector<vector<T>> &gramABInv, const T weight)
{
    vector<T> _row = genRow(factorId, froms, data.loc[i], gramABInv);
    for (size_t j = 0; j < rank; ++j)
    {
        cpd->factors[factorId][data.loc[i][factorId]][j] += data.val[i] * _row[j] * weight;
    }
}

vector<T> TensorCP_ALS::genRow(const unsigned factorId, const vector<size_t> &froms, const vector<size_t> &_loci, const vector<vector<T>> &pinv)
{
    vector<T> row(rank, 0);
    for (size_t j = 0; j < rank; j++)
    {
        T krpJ = cpd->factors[froms[0]][_loci[froms[0]]][j] * cpd->factors[froms[1]][_loci[froms[1]]][j];
        for (size_t k = 0; k < rank; k++)
        {
            row[k] += pinv[k][j] * krpJ;
        }
    }
    return row;
}