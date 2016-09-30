#include "utils.h"
#include "TensorAls.h"
#include "CPDecomp.h"
#include "TensorData.h"
#include "SpAlsLinalg.h"

#include <memory>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std;

TensorCP_SPALS::TensorCP_SPALS(const TensorDataSpAls &_data, shared_ptr<CPDecomp> &_cpd, SpAlsRNGeng &_rngEng)
    : TensorCP_ALS(_data, _cpd), rngEng(_rngEng), rate(1.0), dataSpals(_data)
{
    lvrgScores = vector<vector<T>>(data.ro_dims.size());
    factorCmf = vector<vector<double>>(data.ro_dims.size());
    for (size_t factorId = 0; factorId < data.ro_dims.size(); factorId++)
    {
        lvrgScores[factorId] = vector<T>(data.ro_dims[factorId]);
        factorCmf[factorId] = vector<T>(data.ro_dims[factorId]);
        getLvrgScr(factorId);
    }
}
int TensorCP_SPALS::updateFactor(const unsigned factorId, size_t count)
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
    size_t hitR = 0;
    size_t hitNZ = 0;
    for (size_t iter = 0; iter < count; iter++)
    {
        // sample from krp
        vector<size_t> ps;
        for (auto &fid : froms)
        {
            // SpAlsUtils::printVector(factorCmf[fid]);
            ps.push_back(SpAlsUtils::drawFromCmf(factorCmf[fid], rngEng.nextRNG()));
            if (verbose > 3)
                cout << fid << "\t" << ps.back() << endl;
        }

        //get weight
        double weight = 1.0 / count;
        for (size_t it = 0; it < froms.size(); it++)
        {
            if (ps[it] == 0)
            {
                weight /= factorCmf[froms[it]][0];
            }
            else
            {
                weight /= factorCmf[froms[it]][ps[it]] - factorCmf[froms[it]][ps[it] - 1];
            }
        }

        // get corresponding entries
        int start = -1;
        int end = -1;
        dataSpals.findEntryFromFactor(factorId, ps, start, end);

        if (start >= 0)
        {
            hitR++;
            hitNZ += end - start + 1;
            for (int i = start; i <= end; i++)
            {
                int p = dataSpals.sortArgs[factorId][i];
                updateEntry(factorId, froms, p, gramABpdtInv[factorId], weight);
            }
        }
    }
    if (verbose)
        cout << hitR << "\t" << hitNZ << " " << count << endl;
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

int TensorCP_SPALS::updateFactor(const unsigned factorId)
{

    size_t count = pow(rank, data.ro_dims.size() - 1) * rate;
    for (auto &fid : data.ro_dims)
    {
        if (fid != factorId)
        {
            count *= max(1, (int)(log(data.ro_dims[fid])));
        }
    }
    return updateFactor(factorId, count);
}

void TensorCP_SPALS::getLvrgScr(const unsigned factorId)
{
    cpd->updateGram(factorId);

    size_t nid = data.ro_dims[factorId];
    SpAlsUtils::reset(lvrgScores[factorId]);

    auto &factor = cpd->factors[factorId];
    auto &gramMtxInv = cpd->getGramMtxInv(factorId);

#pragma omp parallel for
    for (int i = 0; i < nid; ++i)
    {
        for (size_t j1 = 0; j1 < rank; j1++)
        {
            for (size_t j2 = 0; j2 < rank; j2++)
            {
                lvrgScores[factorId][i] += factor[i][j1] * gramMtxInv[j1][j2] * factor[i][j2];
            }
        }
    }
    SpAlsUtils::pdf2Cmf(lvrgScores[factorId], factorCmf[factorId]);
}

void TensorCP_SPALS::setRate(double _rate)
{
    rate = _rate;
}