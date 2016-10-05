#include "spals.h"

#include <vector>
#include <memory>
#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char *argv[])
{
    string inputFilename(argv[1]);
    size_t rank = 10;
    size_t nthread = 8;
    omp_set_num_threads(nthread);

    cout << "loading data from: " << inputFilename << endl;
    TensorDataSpAls data(inputFilename.c_str());
    data.verbose = 0;
    data.printDataStats();

    cout << "Sorting Indexes!" << endl;
    data.sortIndexes();

    cout << "Create cpd!" << endl;

    shared_ptr<CPDecomp> cpd = make_shared<CPDecomp>(data, rank);

    cout << "Create SpAlsRNGeng!" << endl;
    vector<SpAlsRNGeng> rngEng(nthread);
    for (int tid = 0; tid < nthread; tid++)
    {
        rngEng[tid].seed((tid + 1984) * rand());
    }

    cout << "randInit cpd!" << endl;
    cpd->randInit(rngEng.data());

    cout << "Init. Tensor CP-ALS!" << endl;
    TensorCP_SPALSOMP als(data, cpd, rngEng.data(), nthread);
    cout << "TensorCP_ALs init done" << endl;

    als.setErrorRecordInterval(1);
    als.setVerbose(0);
    als.setRate(2.0);
    for (int iter = 0; iter < 3; iter++)
    {
        for (size_t factorId = 0; factorId < data.ro_dims.size(); factorId++)
        {
            als.updateFactor(factorId);
        }
    }
    return 0;
}