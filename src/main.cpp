#include "utils.h"
#include "TensorData.h"
#include "TensorAls.h"
#include "CPDecomp.h"

#include <vector>
#include <memory>
#include <iostream>
#include <cmath>
#include <omp.h>

// #include "spals.h"

using namespace std;

int main(int argc, char *argv[])
{

    // string inputFilename("../checkin.txt");
    string inputFilename(argv[1]);
    size_t rank = 10;
    size_t nthread = 8;
    omp_set_num_threads(nthread);

    cout << inputFilename << endl;
    TensorDataSpAls data(inputFilename.c_str());
    data.verbose = 1;
    data.printData();

    cout << data.normData() << endl;
    cout << "Sorting Indexes!" << endl;
    data.sortIndexes();

    cout << "Create cpd!" << endl;

    shared_ptr<CPDecomp> cpd = make_shared<CPDecomp>(data, rank);

    cout << "Create SpAlsRNGeng!" << endl;
    vector<SpAlsRNGeng> rngEng(nthread);
    for (int tid = 0; tid < nthread; tid++)
    {
        rngEng[tid].seed((tid + 1984) * rand());
        // rngEng.push_back(move(SpAlsRNGeng(tid)));
    }

    cout << "randInit cpd!" << endl;
    cpd->randInit(rngEng.data());

    cout << "Init. Tensor CP-ALS!" << endl;
    TensorCP_SPALSOMP als(data, cpd, rngEng.data(), nthread);
    // TensorCP_ALS als(data, cpd);
    cout << "TensorCP_ALs init done" << endl;

    als.setErrorRecordInterval(1);
    als.setVerbose(1);
    als.setRate(2.0);
    for (int iter = 0; iter < 3; iter++)
    {
        // cout << "Iteration:\t" << iter << endl;
        for (size_t factorId = 0; factorId < data.ro_dims.size(); factorId++)
        {
            // cout << "Update factor:\t" << factorId << endl;
            als.updateFactor(factorId);
            // return 0;
        }
    }
    return 0;
}