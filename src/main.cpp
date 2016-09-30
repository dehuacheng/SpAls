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
    omp_set_num_threads(4);
    // string inputFilename("../checkin.txt");
    string inputFilename("/home/dehua/code/data/tensorTest.csv");
    cout << inputFilename << endl;

    TensorDataSpAls data(inputFilename.c_str());
    cout << "Book1.csv!" << endl;
    data.verbose = 1;
    data.printData();

    cout << data.normData() << endl;
    cout << "Sorting Indexes!" << endl;
    data.sortIndexes();

    cout << "Create cpd!" << endl;
    size_t rank = 3;
    shared_ptr<CPDecomp> cpd = make_shared<CPDecomp>(data, rank);

    sitmo::prng_engine rngEngSeed;
    rngEngSeed.seed(1);

    cout << "Create SpAlsRNGeng!" << endl;
    SpAlsRNGeng rngEng(rngEngSeed);

    cout << "randInit cpd!" << endl;
    cpd->randInit(&rngEng);

    cout << "Init. Tensor CP-ALS!" << endl;
    TensorCP_ALS als(data, cpd);
    cout << "TensorCP_ALs init done" << endl;

    als.setErrorRecordInterval(1);
    als.setVerbose(1);

    for (int iter = 0; iter < 100; iter++)
    {
        // cout << "Iteration:\t" << iter << endl;
        for (size_t factorId = 0; factorId < data.ro_dims.size(); factorId++)
        {
            // cout << "Update factor:\t" << factorId << endl;
            als.updateFactor(factorId);
        }
    }
    return 0;
}