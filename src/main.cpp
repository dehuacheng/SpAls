#include "utils.h"
#include "TensorData.h"
#include "TensorDataSpAls.h"
#include "CPDecomp.h"

#include <vector>
#include <iostream>
#include <cmath>
#include <omp.h>

// #include "spals.h"

using namespace std;

int main(int argc, char *argv[])
{
    // string inputFilename("../checkin.txt");
    cout << "Book1.csv!" << endl;
    string inputFilename("../Book1.csv");

    TensorDataSpAls data(inputFilename.c_str());
    cout << "Book1.csv!" << endl;
    data.verbose = 1;
    data.printData();

    cout << data.normData() << endl;
    cout << "Sorting Indexes!" << endl;
    data.sortIndexes();

    cout << "Create cpd!" << endl;
    size_t rank = 3;
    CPDecomp cpd(data, rank);

    puts("Create cpd!");
    sitmo::prng_engine rngEngSeed;
    rngEngSeed.seed(1);

    cout << "Create SpAlsRNGeng!" << endl;
    SpAlsRNGeng rngEng(rngEngSeed);

    cout << "randInit cpd!" << endl;
    cpd.randInit(&rngEng);

    return 0;
}