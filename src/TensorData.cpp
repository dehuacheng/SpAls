#include "TensorData.h"
#include "SpAlsLinalg.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;

TensorData::TensorData(const char *filename, unsigned _verbose) : verbose(_verbose), ro_loc(loc), ro_val(val), ro_nnz(nnz), ro_dims(dims)
{
    fstream fin;
    fin.open(filename, fstream::in);
    if (!fin.is_open())
    {
        printf("Cannot open file: %s \n", filename);
        exit(1);
    }

    chrono::duration<double> diffTime(0);
    auto start = chrono::system_clock::now();

    size_t NDIM = 0;
    fin >> NDIM;
    if (NDIM < 1 || NDIM >= MAXDIM)
    {
        printf("Dimension exceeds limits: %ld \n", NDIM);
        exit(1);
    }
    dims = vector<size_t>(NDIM);
    for (size_t i = 0; i < NDIM; i++)
    {
        dims[i] = 0;
    }

    fin >> nnz;
    if (nnz < 1)
    {
        printf("NNZ should be positive: %ld \n", nnz);
        exit(1);
    }
    loc = vector<vector<size_t>>(nnz, vector<size_t>(NDIM));
    val = vector<T>(nnz);

    for (size_t i = 0; i < nnz; ++i)
    {
        for (size_t idim = 0; idim < NDIM; idim++)
        {
            fin >> loc[i][idim];
            dims[idim] = max(dims[idim], loc[i][idim] + 1);
        }
        fin >> val[i];
    }

    auto end = chrono::system_clock::now();
    diffTime += end - start;

    normT = Linalg::Fnorm2(*this);

    if (verbose)
        printf("Data Loaded\n");
    if (verbose)
        printf("Time elapsed loading data: %f\n", diffTime.count());
    if (verbose)
    {
        printf("Read in %ld", dims[0]);
        for (size_t idim = 1; idim < NDIM; idim++)
        {
            printf("*%ld", dims[idim]);
        }
        printf(" tensor, with %ld nonzeros\n", nnz);
    }
}

void TensorData::toFile(const char *filename)
{
    if (verbose)
        printf("Saving Data to file: %s\n", filename);

    chrono::duration<double> diffTime(0);
    auto start = chrono::system_clock::now();

    FILE *f_out = fopen(filename, "w");
    if (f_out == NULL)
    {
        printf("Cannot open file: %s \n", filename);
        exit(1);
    }
    auto NDIM = dims.size();
    fwrite(&NDIM, sizeof(size_t), 1, f_out);
    fwrite(dims.data(), sizeof(size_t), NDIM, f_out);
    fwrite(&nnz, sizeof(size_t), 1, f_out);
    fwrite(&normT, sizeof(T), 1, f_out);

    for (size_t i = 0; i < nnz; ++i)
    {
        fwrite(loc[i].data(), sizeof(size_t), NDIM, f_out);
    }
    fwrite(val.data(), sizeof(T), nnz, f_out);

    fclose(f_out);

    auto end = chrono::system_clock::now();
    diffTime += end - start;
    if (verbose)
        printf("Time elapsed saving data: %f seconds\n", diffTime.count());
}

void TensorData::printData()
{
    auto NDIM = ro_dims.size();
    printDataStats();
    for (int did = 0; did < ro_nnz; did++)
    {
        for (size_t i = 0; i < NDIM; ++i)
        {
            printf("%ld\t", ro_loc[did][i]);
        }
        printf("%lf\n", ro_val[did]);
    }
}

void TensorData::printDataStats()
{
    auto NDIM = dims.size();
    cout << "Data Stats:" << endl;
    cout << "Tensor Order: " << NDIM << endl;

    for (size_t i = 0; i < NDIM; ++i)
    {
        cout << "Size of dim. " << i << ":\t" << dims[i] << endl;
    }
    cout << "NNZ in tensor: " << nnz << endl;
    cout << "Fnorm^2 tensor: " << normData() << endl;
}

T TensorData::normData() const
{
    return normT;
}

void TensorData::fromFile(const char *filename)
{
    size_t frlength;
    chrono::duration<double> diffTime(0);
    auto start = chrono::system_clock::now();

    FILE *f_in;
    f_in = fopen(filename, "r");
    if (f_in == NULL)
    {
        printf("Cannot open file: %s \n", filename);
        exit(1);
    }

    if (verbose)
        printf("Loading processed data from %s\n", filename);

    size_t NDIM = 0;
    frlength = fread(&NDIM, sizeof(size_t), 1, f_in);
    dims = vector<size_t>(NDIM);
    frlength = fread(dims.data(), sizeof(size_t), NDIM, f_in);
    frlength = fread(&nnz, sizeof(size_t), 1, f_in);
    frlength = fread(&normT, sizeof(T), 1, f_in);
    loc = vector<vector<size_t>>(nnz, vector<size_t>(NDIM));
    for (size_t i = 0; i < nnz; ++i)
    {
        frlength = fread(loc[i].data(), sizeof(size_t), NDIM, f_in);
    }

    val = vector<T>(nnz, 0.0);
    frlength = fread(val.data(), sizeof(double), nnz, f_in);
    fclose(f_in);

    normT = Linalg::Fnorm2(*this);

    auto end = chrono::system_clock::now();
    diffTime += end - start;
    if (verbose)
        printf("Time elapsed loading data: %f seconds\n", diffTime.count());
}
