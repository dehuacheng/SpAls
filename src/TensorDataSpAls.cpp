#include "utils.h"
#include "TensorData.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std;

TensorDataSpAls::TensorDataSpAls(const char *filename) : TensorData(filename), isSorted(false), isCmfReady(false)
{
    dataCmf = vector<double>(nnz);
    SpAlsUtils::pdf2Cmf(val, dataCmf);
}

void TensorDataSpAls::toFile(const char *filename)
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
    fwrite(dataCmf.data(), sizeof(T), nnz, f_out);
    for (size_t i = 0; i < NDIM; ++i)
    {
        fwrite(sortArgs[i].data(), sizeof(size_t), nnz, f_out);
    }

    fclose(f_out);

    auto end = chrono::system_clock::now();
    diffTime += end - start;
    if (verbose)
        printf("Time elapsed saving data: %f seconds\n", diffTime.count());
}

void TensorDataSpAls::fromFile(const char *filename)
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
        cout << "Loading processed data from " << filename << endl;

    size_t NDIM = 0;
    frlength = fread(&NDIM, sizeof(size_t), 1, f_in);
    if (verbose > 2)
        cout << "NDim:\t" << NDIM << endl;

    dims = vector<size_t>(NDIM);
    frlength = fread(dims.data(), sizeof(size_t), NDIM, f_in);
    frlength = fread(&nnz, sizeof(size_t), 1, f_in);
    if (verbose > 2)
        cout << "nnz:\t" << nnz << endl;

    frlength = fread(&normT, sizeof(T), 1, f_in);
    loc = vector<vector<size_t>>(nnz, vector<size_t>(NDIM));
    for (size_t i = 0; i < nnz; ++i)
    {
        frlength = fread(loc[i].data(), sizeof(size_t), NDIM, f_in);
    }

    val = vector<T>(nnz);
    frlength = fread(val.data(), sizeof(double), nnz, f_in);

    dataCmf = vector<T>(nnz);
    frlength = fread(dataCmf.data(), sizeof(T), nnz, f_in);

    sortArgs = vector<vector<size_t>>(NDIM, vector<size_t>(nnz));
    for (size_t i = 0; i < NDIM; ++i)
    {
        frlength = fread(sortArgs[i].data(), sizeof(size_t), nnz, f_in);
    }
    fclose(f_in);

    auto end = chrono::system_clock::now();
    diffTime += end - start;
    if (verbose)
        printf("Time elapsed loading data: %f seconds\n", diffTime.count());

    isSorted = true;
    isCmfReady = true;
}

void TensorDataSpAls::printData(int did)
{
    auto NDIM = dims.size();
    if (did < 0 || did >= NDIM)
    {
        TensorData::printData();
    }
    else
    {
        printDataStats();
        for (int sid = 0; sid < nnz; sid++)
        {
            int nid = sortArgs[did][sid];
            for (size_t i = 0; i < NDIM; ++i)
            {
                printf("%ld\t", loc[nid][i]);
            }
            printf("%lf\n", val[nid]);
        }
    }
}

void TensorDataSpAls::sortIndex(const int notFrom, size_t *s)
{
    for (size_t i = 0; i < nnz; ++i)
        s[i] = i;

    auto froms = SpAlsUtils::getFroms(notFrom, dims.size());
    myclass sort1(loc, froms);
    sort(s, s + nnz, sort1);
}

void TensorDataSpAls::sortIndexes()
{
    auto NDIM = dims.size();

    for (size_t i = 0; i < NDIM; ++i)
    {
        if (verbose)
            cout << "sorting dimension: " << i << endl;

        if (i == sortArgs.size())
            sortArgs.push_back(vector<size_t>(nnz));

        if (sortArgs[i].size() != nnz)
        {
            sortArgs[i] = vector<size_t>(nnz);
        }

        sortIndex(i, sortArgs[i].data());
    }

    if (verbose)
        puts("Sorting fininshed");

    isSorted = true;
}

void TensorDataSpAls::findEntryFromFactor(const size_t factorId, const vector<size_t> &ps, int &start, int &end) const
{
    vector<size_t> froms = SpAlsUtils::getFroms(factorId, dims.size());
    findIndexLoc(froms, ps, sortArgs[factorId], start, end);
    return;
}

void TensorDataSpAls::findIndexLoc(
    const vector<size_t> &froms,
    const vector<size_t> &is,
    const vector<size_t> &s,
    int &start,
    int &end) const
{
    // find start/end, such that start<= i <= end
    // loc[s[i]][from1,from2] = [i1,i2]
    // return start = -1 when empty;
    auto NDIM = dims.size();
    vector<size_t> pivot(NDIM, -1);

    for (int i = 0; i < froms.size(); i++)
    {
        pivot[froms[i]] = is[i];
    }

    if (EqThan(pivot, loc[s[0]], froms))
    {
        start = 0;
    }
    else if (lessThan(pivot, loc[s[nnz - 1]], froms))
    {
        start = -1;
    }
    else if (lessThan(pivot, loc[s[0]], froms))
    {
        start = findIndexLocBiSearchStart(
            froms,
            pivot,
            s,
            1,
            nnz - 1);
    }
    else
    {
        start = -1;
    }

    if (start >= 0)
    {
        int s1 = 0;
        while (EqThan(pivot, loc[s[start + s1]], froms))
        {
            s1++;
            if (start + s1 >= nnz)
            {
                break;
            }
        }
        end = start + s1 - 1;
        return;
    }
    else
    {
        // there is no matching entry.
        return;
    }
}

int TensorDataSpAls::findIndexLocBiSearchStart(
    const vector<size_t> &froms,
    const vector<size_t> &pivot,
    const vector<size_t> &s,
    int start,
    int end) const
{
    if (start > end)
    {
        return -1;
    }
    else if (start == end)
    {
        if (EqThan(pivot, loc[s[start]], froms))
        {
            return start;
        }
        else
        {
            return -1;
        }
    }
    else
    {
        int mid = start + (end - start) / 2;

        if (lessThan(pivot, loc[s[mid]], froms))
        {
            return findIndexLocBiSearchStart(
                froms,
                pivot,
                s,
                mid + 1,
                end);
        }
        else
        {
            return findIndexLocBiSearchStart(
                froms,
                pivot,
                s,
                start,
                mid);
        }
    }
}