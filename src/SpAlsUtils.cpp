#include "pRNG.h"
#include "utils.h"
#include "SpAlsLinalg.h"
#include "TensorData.h"

#include <omp.h>
#include <vector>
#include <algorithm>

using namespace std;

double SpAlsRNGeng::nextRNG()
{
    return rand01();
}

inline double SpAlsRNGeng::rand01()
{
    double x = 0;
    double rng = 1;
    for (int i = 0; i < 6; ++i)
    {
        x = x * double(1 << 10) + double(tmpeng() % (1 << 10));
        rng *= double(1 << 10);
    }
    return double(x) / double(rng);
}

vector<size_t> SpAlsUtils::getFroms(const int notFrom, const int NDIM)
{
    vector<size_t> froms;
    for (int i = 0; i < NDIM; i++)
        if (i != notFrom)
            froms.push_back(i);

    return froms;
}

void SpAlsUtils::reset(vector<vector<T>> &A)
{
    for (auto &row : A)
    {
        fill(row.begin(), row.end(), 0);
    }
}

void SpAlsUtils::printMatrix(const vector<vector<T>> &A)
{
    cout << "print matrix of size " << A.size() << "-by-" << A[0].size() << endl;
    for (auto const &row : A)
    {
        for (auto const &val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }
}
void SpAlsUtils::invert(const vector<vector<T>> &A, vector<vector<T>> &goal, const int N_MAX)
{

    T *a = new T[(N_MAX * (N_MAX + 1)) / 2];
    T *c = new T[(N_MAX * (N_MAX + 1)) / 2];
    int ifault;
    int nullty;
    T *w = new T[N_MAX];

    int k = 0;
    for (int i = 0; i < N_MAX; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            a[k] = A[i][j];
            k++;
        }
    }
    syminv(a, N_MAX, c, w, &nullty, &ifault);
    k = 0;
    for (int i = 0; i < N_MAX; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            goal[i][j] = c[k];
            goal[j][i] = c[k];
            k++;
        }
    }

    delete[] a;
    delete[] c;
    delete[] w;
}
