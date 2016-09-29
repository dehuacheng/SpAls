#pragma once
class Linalg;

#include "CPDecomp.h"
#include "TensorData.h"

class Linalg
{
  public:
    static double Fnorm2(const TensorData &data);
    static double Fnorm2(CPDecomp &cpd);
    static double Fnorm2Diff(const TensorData &data, CPDecomp &cpd);
};