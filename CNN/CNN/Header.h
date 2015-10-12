#pragma once
#include<iostream>
#include"eigen-eigen-10219c95fe65\Eigen\Sparse"
#include"eigen-eigen-10219c95fe65\Eigen\Dense"
#include"raw_io.h"
#include <cmath>
#include<random>
#include <omp.h>

std::mt19937_64 engine(65536);
std::uniform_real<float> dist;