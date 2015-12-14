#pragma once
#include<iostream>
#include"tktlib/eigen-eigen-10219c95fe65\Eigen\Sparse"
#include"tktlib/eigen-eigen-10219c95fe65\Eigen\Dense"
#include"tktlib/raw_io.h"
#include <cmath>
#include<random>
#include <omp.h> //•À—ñ‰»‚µ‚½‚¢‚Á‚Á‚·

std::mt19937_64 engine(65536);
std::uniform_real<float> dist;