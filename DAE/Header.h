#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <time.h>
#include <random>
#include "eigen-eigen-10219c95fe65\Eigen\Core"
#include "raw_io.h"


std::mt19937_64 engine(65536);
std::uniform_real<float> dist;
std::uniform_int<> dist_int;