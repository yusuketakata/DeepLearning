#pragma once

#include <iostream>
#include <random>
#include <algorithm>
#include "eigen-eigen-10219c95fe65\Eigen\Core"

// *** ê≥âÉâÉxÉãÇópÇ¢ÇÈç€ÇÃä÷êî *** //
void random_sort(std::vector<std::vector<float>>& patch_data, std::vector<std::vector<unsigned char>>& patch_answer,
	Eigen::MatrixXf& PatchMatrix, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& AnswerMatrix, int num_case, int patch_size)
{

	std::vector<float> Patch;
	std::vector<unsigned char> Answer;

	long long num_patch = 0;

	for (size_t n_case = 0; n_case < num_case; n_case++)
	{
		std::vector<float> patch;
		std::vector<unsigned char> answer;


		Patch.insert(Patch.end(), patch_data[n_case].begin(), patch_data[n_case].end());
		Answer.insert(Answer.end(), patch_answer[n_case].begin(), patch_answer[n_case].end());

	}

	num_patch = Patch.size() / patch_size;

	std::cout << "patch number = " << num_patch << std::endl;

	std::vector<int> shuffle;
	for (int i = 0; i < num_patch; i++){
		shuffle.push_back(i);
	}

	std::cout << "///// SHUFFLE DATA NOW /////" << std::endl;
	srand(0);
	random_shuffle(shuffle.begin(), shuffle.end());

	Eigen::MatrixXf PatchMatrixCopy = PatchMatrixCopy.Zero(patch_size, num_patch);
	PatchMatrix = PatchMatrix.Zero(patch_size, num_patch);
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> AnswerMatrixCopy = AnswerMatrixCopy.Zero(2, num_patch);
	AnswerMatrix = AnswerMatrix.Zero(2, num_patch);


	for (long long yy = 0; yy < num_patch; yy++){
		for (int xx = 0; xx < patch_size; xx++){
			PatchMatrixCopy(xx, yy) = Patch[yy * patch_size + xx];
		}
	}

	for (long long yy = 0; yy < num_patch; yy++){
		for (int xx = 0; xx < 2; xx++){
			AnswerMatrixCopy(xx, yy) = Answer[yy * 2 + xx];
		}
	}

	for (long long yy = 0; yy < num_patch; yy++){
		for (int xx = 0; xx < patch_size; xx++){
			PatchMatrix(xx, yy) = PatchMatrixCopy(xx, shuffle[yy]);
		}
	}

	for (long long yy = 0; yy < num_patch; yy++){
		for (int xx = 0; xx < 2; xx++){
			AnswerMatrix(xx, yy) = AnswerMatrixCopy(xx, shuffle[yy]);
		}
	}

}


// *** Autoencoderóp *** //
void random_sort(std::vector<std::vector<float>>& noise_data, std::vector<std::vector<float>>& patch_answer,
	Eigen::MatrixXf& NoiseMatrix, Eigen::MatrixXf& AnswerMatrix, int num_case, int patch_size)
{

	std::vector<float> Patch;
	std::vector<float> Answer;

	long long num_patch = 0;

	for (size_t n_case = 0; n_case < num_case; n_case++)
	{
		std::vector<float> patch;
		std::vector<float> answer;


		Patch.insert(Patch.end(), noise_data[n_case].begin(), noise_data[n_case].end());
		Answer.insert(Answer.end(), patch_answer[n_case].begin(), patch_answer[n_case].end());

	}

	num_patch = Patch.size() / patch_size;

	std::cout << "patch number = " << num_patch << std::endl;

	std::vector<int> shuffle;
	for (int i = 0; i < num_patch; i++){
		shuffle.push_back(i);
	}

	std::cout << "///// SHUFFLE DATA NOW /////" << std::endl;
	srand(0);
	random_shuffle(shuffle.begin(), shuffle.end());

	Eigen::MatrixXf NoiseMatrixCopy = NoiseMatrixCopy.Zero(patch_size, num_patch);
	NoiseMatrix = NoiseMatrix.Zero(patch_size, num_patch);
	Eigen::MatrixXf AnswerMatrixCopy = AnswerMatrixCopy.Zero(patch_size, num_patch);
	AnswerMatrix = AnswerMatrix.Zero(patch_size, num_patch);


	for (long long yy = 0; yy < num_patch; yy++){
		for (int xx = 0; xx < patch_size; xx++){
			NoiseMatrixCopy(xx, yy) = Patch[yy * patch_size + xx];
		}
	}

	for (long long yy = 0; yy < num_patch; yy++){
		for (int xx = 0; xx < patch_size; xx++){
			AnswerMatrixCopy(xx, yy) = Answer[yy * patch_size + xx];
		}
	}

	for (long long yy = 0; yy < num_patch; yy++){
		for (int xx = 0; xx < patch_size; xx++){
			NoiseMatrix(xx, yy) = NoiseMatrixCopy(xx, shuffle[yy]);
		}
	}

	for (long long yy = 0; yy < num_patch; yy++){
		for (int xx = 0; xx < patch_size; xx++){
			AnswerMatrix(xx, yy) = AnswerMatrixCopy(xx, shuffle[yy]);
		}
	}

}

void random_sort(std::vector<std::vector<float>>& patch_data, Eigen::MatrixXf& PatchMatrix, int num_case, int patch_size)
{
	std::vector<float> Patch;

	long long num_patch = 0;

	for (size_t n_case = 0; n_case < num_case; n_case++)
	{
		std::vector<float> patch;
		std::vector<unsigned char> answer;


		Patch.insert(Patch.end(), patch_data[n_case].begin(), patch_data[n_case].end());
	}

	std::cout << Patch.size() << std::endl;
	std::cout << patch_size << std::endl;

	num_patch = Patch.size() / patch_size;

	std::cout << "patch number = " << num_patch << std::endl;

	std::vector<int> shuffle;
	for (int i = 0; i < num_patch; i++){
		shuffle.push_back(i);
	}

	std::cout << "///// SHUFFLE DATA NOW /////" << std::endl;
	srand(0);
	random_shuffle(shuffle.begin(), shuffle.end());

	Eigen::MatrixXf PatchMatrixCopy = PatchMatrixCopy.Zero(patch_size, num_patch);
	PatchMatrix = PatchMatrix.Zero(patch_size, num_patch);


	for (long long yy = 0; yy < num_patch; yy++){
		for (int xx = 0; xx < patch_size; xx++){
			PatchMatrixCopy(xx, yy) = Patch[yy * patch_size + xx];
		}
	}


	for (long long yy = 0; yy < num_patch; yy++){
		for (int xx = 0; xx < patch_size; xx++){
			PatchMatrix(xx, yy) = PatchMatrixCopy(xx, shuffle[yy]);
		}
	}

}