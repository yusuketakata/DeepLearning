#pragma once
#include "Header.h"

template<class INPUT_T>
void sigmoid(INPUT_T& x, INPUT_T& y)
{
	y = 1.0 / (1.0 + (x.array() * -1).exp());
}

/**** sigmoidの微分 (sigmoidを通した後の値を想定しています) ****/
template<class INPUT_T>
void sigmoid_deriv(INPUT_T& x, INPUT_T& y)
{
	y = x.array() * (1.0 - x.array());
}

template<class INPUT_T>
void ReLU(INPUT_T& x, INPUT_T& y)
{
	y = (1.0 + x.array().exp()).array().log();
}

/**** ReLUの微分 (ReLUを通す前の値を想定しています) ****/
template<class INPUT_T>
void ReLU_deriv(INPUT_T& x, INPUT_T& y)
{
	y = 1.0 / (1.0 + (x.array() * -1).exp());
}

/**** ReLUの微分に入れる値を計算するためにReLUを通す前の値を算出 ****/
template<class INPUT_T>
void ReLU_invert(INPUT_T& x, INPUT_T& y)
{
	y = (x.array().exp() - 1.0).array().log();

}


template<class INPUT_T>
void softmax(INPUT_T& x, INPUT_T& y)
{
	Eigen::VectorXf temp;
	temp = (x.array() - x.maxCoeff()).exp();
	y = temp / temp.sum();
}

/**** argmax (名前の通り) ****/
void argmax(Eigen::MatrixXf& x, Eigen::MatrixXi& y)
{
	y = y.Zero(1, x.cols());
	for (int i = 0; i < (int)x.cols(); i++)
	{
		float out_max = x(0, i);
		int index = 0;
		for (int j = 1; j < (int)x.rows(); j++)
		{
			if (out_max < x(j, i)){
				out_max = x(j, i);
				index = j;
			}
		}
		y(0, i) = index;
	}
}

/**** 入力は(入力マップ数×出力マップ数,フィルタサイズ)を想定しています ****/
template<class INPUT_T>
void rot180(INPUT_T& X, INPUT_T& rotX)
{
	int row = X.rows();
	int col = X.cols();

	INPUT_T stockX = stockX.Zero(row, col);

	for (int y = 0; y < row; y++){
		for (int x = 0; x < col; x++){
			stockX(y, x) = X(y, col - x - 1);
		}
	}
	rotX = stockX;
}

/**** 入力サンプルをランダムソートして入れ替えます ****/
template<class INPUT_T, class LABEL_T>
void shuffle(INPUT_T& x, INPUT_T& sort_x, LABEL_T& y, LABEL_T& sort_y)
{
	INPUT_T stock_x;
	LABEL_T stock_y;
	for (int i = 0; i < (int)x.cols(); i++)
	{
		int index = rand() % (int)x.cols();
		stock_x = x.col(i);
		stock_y = y.col(i);
		x.col(i) = x.col(index);
		x.col(index) = stock_x;
		y.col(i) = y.col(index);
		y.col(index) = stock_y;
	}
	sort_x = x;
	sort_y = y;
}


////// 入出力用Utility //////////////

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	return split(s, delim, elems);
}


std::string Replace(std::string String1, std::string String2, std::string String3)
{
	std::string::size_type  Pos(String1.find(String2));

	while (Pos != std::string::npos)
	{
		String1.replace(Pos, String2.length(), String3);
		Pos = String1.find(String2, Pos + String3.length());
	}

	return String1;
}