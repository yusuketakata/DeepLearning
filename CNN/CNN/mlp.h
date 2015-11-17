#pragma once
#include "Header.h"
#include "Utils.h"


class MLP
{
public:
	int numInput, numOutput; //入力層，出力層のユニット数
	int error_function;      // 誤差関数
	Eigen::MatrixXf weight1; //入力層-出力層の重み行列
	Eigen::VectorXf b1, b2; //入力層-出力層のバイアスベクトル

	std::string act1;

	MLP(){};
	~MLP(){};

	MLP(std::string, int, int, std::string, int, int, int);

	void train(Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, float, Eigen::MatrixXf&, float&, float, float);
	void valid_test(Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, float&);
	void forward_propagation(Eigen::MatrixXf&, Eigen::MatrixXf&);

};



MLP::MLP(std::string dir_p, int _numInput, int _numOutput, std::string _act1, int _error_function, int ini_switch, int cv_loop)
{
	numInput = _numInput;
	numOutput = _numOutput;

	act1 = _act1;

	error_function = _error_function;

	weight1 = Eigen::MatrixXf::Zero(numOutput, numInput);
	dist = std::uniform_real_distribution<float>(-(float)sqrt(6.0 / (float)(numOutput + numInput)), (float)sqrt(6.0 / (float)(numOutput + numInput)));

	if (ini_switch == 0){
		for (int j = 0; j < (int)weight1.rows(); j++)
		{
			for (int i = 0; i < (int)weight1.cols(); i++)
			{
				weight1(j, i) = dist(engine);

			}
		}
	}
	else if (ini_switch == 1){

		std::vector<float> W_dummy;
		read_vector(W_dummy, dir_p + "/group" + std::to_string(cv_loop + 1) + "/full_W.raw");
		std::cout << "read_full_W" << std::endl;
		for (int j = 0; j < weight1.cols(); j++){
			for (int i = 0; i < weight1.rows(); i++){
				weight1(i, j) = W_dummy[j * weight1.rows() + i];

			}
		}

		std::cout << "param get" << std::endl;

	}

	b1 = Eigen::VectorXf::Zero(numOutput);

}

//学習
void MLP::train(Eigen::MatrixXf& X, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& t, float learning_rate, Eigen::MatrixXf& back_delta, float& cost, float moment, float lamda)
{
	//誤差勾配の宣言
	Eigen::MatrixXf grad_W1 = Eigen::MatrixXf::Zero(numOutput, numInput);
	Eigen::VectorXf grad_b1 = Eigen::VectorXf::Zero(numOutput);

	//伝搬誤差
	Eigen::MatrixXf delta1 = Eigen::MatrixXf::Zero(numOutput, X.cols());

	Eigen::MatrixXf pre_delta = Eigen::MatrixXf::Zero(numInput, X.cols());

	Eigen::VectorXf x, z, deriv_z; //x : 入力層出力, z : 出力層出力,  derive_z : zの微分後
	Eigen::VectorXf t_vec; //コスト計算用

	for (int k = 0; k < X.cols(); k++)
	{
		x = X.col(k);
		z = weight1 * x + b1;

		if (act1 == "sigmoid")
		{

			sigmoid(z, z);
			sigmoid_deriv(z, deriv_z);

		}
		else if (act1 == "ReLU")
		{
			Eigen::VectorXf relu_z;
			relu_z = z;
			ReLU(z, z);

			ReLU_deriv(relu_z, deriv_z);

		}

		t_vec = t.col(k).cast<float>();
		if (error_function == 0)  //2乗誤差
		{
			cost += (((t_vec - z).array() * (t_vec - z).array()) / 2.0).sum();
			delta1.col(k) = deriv_z.array() * (z - t.col(k).cast<float>()).array();

		}
		else if (error_function == 1)  //クロスエントロピー
		{
			cost += -((t_vec.array() * (z.array() + FLT_MIN).log()).sum() + ((1.0 - t_vec.array()) * (1.0 - z.array() + FLT_MIN).log()).sum());
			delta1.col(k) = z - t.col(k).cast<float>();
		}

		pre_delta.col(k) = weight1.transpose() * delta1.col(k);

		grad_b1 += delta1.col(k);
		grad_W1 += delta1.col(k) * x.transpose();

	}

	back_delta = pre_delta;

	grad_W1 /= (float)X.cols();
	grad_b1 /= (float)X.cols();

	cost /= (float)X.cols();


	weight1 = weight1 - learning_rate * grad_W1 - lamda * weight1 + moment * weight1;

	b1 -= learning_rate * grad_b1;

}

void MLP::valid_test(Eigen::MatrixXf& X, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& t, float& cost)
{


	Eigen::VectorXf x, z, deriv_z; //x : 入力層出力, z : 出力層出力,  derive_z : zの微分後
	Eigen::VectorXf t_vec; //コスト計算用


	for (int k = 0; k < X.cols(); k++)
	{
		x = X.col(k);
		z = weight1 * x + b1;
		if (act1 == "sigmoid")
		{
			sigmoid(z, z);
		}
		else if (act1 == "ReLU")
		{
			ReLU(z, z);
		}

		t_vec = t.col(k).cast<float>();

		if (error_function == 0)  //2乗誤差
		{
			cost += (((t_vec - z).array() * (t_vec - z).array()) / 2.0).sum();

		}
		else if (error_function == 1)  //クロスエントロピー
		{
			cost += -((t_vec.array() * (z.array() + FLT_MIN).log()).sum() + ((1.0 - t_vec.array()) * (1.0 - z.array() + FLT_MIN).log()).sum());
		}

	}

	cost /= (float)X.cols();


}


void MLP::forward_propagation(Eigen::MatrixXf& X, Eigen::MatrixXf& Y)
{
	Eigen::MatrixXf x, z;
	Eigen::MatrixXf stock_output(weight1.rows(), X.cols());

	Eigen::VectorXf vec_output;

	for (int i = 0; i < (int)X.cols(); i++)
	{
		x = X.col(i);
		z = weight1 * x + b1;

		if (act1 == "sigmoid")
		{
			sigmoid(z, z);
		}
		else if (act1 == "ReLU")
		{
			ReLU(z, z);
		}

		stock_output.block(0, i, stock_output.rows(), 1) = z;

	}

	Y = stock_output;

}
