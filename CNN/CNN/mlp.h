#pragma once
#include "Header.h"
#include "Utils.h"

class MLP
{
public:
	int numInput, numHidden, numOutput; //入力層，中間層，出力層のユニット数
	int error_function;      // 誤差関数
	int dropout;             // dropout行うか
	Eigen::MatrixXf weight1, weight2; //入力層-中間層，中間層-出力層の重み行列
	Eigen::VectorXf b1, b2; //入力層-中間層，中間層-出力層のバイアスベクトル

	bool full_connect;

	std::string act1, act2;

	MLP(){};
	~MLP(){};

	MLP(int, int, std::string, int, int); //中間層挟まない
	MLP(int, int, int, std::string, std::string, int, int); //中間層挟む

	MLP(int, int, std::string, int, std::string, std::string, int); //中間層挟まない(学習後重み使用)
	MLP(int, int, int, std::string, std::string, int, std::vector<std::string>, std::vector<std::string>, int); //中間層挟む(学習後重み使用)

	void train(Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, float, Eigen::MatrixXf&, float&, float, float);
	void train(Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, float, Eigen::MatrixXf&, float&, float, float, std::vector<std::vector<float>>&); //出力ヒストグラム用
	void train(Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, float, float&, float, float, std::vector<std::vector<float>>&, int batch_size); //特徴量生成後のMLP用

	void valid_test(Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, float&);
	void valid_test(Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, Eigen::MatrixXf&, float&); //アトラス用

	void forward_propagation(Eigen::MatrixXf&, Eigen::MatrixXf&);
	void forward_propagation(Eigen::MatrixXf&, Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, std::vector<std::vector<float>>&); //出力ヒストグラム用
	void forward_propagation(Eigen::MatrixXf&, Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, Eigen::MatrixXf&, std::vector<std::vector<float>>&); //出力ヒストグラムとアトラス用

};


/**** full-connectに層を挟まない場合のコンストラクタ ****/
MLP::MLP(int _numInput, int _numOutput, std::string _act1, int _error_function, int _dropout)
{
	numInput = _numInput;
	numOutput = _numOutput;

	act1 = _act1;

	error_function = _error_function;

	dropout = _dropout;

	full_connect = false;

	weight1 = Eigen::MatrixXf::Zero(numOutput, numInput);
	dist = std::uniform_real_distribution<float>(-(float)sqrt(6.0 / (float)(numOutput + numInput)), (float)sqrt(6.0 / (float)(numOutput + numInput)));
	for (int j = 0; j < (int)weight1.rows(); j++)
	{
		for (int i = 0; i < (int)weight1.cols(); i++)
		{
			weight1(j, i) = dist(engine);

		}
	}

	b1 = Eigen::VectorXf::Zero(numOutput);

}


/**** full-connectに層を挟む場合のコンストラクタ ****/
MLP::MLP(int _numInput, int _numHidden, int _numOutput, std::string _act1, std::string _act2, int _error_function, int _dropout = 1)
{

	numInput = _numInput;
	numHidden = _numHidden;
	numOutput = _numOutput;

	act1 = _act1;
	act2 = _act2;

	error_function = _error_function;

	dropout = _dropout;

	full_connect = true;

	weight1 = Eigen::MatrixXf::Zero(numHidden, numInput);
	weight2 = Eigen::MatrixXf::Zero(numOutput, numHidden);
	dist = std::uniform_real_distribution<float>(-(float)sqrt(6.0 / (float)(numHidden + numInput)), (float)sqrt(6.0 / (float)(numHidden + numInput)));
	for (int j = 0; j < (int)weight1.rows(); j++)
	{
		for (int i = 0; i < (int)weight1.cols(); i++)
		{
			weight1(j, i) = dist(engine);

		}
	}
	dist = std::uniform_real_distribution<float>(-(float)sqrt(6.0 / (float)(numHidden + numOutput)), (float)sqrt(6.0 / (float)(numHidden + numOutput)));
	for (int j = 0; j < (int)weight2.rows(); j++)
	{
		for (int i = 0; i < (int)weight2.cols(); i++)
		{
			weight2(j, i) = dist(engine);
		}
	}

	b1 = Eigen::VectorXf::Zero(numHidden);
	b2 = Eigen::VectorXf::Zero(numOutput);
}

/**** full-connectに層を挟まない場合のコンストラクタ(学習後重み使用) ****/
MLP::MLP(int _numInput, int _numOutput, std::string _act1, int _error_function, std::string w_adress, std::string b_adress, int _dropout)
{
	numInput = _numInput;
	numOutput = _numOutput;

	act1 = _act1;

	error_function = _error_function;

	dropout = _dropout;

	full_connect = false;

	weight1 = Eigen::MatrixXf::Zero(numOutput, numInput);

	std::vector<float> w, b;
	read_vector(w, w_adress);
	read_vector(b, b_adress);

	for (int col = 0; col < (int)weight1.cols(); col++)
	{
		for (int row = 0; row < (int)weight1.rows(); row++)
		{
			weight1(row, col) = w[col * weight1.rows() + row];
		}
	}

	b1 = Eigen::VectorXf::Zero(numOutput);
	for (int i = 0; i < (int)b1.size(); i++)
	{
		b1(i) = b[i];
	}

}

/**** full-connectに層を挟む場合のコンストラクタ(学習後重み使用) ****/
MLP::MLP(int _numInput, int _numHidden, int _numOutput, std::string _act1, std::string _act2, int _error_function, std::vector<std::string> w_adress, std::vector<std::string> b_adress, int _dropout)
{
	numInput = _numInput;
	numHidden = _numHidden;
	numOutput = _numOutput;

	act1 = _act1;
	act2 = _act2;

	error_function = _error_function;

	dropout = _dropout;

	full_connect = true;

	weight1 = Eigen::MatrixXf::Zero(numHidden, numInput);
	weight2 = Eigen::MatrixXf::Zero(numOutput, numHidden);

	std::vector<float> w1, w2, b1_vec, b2_vec;
	read_vector(w1, w_adress[0]);
	read_vector(w2, w_adress[1]);
	read_vector(b1_vec, b_adress[0]);
	read_vector(b2_vec, b_adress[1]);

	for (int col = 0; col < (int)weight1.cols(); col++)
	{
		for (int row = 0; row < (int)weight1.rows(); row++)
		{
			weight1(row, col) = w1[col * weight1.rows() + row];
		}
	}

	for (int col = 0; col < (int)weight2.cols(); col++)
	{
		for (int row = 0; row < (int)weight2.rows(); row++)
		{
			weight2(row, col) = w2[col * weight2.rows() + row];
		}
	}

	b1 = Eigen::VectorXf::Zero(numHidden);
	b2 = Eigen::VectorXf::Zero(numOutput);

	for (int i = 0; i < (int)b1.size(); i++)
	{
		b1(i) = b1_vec[i];
	}

	for (int i = 0; i < (int)b2.size(); i++)
	{
		b1(i) = b2_vec[i];
	}
}

/**** 学習 ****/
void MLP::train(Eigen::MatrixXf& X, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& t, float learning_rate, Eigen::MatrixXf& back_delta, float& cost, float moment, float lamda)
{

	if (full_connect) //中間層を挟む場合
	{
		Eigen::VectorXi mask(numHidden);
		if (dropout)
		{
			for (int i = 0; i < mask.size(); i++)
			{
				mask(i) = rand() % 2;
			}
		}
		else
		{
			mask = mask.Ones(numHidden);
		}

		//誤差勾配の宣言
		Eigen::MatrixXf grad_W1 = Eigen::MatrixXf::Zero(numHidden, numInput);
		Eigen::MatrixXf grad_W2 = Eigen::MatrixXf::Zero(numOutput, numHidden);
		Eigen::VectorXf grad_b1 = Eigen::VectorXf::Zero(numHidden);
		Eigen::VectorXf grad_b2 = Eigen::VectorXf::Zero(numOutput);

		//伝搬誤差
		Eigen::MatrixXf delta1 = Eigen::MatrixXf::Zero(numHidden, X.cols());
		Eigen::MatrixXf delta2 = Eigen::MatrixXf::Zero(numOutput, X.cols());

		Eigen::MatrixXf pre_delta = Eigen::MatrixXf::Zero(numInput, X.cols());

		Eigen::VectorXf x, z, y, deriv_z, deriv_y; //x : 入力層出力, z : 中間層出力, y : 出力層出力, derive_z : zの微分後, deriv_x : yの微分後
		Eigen::VectorXf t_vec; //コスト計算用

		for (int k = 0; k < X.cols(); k++)
		{
			x = X.col(k);
			z = weight1 * x + b1;

			if (act1 == "sigmoid")
			{
				sigmoid(z, z);
				z.array() *= mask.cast<float>().array();  ///dropout
				sigmoid_deriv(z, deriv_z);

			}
			else if (act1 == "ReLU")
			{
				Eigen::VectorXf relu_z;
				relu_z = z;
				ReLU(z, z);
				z.array() *= mask.cast<float>().array();  ///dropout
				ReLU_deriv(relu_z, deriv_z);

			}

			y = weight2 * z + b2;

			if (act2 == "sigmoid")
			{
				sigmoid(y, y);
				sigmoid_deriv(y, deriv_y);
			}
			else if (act2 == "ReLU")
			{
				Eigen::VectorXf relu_y;
				relu_y = y;
				ReLU(y, y);
				ReLU_deriv(relu_y, deriv_y);

			}
			else if (act2 == "softmax")
			{

				softmax(y, y);
				sigmoid_deriv(y, deriv_y);
			}


			t_vec = t.col(k).cast<float>();

			if (error_function == 0)  //2乗誤差
			{
				cost += (((t_vec - y).array() * (t_vec - y).array()) / 2.0).sum();
				delta2.col(k) = deriv_y.array() * (y - t.col(k).cast<float>()).array();

			}
			else if (error_function == 1)  //クロスエントロピー
			{
				cost += -((t_vec.array() * (y.array() + FLT_MIN).log()).sum() + ((1.0 - t_vec.array()) * (1.0 - y.array() + FLT_MIN).log()).sum());
				delta2.col(k) = y - t.col(k).cast<float>();
			}

			delta1.col(k) = deriv_z.array() * (weight2.transpose() * delta2.col(k)).array();

			delta1.col(k).array() *= mask.cast<float>().array();  ///dropout

			pre_delta.col(k) = weight1.transpose() * delta1.col(k);

			grad_b1 += delta1.col(k);
			grad_b2 += delta2.col(k);
			grad_W1 += delta1.col(k) * x.transpose();
			grad_W2 += delta2.col(k) * z.transpose();

		}

		back_delta = pre_delta;

		grad_W1 /= (float)X.cols();
		grad_W2 /= (float)X.cols();
		grad_b1 /= (float)X.cols();
		grad_b2 /= (float)X.cols();

		cost /= (float)X.cols();


		weight1 = weight1 - learning_rate * grad_W1 - lamda * weight1 + moment * weight1;
		weight2 = weight2 - learning_rate * grad_W2 - lamda * weight2 + moment * weight2;
		b1 -= learning_rate * grad_b1;
		b2 -= learning_rate * grad_b2;

	}
	else{ //中間層を挟まない場合

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

}

/**** 検証 ****/
void MLP::valid_test(Eigen::MatrixXf& X, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& t, float& cost)
{

	if (full_connect) //中間層を挟む場合
	{

		Eigen::VectorXf x, z, y; //x : 入力層出力, z : 中間層出力, y : 出力層出力
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

			if (dropout)
			{
				y = 0.5 * weight2 * z + b2;
			}
			else
			{
				y = weight2 * z + b2;
			}

			if (act2 == "sigmoid")
			{
				sigmoid(y, y);

			}
			else if (act2 == "ReLU")
			{
				ReLU(y, y);

			}
			else if (act2 == "softmax")
			{
				softmax(y, y);
			}

			t_vec = t.col(k).cast<float>();

			if (error_function == 0)  //2乗誤差
			{
				cost += (((t_vec - y).array() * (t_vec - y).array()) / 2.0).sum();

			}
			else if (error_function == 1)  //クロスエントロピー
			{
				cost += -((t_vec.array() * (y.array() + FLT_MIN).log()).sum() + ((1.0 - t_vec.array()) * (1.0 - y.array() + FLT_MIN).log()).sum());
			}

		}

		cost /= (float)X.cols();

	}
	else{ //中間層を挟まない場合

		Eigen::VectorXf x, z; //x : 入力層出力, z : 出力層出力
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
}

/**** テスト ****/
void MLP::forward_propagation(Eigen::MatrixXf& X, Eigen::MatrixXf& Y)
{
	if (full_connect) //中間層を挟む場合
	{

		Eigen::VectorXf x, z, y; //x : 入力層出力, z : 中間層出力, y : 出力層出力
		Eigen::MatrixXf stock_output(weight2.rows(), X.cols());

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

			if (dropout)
			{
				y = 0.5 * weight2 * z + b2;
			}
			else
			{
				y = weight2 * z + b2;
			}

			if (act2 == "sigmoid")
			{
				sigmoid(y, y);

			}
			else if (act2 == "ReLU")
			{
				ReLU(y, y);

			}
			else if (act2 == "softmax")
			{
				softmax(y, y);
			}


			stock_output.block(0, k, stock_output.rows(), 1) = y;

		}

		Y = stock_output;

	}
	else{ //中間層を挟まない場合

		Eigen::VectorXf x, z; //x : 入力層出力, z : 出力層出力
		Eigen::MatrixXf stock_output(weight1.rows(), X.cols());



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


			stock_output.block(0, k, stock_output.rows(), 1) = z;

		}

		Y = stock_output;

	}

}
