#pragma once

#include "Header.h"
#include "autoencoder.h"
#include "tktlib\raw_io.h"
#include <numeric>
#include <fstream>
#include "nalib\system\nariwindows.h"


#define PRINT_TIME std::cout << (double)(end - start) / CLOCKS_PER_SEC << std::endl


void pretrain(const Eigen::MatrixXf& train_X, const Eigen::MatrixXf& train_Y, const Eigen::MatrixXf& valid_X, const Eigen::MatrixXf& valid_Y, Eigen::MatrixXf& W, Eigen::MatrixXf& bout, Eigen::MatrixXf& bhid,
	float lr, int train_epoch, int batch_size, int n_hidden, const std::string& dir_o, int valid_num = 10)
{


	size_t n_visible = train_X.rows();
	size_t n_sample = train_X.cols();
	size_t n_batch = n_sample / batch_size;
	double batch_cost = 0.0;
	double valid_error = 0.0;
	time_t start, end;
	double epoch_cost;
	//std::vector<float> cost;
	//std::vector<float> valid_cost;
	Eigen::MatrixXf batch = batch.Zero(n_visible, batch_size);
	Eigen::MatrixXf batch_ans = batch_ans.Zero(n_visible, batch_size);

	std::vector<float> bb(n_visible);
	std::vector<float> cc(n_hidden);

	std::vector<double> cost_vec;
	std::vector<double> valid_cost_vec;
	std::vector<double> valid_cost_avg;

	//constract
	AE sdae(n_visible, n_hidden, n_sample);
	sdae.W = W;
	sdae.bout = bout;
	sdae.bhid = bhid;


	start = clock();
	//train
	std::cout << "pre_training start v(^_^)v" << std::endl;

	int count_valid = 0;

	for (size_t epoch = 0; epoch < train_epoch; epoch++){
		for (int batch_index = 0; batch_index < n_batch; batch_index++){
			batch = train_X.block(0, batch_size * batch_index, n_visible, batch_size);
			batch_ans = train_Y.block(0, batch_size * batch_index, n_visible, batch_size);
			sdae.param_update(batch, batch_ans, lr, batch_cost);
			epoch_cost += batch_cost;

			if (0 == batch_index % 100)
			{
				std::cout << "pretrain_epoch_" << epoch + 1 << "/" << train_epoch << "...batch" << batch_index << "/" << n_batch << std::endl;
				std::cout << "cost = " << batch_cost / batch_size << std::endl;
			}
			batch_cost = 0.0;
		} // ミニバッチループ

		
		epoch_cost /= n_sample;
		cost_vec.push_back(epoch_cost);
		printf("epoch : %d, cost = %f\n", (epoch + 1), epoch_cost);
		epoch_cost = 0.0;


		sdae.valid_test(valid_X, valid_Y, lr, valid_error);
		valid_cost_vec.push_back(valid_error);

		std::cout << "cross-validation error : " << valid_error << std::endl;

		// validation_num回おきに検定を行う

		if ((valid_cost_avg.size() > 0) && ((epoch + 1) % valid_num == 0)){
			std::cout << count_valid + 1 << "回目の検定を行います(*'ω'*)" << std::endl;
			///////////////////////////////////////////////////////////
			double total = 0;
			for (int val_n = 0; val_n < valid_num; val_n++)
			{
				total += valid_cost_vec[epoch - val_n];
			}
			valid_cost_avg.push_back(total / valid_num);


			std::cout << "前回 : " << valid_cost_avg[count_valid] << std::endl;
			std::cout << "今回 : " << valid_cost_avg[count_valid + 1] << std::endl;

			if (valid_cost_avg[count_valid] < valid_cost_avg[count_valid + 1])
			{
				break;	// early-stopping
			}
			count_valid++;
			std::cout << "処理続行します('◇')ゞ" << std::endl;
		}

		// 1回目のvalidation検定の平均を計算
		if (valid_cost_vec.size() == valid_num)
		{
			double total = 0;
			for (int val_n = 0; val_n < valid_num; val_n++)
			{
				total += valid_cost_vec[val_n];
			}
			valid_cost_avg.push_back(total / valid_num);
		}


		//if (epoch % 10 == 0)
		//{
		//	sdae.valid_test(valid_X, valid_Y, lr, valid_error);
		//	valid_cost.push_back((float)valid_error / valid_X.cols());
		//	std::cout << "cross-validation error : " << valid_error / valid_X.cols() << std::endl;

		//	valid_error = 0.0;

		//	if ((epoch > 0) & (valid_cost[epoch / 10 - 1] < valid_cost[epoch / 10])){
		//		break;
		//	}
		//}
	}
	end = clock();

	std::cout << "time = ";
	PRINT_TIME;
	printf("\n");

	std::ofstream time(dir_o + "/time.txt");
	time << "time : " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
	vec_to_txt(cost_vec, dir_o + "/cost.txt");
	vec_to_txt(valid_cost_vec, dir_o + "/valid_cost.txt");

	write_raw_and_txt(sdae.W, dir_o + "/W");
	write_raw_and_txt(sdae.bhid, dir_o + "/bhid");
	write_raw_and_txt(sdae.bout, dir_o + "/bout");

	W = sdae.W;
	bout = sdae.bout;
	bhid = sdae.bhid;

}

// 学習サンプルだけでパラメータ決定するならこっち
//void pretrain(const Eigen::MatrixXf& train_X, Eigen::MatrixXf& W, Eigen::MatrixXf& b, Eigen::MatrixXf& c,
//	double lr, int train_epoch, int batch_size, int regu_switch, double lambda, int k, int n_hidden, const std::string& dir_o)
//{
//
//
//	size_t n_visible = train_X.rows();
//	size_t n_sample = train_X.cols();
//	size_t n_batch = n_sample / batch_size;
//	double error = 0.0;
//	double valid_error = 0.0;
//	time_t start, end;
//	Eigen::VectorXf error_sum(n_batch);
//	std::vector<double> cost;
//	std::vector<double> valid_cost;
//	Eigen::MatrixXf batch = batch.Zero(n_visible, batch_size);
//
//	std::vector<double> bb(n_visible);
//	std::vector<double> cc(n_hidden);
//
//	//constract
//	RBM rbm(n_visible, n_hidden, n_sample);
//	rbm.W = W;
//	rbm.b = b;
//	rbm.c = c;
//
//
//	start = clock();
//	//train
//	std::cout << "pre_training start v(^_^)v" << std::endl;
//	for (size_t epoch = 0; epoch < train_epoch; epoch++){
//		/*#ifdef _OPENMP
//		#pragma omp parallel for
//		#endif*/
//		for (int batch_index = 0; batch_index < n_batch; batch_index++){
//			batch = train_X.block(0, batch_size * batch_index, n_visible, batch_size);
//			rbm.param_update(batch, lr, k, error, regu_switch, lambda);
//			error_sum(batch_index) = error;
//
//			if (0 == batch_index % 100){
//				std::cout << "pretrain_epoch_" << epoch + 1 << "/" << train_epoch << "...batch" << batch_index << "/" << n_batch << std::endl;
//				std::cout << "cost = " << error / batch_size << std::endl;
//
//			}
//			error = 0.0;
//		}
//		cost.push_back(error_sum.sum() / n_sample);
//
//		std::cout << cost[epoch] << std::endl;
//
//		error_sum = Eigen::VectorXf::Zero(error_sum.rows());
//
//		if ((epoch > 0) & (epoch % 10 == 0)){
//			double COST = cost[epoch] + cost[epoch - 1] + cost[epoch - 2] + cost[epoch - 3] + cost[epoch - 4] + cost[epoch - 4] + cost[epoch - 5] + cost[epoch - 6] + cost[epoch - 7] + cost[epoch - 8] + cost[epoch - 9];
//			COST = COST / 10.;
//			if (cost[epoch] >= COST){
//				break;
//			}
//		}
//	}
//	end = clock();
//
//	std::cout << (end - start) / 1000 << "[s]" << std::endl;
//
//
//
//	save_vector_to_txt(cost, dir_o + "/cost.txt");
//	vec_to_txt(valid_cost, dir_o + "/valid_cost.txt");
//
//	write_raw_and_txt(rbm.W, dir_o + "/W");
//	write_raw_and_txt(rbm.b, dir_o + "/b");
//	write_raw_and_txt(rbm.c, dir_o + "/c");
//
//	W = rbm.W;
//	b = rbm.b;
//	c = rbm.c;
//
//}

class hidden_layer
{
	std::vector<AE> h;
public:

	void push_back(AE& sdae)
	{
		h.push_back(sdae);
	}

	void return_param(Eigen::MatrixXf& Weight, Eigen::MatrixXf& bhidden, Eigen::MatrixXf& bout_, int i)
	{
		Weight = h[i].W;
		bhidden = h[i].bhid;
		bout_ = h[i].bout;
	}

	void pretraining(int layer, const Eigen::MatrixXf& train_X, const Eigen::MatrixXf & train_Y, const Eigen::MatrixXf & valid_X, const Eigen::MatrixXf & valid_Y, double lr, int train_epoch, int batch_size, int n_hidden, const std::string& dir_o)
	{
		pretrain(train_X, train_Y, valid_X, valid_Y, h[layer].W, h[layer].bout, h[layer].bhid, (float)lr, train_epoch, batch_size, n_hidden, dir_o);
	}


	void hidden_train_data(int layer, Eigen::MatrixXf& trainX)
	{
		Eigen::MatrixXf I = I.Ones(1, trainX.cols());
		Eigen::MatrixXf BHID = h[layer].bhid * I;
		Eigen::MatrixXf X = h[layer].W * trainX;
		trainX = 1 / (1 + exp((-BHID - X).array()));
	}

	void feedforward(const Eigen::MatrixXf& visible, Eigen::MatrixXf& sigmoid_activation, size_t n_layer)
	{
		Eigen::MatrixXf layer_output = visible;

		for (size_t layer = 0; layer < n_layer - 1; layer++)
		{
			layer_output = 1 / (1 + exp((-h[layer].W * layer_output).array()));
		}
		sigmoid_activation = layer_output;
	}

	void feedforward(const Eigen::MatrixXf& visible, size_t n_layer)
	{
		Eigen::MatrixXf dummy = visible;
		for (auto it = h.begin(); it != h.end(); it++)
		{
			it->output = (1 / (1 + exp((-it->W * dummy).array())));
			dummy = it->output;
		}
	}

	void finetune_forward(const Eigen::MatrixXf& visible, size_t n_layer)
	{
		Eigen::MatrixXf dummy = visible;
		for (auto it = h.begin(); it != h.end(); it++)
		{
			Eigen::MatrixXf I = I.Ones(1, dummy.cols());
			Eigen::MatrixXf BHID = it->bhid * I;
			Eigen::MatrixXf X = it->W * dummy + BHID;

			it->output = (1 / (1 + exp((-X).array())));
			dummy = it->output;
		}
	}
};

class SDAE
{
public:
	Eigen::MatrixXf W;			//weight
	Eigen::MatrixXf bhid;		//hidden bias
	Eigen::MatrixXf output;		// encode output
	Eigen::MatrixXf d;			// encode delta

	int n_visible;
	int n_hidden;
	int batch_size;

	SDAE(const Eigen::MatrixXf&, const Eigen::MatrixXf&);

};

SDAE::SDAE(const Eigen::MatrixXf& Weight, const Eigen::MatrixXf& bhidden)
{
	W = Weight;
	bhid = bhidden;
}

void forward_prop(const Eigen::MatrixXf& trainX, const Eigen::MatrixXf& Weight, const Eigen::MatrixXf& bhidden, Eigen::MatrixXf& Output)
{
	Eigen::MatrixXf I = I.Ones(1, trainX.cols());
	Eigen::MatrixXf BHID = bhidden * I;
	Eigen::MatrixXf X = Weight * trainX + BHID;
	Output = 1 / (1 + exp((-X).array()));
}


void fine_tuning(std::vector<SDAE>& sdae, const Eigen::MatrixXf& trainX, const Eigen::MatrixXf& trainY, const Eigen::MatrixXf& validX, const Eigen::MatrixXf& validY, float eps, int train_epoch, int batch_size, std::string dir_o, int valid_num = 10)
{

	size_t n_sample = trainX.cols();
	size_t n_batch = n_sample / batch_size;
	time_t start, end;
	double epoch_cost = 0.0;
	double batch_cost = 0.0;
	double valid_cost = 0.0;

	Eigen::MatrixXf batchX;
	Eigen::MatrixXf batchY;
	Eigen::MatrixXf Output;
	Eigen::MatrixXf diff;
	std::vector<Eigen::MatrixXf> valid_out(sdae.size());

	std::vector<double> cost_vec;
	std::vector<double> valid_cost_vec;
	std::vector<double> valid_cost_avg;

	start = clock();
	int count_valid = 0;

	for (size_t epoch = 0; epoch < train_epoch; epoch++){
		for (size_t batch_index = 0; batch_index < n_batch; batch_index++){

			batchX = trainX.block(0, batch_size * batch_index, trainX.rows(), batch_size);
			batchY = trainY.block(0, batch_size * batch_index, trainY.rows(), batch_size);

			for (auto it = sdae.begin(); it != sdae.end(); it++)
			{
				if (it == sdae.begin())
				{
					forward_prop(batchX, it->W, it->bhid, it->output);
				}
				else
				{
					forward_prop((it - 1)->output, it->W, it->bhid, it->output);
				}

			}

			diff = (sdae.end() - 1)->output - batchY;

			// クロスエントロピー
			//error = -(batchY.array() * (sdae.back().output.array() + FLT_MIN).log() + (1. - batchY.array()) * (1. - sdae.back().output.array() + FLT_MIN).log()).sum();

			// 二乗誤差
			batch_cost = ((diff.array() * diff.array()).matrix()).sum();

			if (0 == batch_index % 100){
				std::cout << "finetuning_epoch_" << epoch + 1 << "/" << train_epoch << "...batch" << batch_index << "/" << n_batch << std::endl;
				std::cout << "cost = " << batch_cost / batch_size << std::endl;
			}

			epoch_cost += batch_cost;
			batch_cost = 0.0;
			for (auto it = sdae.rbegin(); it != sdae.rend(); it++)
			{
				// 入力層のところ
				if (it == sdae.rend() - 1)
				{

					it->d = diff.array() * (it->output.array() * (1 - it->output.array()));

					Eigen::MatrixXf dW = it->d * batchX.transpose() / (float)batchX.cols();
					Eigen::MatrixXf db = it->d.rowwise().sum() / (float)batchX.cols();

					it->W = it->W.array() - eps * dW.array();
					it->bhid = it->bhid.array() - eps * db.array();

				}

				// その他
				else
				{
					it->d = diff.array() * it->output.array() * (1 - it->output.array());

					Eigen::MatrixXf dW = it->d * (it + 1)->output.transpose() / (float)batchX.cols();
					Eigen::MatrixXf db = it->d.rowwise().sum() / (float)batchX.cols();

					it->W = it->W.array() - eps * dW.array();
					it->bhid = it->bhid.array() - eps * db.array();

					diff = it->W.transpose() * it->d;
				}

			}
		} // ミニバッチループ

		epoch_cost /= n_sample;
		cost_vec.push_back(epoch_cost);
		printf("epoch : %d, cost = %f\n", (epoch + 1), epoch_cost);
		epoch_cost = 0.0;


		// validationエラーを計算
		for (int i = 0; i < sdae.size(); i++)
		{
			if (i == 0)
			{
				forward_prop(validX, sdae[i].W, sdae[i].bhid, valid_out[i]);
			}
			else
			{
				forward_prop(valid_out[i - 1], sdae[i].W, sdae[i].bhid, valid_out[i]);
			}
		}
		// 二乗誤差
		valid_cost = ((validY.array() - valid_out.back().array()).array() * (validY.array() - valid_out.back().array()).array()).sum() / validY.cols();
		valid_cost_vec.push_back(valid_cost);
		std::cout << "cross-validation error : " << valid_cost << std::endl;
		valid_cost = 0.0;
		
		//validation_num回おきに検定を行う
		if ((valid_cost_avg.size() > 0) && ((epoch + 1) % valid_num == 0)){
			std::cout << count_valid + 1 << "回目の検定を行います(*'ω'*)" << std::endl;
			///////////////////////////////////////////////////////////
			double total = 0;
			for (int val_n = 0; val_n < valid_num; val_n++)
			{
				total += valid_cost_vec[epoch - val_n];
			}
			valid_cost_avg.push_back(total / valid_num);


			std::cout << "前回 : " << valid_cost_avg[count_valid] << std::endl;
			std::cout << "今回 : " << valid_cost_avg[count_valid + 1] << std::endl;

			if (valid_cost_avg[count_valid] < valid_cost_avg[count_valid + 1])
			{
				break;	// early-stopping
			}
			count_valid++;
			std::cout << "処理続行します('◇')ゞ" << std::endl;
		}

		// 1回目のvalidation検定の平均を計算
		if (valid_cost_vec.size() == valid_num)
		{
			double total = 0;
			for (int val_n = 0; val_n < valid_num; val_n++)
			{
				total += valid_cost_vec[val_n];
			}
			valid_cost_avg.push_back(total / valid_num);
		}



		//if (epoch % 10 == 0){
		//	for (int i = 0; i < sdae.size(); i++)
		//	{
		//		if (i == 0)
		//		{
		//			forward_prop(validX, sdae[i].W, sdae[i].bhid, valid_out[i]);
		//		}
		//		else
		//		{
		//			forward_prop(valid_out[i-1], sdae[i].W, sdae[i].bhid, valid_out[i]);
		//		}
		//	}

		//	// クロスエントロピー
		//	//valid_cost.push_back((-(validY.array() * (valid_out.back().array() + FLT_MIN).log() + (1. - validY.array()) * (1. - valid_out.back().array() + FLT_MIN).log())).sum() / validY.cols());

		//	// 二乗誤差
		//	valid_cost.push_back(((validY.array() - valid_out.back().array()).array() * (validY.array() - valid_out.back().array()).array()).sum() / validY.cols());
		//	std::cout << "cross-validation error : " << valid_cost[epoch / 10] << std::endl;

		//	if ((epoch > 0) & (valid_cost[epoch / 10 - 1] < valid_cost[epoch / 10])){
		//		break;
		//	}
		//}
	}

	end = clock();
	std::cout << "time = ";
	PRINT_TIME;
	printf("\n");


	std::ofstream time(dir_o + "/time.txt");
	time << "time : " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
	vec_to_txt(cost_vec, dir_o + "/cost.txt");
	vec_to_txt(valid_cost_vec, dir_o + "/valid_cost.txt");

	int cnt = 1;
	for (auto it = sdae.begin(); it != sdae.end(); it++)
	{
		write_raw_and_txt(it->W, dir_o + "/W" + std::to_string(cnt));
		write_raw_and_txt(it->bhid, dir_o + "/b" + std::to_string(cnt));
		cnt++;
	}

}

void test(std::vector<SDAE>& sdae, Eigen::MatrixXf testX, std::string dir_o, std::string name)
{
	Eigen::MatrixXf expect;
	for (int i = 0; i < sdae.size(); i++)
	{
		std::string dir_o_ = dir_o;
		if (i == 0)
		{
			std::cout << "(-_-)" << std::endl;
			expect = testX;
		}
		forward_prop(expect, sdae[i].W, sdae[i].bhid, expect);

		if (i == (sdae.size() - 1))
		{
			std::cout << "(^_^)" << std::endl;
			if (!nari::system::directry_is_exist(dir_o_ + "/hidden" + std::to_string(i + 1))) nari::system::make_directry(dir_o_ + "/expect");
			write_raw_and_txt(expect, dir_o + "/expect/" + name);
		}
		else
		{
			// 中間層の出力が必要な場合はコメントアウトを外す
			//if (!nari::system::directry_is_exist(dir_o_ + "/hidden" + std::to_string(i + 1))) nari::system::make_directry(dir_o_ + "/hidden" + std::to_string(i + 1));
			//write_raw_and_txt(expect, dir_o_ + "/hidden" + std::to_string(i + 1) + "/" + name);
		}
	}



}


