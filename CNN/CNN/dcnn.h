#pragma once
#include"Header.h"
#include"cnn.h"
#include "mlp.h"
#include "dcnn_info.h"

class DCNN
{
private:
public:
	std::vector<CNN> layer_h;
	std::vector<MLP> layer_f;

	int n_layer, out, in, dim, in_dim, out_dim, w_dim, in_map, error_function, validation_num;

	std::vector<int> map, w_size, pool_size;
	std::vector<char> c_p;
	std::vector<std::string> activation;

	float cost, batch_cost;

	std::vector<Eigen::Triplet<float>> t;

	DCNN(dcnn_info, std::vector<char>, int, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<std::string>, int, int);

	void train(std::string, Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, Eigen::MatrixXf&, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&, int, int, float, float, float, int); //(入力，正解，epoch, バッチサイズ, 学習率)

	void predict(Eigen::MatrixXf&, Eigen::MatrixXf&); //入力，出力
};

DCNN::DCNN(dcnn_info input_info, std::vector<char> prosess, int _in_map, std::vector<int> _map, std::vector<int> _w_size, std::vector<int> _pool_size, std::vector<std::string> _activation, int cv_loop, int dropout)
{
	n_layer = input_info.n_layer;
	in = input_info.in_size;
	out = input_info.out_size;
	dim = input_info.dim;
	error_function = input_info.error_funcion;
	validation_num = input_info.validation_num;

	map = _map;
	w_size = _w_size;
	pool_size = _pool_size;
	c_p = prosess;
	activation = _activation;
	in_map = _in_map;

	if (dim == 2){
		in_dim = in *in;
	}

	if (dim == 3){
		in_dim = in *in * in;
	}


	int w_size_count = 0;
	int pool_size_count = 0;
	for (int i = 0; i < n_layer - 1; i++){

		if (i == 0 && c_p[i] == 'c'){
			CNN cnn(input_info.dir_p, in, in_map, map[i], w_size[w_size_count], c_p[i], in - w_size[w_size_count] + 1, dim, activation[i], input_info.ini_switch, cv_loop, i);
			layer_h.push_back(cnn);
			w_size_count++;
		}
		// ※2層目以降は違うコンストラクタを呼ぶ
		// ※3番目の引数は1層目のコンストラクタと区別をつけるための意味はない引数
		else if (i != 0 && c_p[i] == 'c'){
			CNN cnn(input_info.dir_p, layer_h[i - 1].out_size, layer_h[i - 1].W.rows(), map[i - 2], map[i], w_size[w_size_count], c_p[i], layer_h[i - 1].out_size - w_size[w_size_count] + 1, dim, activation[i], input_info.ini_switch, cv_loop, i);
			layer_h.push_back(cnn);
			w_size_count++;
		}
		else if (i != 0 && c_p[i] == 'p'){
			if (layer_h[i - 1].out_size % pool_size[pool_size_count] == 0)
			{
				CNN cnn(input_info.dir_p, layer_h[i - 1].out_size, layer_h[i - 1].W.rows(), map[i - 1], layer_h[i - 1].n_map, pool_size[pool_size_count], c_p[i], layer_h[i - 1].out_size / pool_size[pool_size_count], dim, activation[i], input_info.ini_switch, cv_loop, i);
				layer_h.push_back(cnn);
			}
			else
			{
				CNN cnn(input_info.dir_p, layer_h[i - 1].out_size, layer_h[i - 1].W.rows(), map[i - 1], layer_h[i - 1].n_map, pool_size[pool_size_count], c_p[i], (layer_h[i - 1].out_size / pool_size[pool_size_count]) + 1, dim, activation[i], input_info.ini_switch, cv_loop, i);
				layer_h.push_back(cnn);
			}
			pool_size_count++;
		}
		// ※Full connect層も違うコンストラクタを呼ぶ
		else if (i != 0 && c_p[i] == 'f'){
			// *** ここのコンストラクタの呼び方によってフルコネクト層の中間層の有無を決める *** //
			MLP mlp((int)layer_h[i - 1].W.rows(), (int)layer_h[i - 1].W.rows(), out, activation[i], activation[i], error_function, dropout);
			layer_f.push_back(mlp);
		}
	}

}


// ネットワークの訓練 長くなったので小分けしたい
void DCNN::train(std::string dir_o, Eigen::MatrixXf& trainX, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& trainY,
	Eigen::MatrixXf& validX, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& validY, int epochs = 500, int batch_size = 50, float learning_rate = 0.1, float moment = 0.0, float lamda = 0.0, int dropout = 1)
{

	int batch_num;  //1epochごとに重みを更新する回数

	batch_num = (int)trainX.cols() / batch_size;

	Eigen::MatrixXf batch;
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> batch_label;
	Eigen::MatrixXf output;
	Eigen::VectorXf vec_output;

	Eigen::MatrixXf back_delta;

	std::vector<float> cost_vec;
	std::vector<float> valid_cost_vec;
	std::vector<float> valid_cost_avg;

	int count_valid = 0;

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		cost = 0.0;

		// *** ミニバッチループ *** //
		for (int batch_index = 0; batch_index < batch_num; batch_index++)
		{

			if ((batch_index + 1) % 100 == 0)
			{
				printf("epoch : %d / %d , batch : %d / %d\n", epoch + 1, epochs, batch_index + 1, batch_num);
				printf("cost = %f\n", batch_cost);
			}

			// *** バッチの取り出し *** //
			batch = trainX.block(0, batch_size * batch_index, trainX.rows(), batch_size);
			batch_label = trainY.block(0, batch_size * batch_index, trainY.rows(), batch_size);

			std::vector<Eigen::MatrixXf> output_stock;
			output_stock.push_back(batch);

			/**** foward propagation ****/
			for (int layer = 0; layer < n_layer - 1; layer++)
			{
				if (layer == 0){
					layer_h[0].convn(batch, output, layer_h[0].act);
					output_stock.push_back(output);

				}
				else if (layer != 0 && c_p[layer] == 'c'){
					layer_h[layer].convn(output, output, layer_h[layer].act);
					output_stock.push_back(output);

				}
				else if (layer != 0 && c_p[layer] == 'p')
				{
					layer_h[layer].pooling(output, output, layer_h[layer].act);
					output_stock.push_back(output);
				}
				else if (layer == n_layer - 2)
				{
					batch_cost = 0.0;
					layer_f[0].train(output, batch_label, learning_rate, back_delta, batch_cost, moment, lamda);

					cost += batch_cost;

				}

			}
			/**** foward propagation ここまで ****/

			Eigen::MatrixXf grad_W;
			Eigen::MatrixXf deriv_output;
			Eigen::MatrixXf delta;
			Eigen::MatrixXf W_grad_stock;

			int n_map, pre_map, w_size, in_size, out_size, w_num, out_dim, w_dim, in_dim;


			/**** back propagation ****/
			for (int layer = n_layer - 3; layer >= 0; layer--)
			{
				if (layer == n_layer - 3) //full-connectにつながるプーリング層
				{
					back_delta = layer_h[layer].W.transpose() * back_delta;
				}
				else if (layer != 0 && c_p[layer] == 'c') //入力層につながる以外の畳み込み層
				{
					grad_W.resize(layer_h[layer].W.rows(), layer_h[layer].W.cols());
					grad_W = Eigen::MatrixXf::Zero(grad_W.rows(), grad_W.cols());

					if (layer_h[layer].act == "sigmoid")
					{
						sigmoid_deriv(output_stock[layer + 1], deriv_output);
					}
					else if (layer_h[layer].act == "ReLU")
					{
						Eigen::MatrixXf output_invert;
						ReLU_invert(output_stock[layer + 1], output_invert);
						ReLU_deriv(output_invert, deriv_output);
					}

					back_delta = deriv_output.array() * back_delta.array();

					grad_W = back_delta * output_stock[layer].transpose();

					grad_W /= (float)back_delta.cols();

					pre_map = layer_h[layer - 2].n_map;
					n_map = layer_h[layer].n_map;
					in_size = layer_h[layer].in_size;
					out_size = layer_h[layer].out_size;
					w_size = layer_h[layer].w_size;
					w_num = layer_h[layer].w_num;
					out_dim = layer_h[layer].out_dim;
					w_dim = layer_h[layer].w_dim;
					in_dim = layer_h[layer].in_dim;

					W_grad_stock.resize(n_map * pre_map, w_dim);
					W_grad_stock = W_grad_stock.Zero(n_map * pre_map, w_dim);

					/**** フィルタの重みの修正分計算 → スパース行列に再代入という形式になってます ****/
					if (dim == 3)
					{
						int count = 0; //全結合数カウント
						int cout_init = 0;
						int first_unit = 0; //畳み込み最初の画素
						for (int k = 0; k < n_map; k++){ //特徴マップループ
							first_unit = 0;
							for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //前層の特徴マップループ
								first_unit += (pre_map_loop * in_dim - first_unit);
								for (int j = 0; j < ((int)(grad_W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
									int count_w = 0; //フィルタサイズカウント
									if (j != 0 && j % out_size == 0){
										first_unit += (w_size - 1);
									}
									if (j != 0 && j % (out_size * out_size) == 0){
										first_unit += in_size * (w_size - 1);
									}
									for (int i = first_unit; i < grad_W.cols(); i++){ //入力層ユニットループ
										if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
											i += (in_size - w_size);
										}
										if (count_w != 0 && count_w % (w_size * w_size) == 0){
											i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
										}
										W_grad_stock(cout_init, count_w) += grad_W(j + (k * out_dim), i);
										count++;
										count_w++;
										if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
									}
									if (count == w_num) break;
									first_unit++;
								}
								cout_init++;
							}
						}

						//rot180(W_grad_stock, W_grad_stock);
						layer_h[layer].W_stock = layer_h[layer].W_stock - learning_rate * W_grad_stock - lamda * layer_h[layer].W_stock + moment * layer_h[layer].W_stock;


						t.resize(w_num);

						count = 0; //全結合数カウント
						cout_init = 0;
						first_unit = 0; //畳み込み最初の画素
						for (int k = 0; k < n_map; k++){ //特徴マップループ
							first_unit = 0;
							for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //前層の特徴マップループ
								first_unit += (pre_map_loop * in_dim - first_unit);
								for (int j = 0; j < ((int)(grad_W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
									int count_w = 0; //フィルタサイズカウント
									if (j != 0 && j % out_size == 0){
										first_unit += (w_size - 1);
									}
									if (j != 0 && j % (out_size * out_size) == 0){
										first_unit += in_size * (w_size - 1);
									}
									for (int i = first_unit; i < grad_W.cols(); i++){ //入力層ユニットループ
										if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
											i += (in_size - w_size);
										}
										if (count_w != 0 && count_w % (w_size * w_size) == 0){
											i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
										}

										t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, layer_h[layer].W_stock(cout_init, count_w));
										count++;
										count_w++;
										if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
									}
									if (count == w_num) break;
									first_unit++;
								}
								cout_init++;
							}
						}
					}
					else if (dim == 2)
					{
						int count = 0; //全結合数カウント
						int cout_init = 0;
						int first_unit = 0; //畳み込み最初の画素
						for (int k = 0; k < n_map; k++){ //特徴マップループ
							first_unit = 0;


							for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //前層の特徴マップループ
								first_unit += (pre_map_loop * in_dim - first_unit);
								for (int j = 0; j < ((int)(grad_W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
									int count_w = 0; //フィルタサイズカウント
									if (j != 0 && j % out_size == 0){
										first_unit += (w_size - 1);
									}
									for (int i = first_unit; i < grad_W.cols(); i++){ //入力層ユニットループ
										if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
											i += (in_size - w_size);
										}
										W_grad_stock(cout_init, count_w) += grad_W(j + (k * out_dim), i);
										count++;
										count_w++;
										if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
									}
									if (count == w_num) break;
									first_unit++;
								}
								cout_init++;
							}
						}

						//rot180(W_grad_stock, W_grad_stock);
						layer_h[layer].W_stock = layer_h[layer].W_stock - learning_rate * W_grad_stock - lamda * layer_h[layer].W_stock + moment * layer_h[layer].W_stock;
						t.resize(w_num);

						count = 0; //全結合数カウント
						cout_init = 0;
						first_unit = 0; //畳み込み最初の画素

						for (int k = 0; k < n_map; k++){ //特徴マップループ
							first_unit = 0;
							for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //前層の特徴マップループ
								first_unit += (pre_map_loop * in_dim - first_unit);
								for (int j = 0; j < ((int)(grad_W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
									int count_w = 0; //フィルタサイズカウント
									if (j != 0 && j % out_size == 0){
										first_unit += (w_size - 1);
									}
									for (int i = first_unit; i < grad_W.cols(); i++){ //入力層ユニットループ
										if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
											i += (in_size - w_size);
										}

										t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, layer_h[layer].W_stock(cout_init, count_w));
										count++;
										count_w++;
										if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
									}
									if (count == w_num) break;
									first_unit++;
								}
								cout_init++;
							}
						}
					}
					back_delta = layer_h[layer].W.transpose() * back_delta;

					layer_h[layer].W.setFromTriplets(t.begin(), t.end());

					/**** フィルタの重みの修正ここまで ****/

				}
				else if (layer != n_layer - 3 && c_p[layer] == 'p')  //full-connectにつながる以外のプーリング層
				{
					back_delta = layer_h[layer].W.transpose() * back_delta; //誤差δの算出	
				}
				else if (layer == 0)  //入力層につながる畳み込み層
				{

					grad_W.resize(layer_h[layer].W.rows(), layer_h[layer].W.cols());
					grad_W = Eigen::MatrixXf::Zero(grad_W.rows(), grad_W.cols());

					if (layer_h[layer].act == "sigmoid")
					{
						sigmoid_deriv(output_stock[layer + 1], deriv_output);
					}
					else if (layer_h[layer].act == "ReLU")
					{
						Eigen::MatrixXf output_invert;
						ReLU_invert(output_stock[layer + 1], output_invert);
						ReLU_deriv(output_invert, deriv_output);
					}

					back_delta = deriv_output.array() * back_delta.array();

					grad_W = back_delta * output_stock[layer].transpose();

					grad_W /= (float)back_delta.cols();

					pre_map = in_map;
					n_map = layer_h[layer].n_map;
					in_size = layer_h[layer].in_size;
					out_size = layer_h[layer].out_size;
					w_size = layer_h[layer].w_size;
					w_num = layer_h[layer].w_num;
					out_dim = layer_h[layer].out_dim;
					w_dim = layer_h[layer].w_dim;
					in_dim = layer_h[layer].in_dim;

					W_grad_stock.resize(n_map * pre_map, w_dim);
					W_grad_stock = W_grad_stock.Zero(n_map * pre_map, w_dim);

					/**** フィルタの重みの修正分計算 → スパース行列に再代入という形式になってます ****/
					if (dim == 3)
					{
						int count = 0; //全結合数カウント
						int cout_init = 0;
						int first_unit = 0; //畳み込み最初の画素
						for (int k = 0; k < n_map; k++){ //特徴マップループ
							first_unit = 0;
							for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //前層の特徴マップループ
								first_unit += (pre_map_loop * in_dim - first_unit);
								for (int j = 0; j < ((int)(grad_W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
									int count_w = 0; //フィルタサイズカウント
									if (j != 0 && j % out_size == 0){
										first_unit += (w_size - 1);
									}
									if (j != 0 && j % (out_size * out_size) == 0){
										first_unit += in_size * (w_size - 1);
									}
									for (int i = first_unit; i < grad_W.cols(); i++){ //入力層ユニットループ
										if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
											i += (in_size - w_size);
										}
										if (count_w != 0 && count_w % (w_size * w_size) == 0){
											i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
										}
										W_grad_stock(cout_init, count_w) += grad_W(j + (k * out_dim), i);
										count++;
										count_w++;
										if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
									}
									if (count == w_num) break;
									first_unit++;
								}
								cout_init++;
							}
						}

						//rot180(W_grad_stock, W_grad_stock);
						layer_h[layer].W_stock = layer_h[layer].W_stock - learning_rate * W_grad_stock - lamda * layer_h[layer].W_stock + moment * layer_h[layer].W_stock;

						t.resize(w_num);

						count = 0; //全結合数カウント
						cout_init = 0;
						first_unit = 0; //畳み込み最初の画素
						for (int k = 0; k < n_map; k++){ //特徴マップループ
							first_unit = 0;
							for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //前層の特徴マップループ
								first_unit += (pre_map_loop * in_dim - first_unit);
								for (int j = 0; j < ((int)(grad_W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
									int count_w = 0; //フィルタサイズカウント
									if (j != 0 && j % out_size == 0){
										first_unit += (w_size - 1);
									}
									if (j != 0 && j % (out_size * out_size) == 0){
										first_unit += in_size * (w_size - 1);
									}
									for (int i = first_unit; i < grad_W.cols(); i++){ //入力層ユニットループ
										if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
											i += (in_size - w_size);
										}
										if (count_w != 0 && count_w % (w_size * w_size) == 0){
											i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
										}

										t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, layer_h[layer].W_stock(cout_init, count_w));
										count++;
										count_w++;
										if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
									}
									if (count == w_num) break;
									first_unit++;
								}
								cout_init++;
							}
						}
					}
					else if (dim == 2)
					{
						int count = 0; //全結合数カウント
						int cout_init = 0;
						int first_unit = 0; //畳み込み最初の画素

						for (int k = 0; k < n_map; k++){ //特徴マップループ
							first_unit = 0;
							for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //前層の特徴マップループ
								first_unit += (pre_map_loop * in_dim - first_unit);
								for (int j = 0; j < ((int)(grad_W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
									int count_w = 0; //フィルタサイズカウント
									if (j != 0 && j % out_size == 0){
										first_unit += (w_size - 1);
									}
									for (int i = first_unit; i < grad_W.cols(); i++){ //入力層ユニットループ
										if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
											i += (in_size - w_size);
										}
										W_grad_stock(cout_init, count_w) += grad_W(j + (k * out_dim), i);
										count++;
										count_w++;
										if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
									}
									if (count == w_num) break;
									first_unit++;
								}
								cout_init++;
							}
						}

						//rot180(W_grad_stock, W_grad_stock);
						layer_h[layer].W_stock = layer_h[layer].W_stock - learning_rate * W_grad_stock - lamda * layer_h[layer].W_stock + moment * layer_h[layer].W_stock;

						t.resize(w_num);

						count = 0; //全結合数カウント
						cout_init = 0;
						first_unit = 0; //畳み込み最初の画素


						for (int k = 0; k < n_map; k++){ //特徴マップループ
							first_unit = 0;
							for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //前層の特徴マップループ
								first_unit += (pre_map_loop * in_dim - first_unit);
								for (int j = 0; j < ((int)(grad_W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
									int count_w = 0; //フィルタサイズカウント
									if (j != 0 && j % out_size == 0){
										first_unit += (w_size - 1);
									}
									for (int i = first_unit; i < grad_W.cols(); i++){ //入力層ユニットループ
										if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
											i += (in_size - w_size);
										}

										t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, layer_h[layer].W_stock(cout_init, count_w));
										count++;
										count_w++;
										if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
									}
									if (count == w_num) break;
									first_unit++;
								}
								cout_init++;
							}
						}
					}

					layer_h[layer].W.setFromTriplets(t.begin(), t.end());
					/**** フィルタの重みの修正ここまで ****/
				}
			}
			///**** back propagation ここまで ****/

		} // バッチループ
		cost /= (float)batch_num;
		cost_vec.push_back(cost);
		printf("epoch : %d, cost = %f\n", (epoch + 1), cost);


		// Calculate validaton-cost
		Eigen::MatrixXf valid_output;

		for (int layer = 0; layer < n_layer - 1; layer++)
		{
			std::vector<Eigen::MatrixXf> valid_output_stock;


			if (layer == 0){

				layer_h[0].convn(validX, valid_output, layer_h[0].act);

			}
			else if (layer != 0 && c_p[layer] == 'c'){

				layer_h[layer].convn(valid_output, valid_output, layer_h[layer].act);

			}
			else if (layer != 0 && c_p[layer] == 'p')
			{

				layer_h[layer].pooling(valid_output, valid_output, layer_h[layer].act);

			}
			else if (layer == n_layer - 2)
			{

				float valid_cost = 0.0;
				layer_f[0].valid_test(valid_output, validY, valid_cost);

				std::cout << "validation-cost : " << valid_cost << std::endl;
				valid_cost_vec.push_back(valid_cost);
			}
		}


		// validation_num回おきに検定を行う

		if ((valid_cost_avg.size() > 0) && ((epoch + 1) % validation_num == 0)){
			std::cout << count_valid + 1 << "回目の検定を行います(*'ω'*)" << std::endl;
			///////////////////////////////////////////////////////////
			float total = 0;
			for (int val_n = 0; val_n < validation_num; val_n++)
			{
				total += valid_cost_vec[epoch - val_n];
			}
			valid_cost_avg.push_back(total / validation_num);


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
		if (valid_cost_vec.size() == validation_num)
		{
			float total = 0;
			for (int val_n = 0; val_n < validation_num; val_n++)
			{
				total += valid_cost_vec[val_n];
			}
			valid_cost_avg.push_back(total / validation_num);
		}




	} // epochループ

	for (int i = 0; i < layer_h.size(); i++){
		Eigen::MatrixXf W = layer_h[i].W;
		write_raw_and_txt(W, dir_o + "/hidden_W" + std::to_string(i));
	}

	for (int i = 0; i < layer_f.size(); i++){
		write_raw_and_txt(layer_f[i].weight1, dir_o + "/full_W");
	}
	vec_to_txt(cost_vec, dir_o + "/cost.txt");
	vec_to_txt(valid_cost_vec, dir_o + "/valid_cost.txt");

}

void DCNN::predict(Eigen::MatrixXf& INPUT, Eigen::MatrixXf& OUTPUT)
{
	for (int layer = 0; layer < n_layer - 1; layer++)
	{

		if (layer == 0){
			layer_h[0].convn(INPUT, OUTPUT, layer_h[layer].act);


		}
		else if (layer != 0 && c_p[layer] == 'c'){
			layer_h[layer].convn(OUTPUT, OUTPUT, layer_h[layer].act);
		}
		else if (layer != 0 && c_p[layer] == 'p')
		{
			layer_h[layer].pooling(OUTPUT, OUTPUT, layer_h[layer].act);
		}
		else if (layer == n_layer - 2)
		{
			layer_f[0].forward_propagation(OUTPUT, OUTPUT);

		}
	}
}