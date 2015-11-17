#pragma once
#include"Header.h"
#include "Utils.h"
#include "mlp.h"


class CNN
{
private:
public:
	Eigen::MatrixXf W_init; //Initial filter
	Eigen::MatrixXf W_stock; //スパース行列の重み修正用
	Eigen::SparseMatrix<float> W; //Sparse Weight for convolution & pooling layer
	//Eigen::VectorXf b; //bias for convolution & pooling layer
	std::vector<Eigen::Triplet<float>> t;
	int w_num;
	int in_size;   //Input image size
	int n_map;     //Number of feature map
	int w_size;    //Filter size
	int out_size;  //Output size
	int in_dim;
	int out_dim;
	int w_dim;
	int dim;

	std::string act;

	CNN(std::string, int, int, int, int, char, int, int, std::string, int, int, int);
	CNN(std::string, int, int, int, int, int, char, int, int, std::string, int, int, int); //2層目以降Convolution

	void convn(Eigen::MatrixXf&, Eigen::MatrixXf&, std::string);
	void pooling(Eigen::MatrixXf&, Eigen::MatrixXf&, std::string, int);

};


// 1層目のコンストラクタ
CNN::CNN(std::string dir_p, int in_s, int in_map, int n_m, int w_s, char p, int out_s, int _dim, std::string _act, int ini_switch, int cv_loop, int layer)
{
	dim = _dim;

	if (p == 'c'){
		in_size = in_s;
		n_map = n_m;
		w_size = w_s;
		out_size = out_s;
		act = _act;

		try{
			if (in_s < w_s){
				throw "( ・_ゝ・)このフィルタサイズはおかしいよ!!( ・_ゝ・)";
			}
		}
		catch (char* error){
			std::cout << error << std::endl;
			exit(1);
		}


		//int in_dim;
		//int out_dim;
		//int w_dim;

		if (dim == 2){
			in_dim = in_size *in_size;
			out_dim = out_size * out_size;
			w_dim = w_size * w_size;
		}

		if (dim == 3){
			in_dim = in_size *in_size * in_size;
			out_dim = out_size * out_size * out_size;
			w_dim = w_size * w_size * w_size;
		}

		W_init = W_init.Zero(n_map * in_map, w_dim);
		//b = Eigen::VectorXf::Zero(n_map);

		int fan_in;
		fan_in = in_map * w_size * w_size;

		dist = std::uniform_real_distribution<float>(-(float)sqrt(3.0 / (float)fan_in), (float)sqrt(3.0 / (float)fan_in));

		if (ini_switch == 0){
			for (int j = 0; j < W_init.rows(); j++){
				for (int i = 0; i < W_init.cols(); i++){
					W_init(j, i) = dist(engine);
				}
			}
		}
		else if (ini_switch == 1){
			std::vector<float> W_dummy;
			read_vector(W_dummy, dir_p + "/group" + std::to_string(cv_loop+1) + "/hidden_W" + std::to_string(layer) + ".raw");
			for (int j = 0; j < W_init.cols(); j++){
				for (int i = 0; i < W_init.rows(); i++){
					W_init(i, j) = W_dummy[j * W_init.rows() + i];
					
				}
			}
			std::cout << "param get" << std::endl;
		}

		W_stock = W_init;

		W.resize(out_dim * n_map, in_dim * in_map);
		std::cout << "layer_c = " << W.rows() << ", " << W.cols() << std::endl; ////////////////////////////
		w_num = w_dim *  (int)W.rows() * in_map;

		t.resize(w_num);

		int count = 0; //全結合数カウント
		int cout_init = 0;
		int first_unit = 0; //畳み込み最初の画素
		if (dim == 3)
		{
			for (int k = 0; k < n_map; k++){ //特徴マップループ
				first_unit = 0;
				for (int pre_map_loop = 0; pre_map_loop < in_map; pre_map_loop++){ //入力層の特徴マップループ
					first_unit += (pre_map_loop * in_dim - first_unit);
					for (int j = 0; j < ((int)(W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
						int count_w = 0; //フィルタサイズカウント
						if (j != 0 && j % out_size == 0){
							first_unit += (w_size - 1);
						}
						if (j != 0 && j % (out_size * out_size) == 0){
							first_unit += in_size * (w_size - 1);
						}
						for (int i = first_unit; i < (int)W.cols(); i++){ //入力層ユニットループ
							if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
								i += (in_size - w_size);
							}
							if (count_w != 0 && count_w % (w_size * w_size) == 0){
								i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
							}

							t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, W_init(cout_init, count_w));
							count++;
							count_w++;
							if (count_w == w_size * w_size * w_size)	break; //フィルタサイズ分代入したら終了
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
			for (int k = 0; k < n_map; k++){ //特徴マップループ
				first_unit = 0;
				for (int pre_map_loop = 0; pre_map_loop < in_map; pre_map_loop++){ ///入力層の特徴マップループ
					first_unit += (pre_map_loop * in_dim - first_unit);
					for (int j = 0; j < ((int)(W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
						int count_w = 0; //フィルタサイズカウント
						if (j != 0 && j % out_size == 0){
							first_unit += (w_size - 1);
						}
						for (int i = first_unit; i < (int)W.cols(); i++){ //入力層ユニットループ
							if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
								i += (in_size - w_size);
							}

							t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, W_init(cout_init, count_w));
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

		W.setFromTriplets(t.begin(), t.end());
	}

	if (p == 'p'){
		in_size = in_s;
		n_map = n_m;
		w_size = w_s;
		out_size = out_s;
		act = _act;

		int in_dim = in_size*in_size*in_size;
		int w_dim = w_size*w_size*w_size;
		int out_dim = out_size*out_size*out_size;

		//プーリング用の仮想的なスパース行列
		W.resize(in_dim * n_map / (w_dim), in_dim * n_map);
		std::cout << "layer_p = " << W.rows() << ", " << W.cols() << std::endl; /////////////

		//スパース行列のnon-zero成分数
		w_num = w_dim * (int)W.rows();
		t.resize(w_num);


		/**** こっから仮想的な重み行列作成 ****/
		int count = 0; //全結合数カウント
		int first_unit = 0; //畳み込み最初の画素

		for (int j = 0; j < (int)(W.rows()); j++){ //出力層ユニットループ(1つの特徴マップ分)

			int count_w = 0; //フィルタサイズカウント
			if (j == 0){

			}


			else if (j % (in_size / w_size) != 0){
				first_unit += (w_size - 1);
			}
			else if (j % (in_size / w_size) == 0 && j % (out_size * out_size) != 0){
				first_unit += (w_size - 1)*in_size + w_size - 1;
			}
			else if (j % (out_size * out_size) == 0){
				first_unit += (w_size - 1)*in_size + w_size - 1 + (in_size * in_size * (w_size - 1));
			}

			for (int i = first_unit; i < (int)W.cols(); i++){ //入力層ユニットループ
				if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
					i += (in_size - w_size);
				}
				if (count_w != 0 && count_w % (w_size * w_size) == 0){
					i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
				}

				t[count] = Eigen::Triplet<float>(j, i, (float)1 / w_dim);

				count++;
				count_w++;
				if (count_w == w_size * w_size * w_size)	break; //フィルタサイズ分代入したら終了
			}
			if (count == w_num) break;
			first_unit++;
		}


		W.setFromTriplets(t.begin(), t.end());


	}


}

// 2層目以降のコンストラクタ
CNN::CNN(std::string dir_p, int in_s, int pre_out, int pre_map, int n_m, int w_s, char p, int out_s, int _dim, std::string _act, int ini_switch, int cv_loop, int layer)
{
	dim = _dim;

	if (p == 'c'){
		in_size = in_s;
		n_map = n_m;
		w_size = w_s;
		out_size = out_s;
		act = _act;

		try{
			if (in_s < w_s){
				throw "( ・_ゝ・)このフィルタサイズはおかしいよ!!( ・_ゝ・)";
			}
		}
		catch (char* error){
			std::cout << error << std::endl;
			exit(1);
		}

		//int in_dim;
		//int out_dim;
		//int w_dim;

		if (dim == 2){
			in_dim = in_size *in_size;
			out_dim = out_size * out_size;
			w_dim = w_size * w_size;
		}

		if (dim == 3){
			in_dim = in_size *in_size * in_size;
			out_dim = out_size * out_size * out_size;
			w_dim = w_size * w_size * w_size;
		}
		//int pre_out = in_size * dim * n_map;
		//int pre_map = pre_out / in_dim;

		/******************************************************/
		/* 岸本修正 : 入力の特徴マップが複数枚ある場合に対応  */
		/******************************************************/
		W_init = W_init.Zero(n_map * pre_map, w_dim);

		int fan_in;
		fan_in = n_map * w_size * w_size;
		dist = std::uniform_real_distribution<float>(-(float)sqrt(3.0 / (float)fan_in), (float)sqrt(3.0 / (float)fan_in));
		if (ini_switch == 0){
			for (int j = 0; j < W_init.rows(); j++){
				for (int i = 0; i < W_init.cols(); i++){
					W_init(j, i) = dist(engine);
				}
			}
		}
		else if (ini_switch == 1){

			std::vector<float> W_dummy;
			read_vector(W_dummy, dir_p + "/group" + std::to_string(cv_loop+1) + "/hidden_W" + std::to_string(layer) + ".raw");

			for (int j = 0; j < W_init.cols(); j++){
				for (int i = 0; i < W_init.rows(); i++){
					W_init(i, j) = W_dummy[j * W_init.rows() + i];

				}
			}
			std::cout << "param get" << std::endl;

		}

		W_stock = W_init;

		W.resize(out_dim * n_map, pre_out);
		std::cout << "layer_c = " << W.rows() << ", " << W.cols() << std::endl; ////////////////
		w_num = w_dim *  (int)W.rows() * pre_map;

		t.resize(w_num);


		int count = 0; //全結合数カウント
		int cout_init = 0;
		int first_unit = 0; //畳み込み最初の画素
		if (dim == 3)
		{
			for (int k = 0; k < n_map; k++){ //特徴マップループ
				first_unit = 0;
				for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //前層の特徴マップループ
					first_unit += (pre_map_loop * in_dim - first_unit);
					for (int j = 0; j < ((int)(W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
						int count_w = 0; //フィルタサイズカウント
						if (j != 0 && j % out_size == 0){
							first_unit += (w_size - 1);
						}
						if (j != 0 && j % (out_size * out_size) == 0){
							first_unit += in_size * (w_size - 1);
						}
						for (int i = first_unit; i < (int)W.cols(); i++){ //入力層ユニットループ
							if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
								i += (in_size - w_size);
							}
							if (count_w != 0 && count_w % (w_size * w_size) == 0){
								i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
							}

							t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, W_init(cout_init, count_w));
							count++;
							count_w++;
							if (count_w == w_size * w_size * w_size)	break; //フィルタサイズ分代入したら終了
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
			for (int k = 0; k < n_map; k++){ //特徴マップループ
				first_unit = 0;
				for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //前層の特徴マップループ
					first_unit += (pre_map_loop * in_dim - first_unit);
					for (int j = 0; j < ((int)(W.rows()) / n_map); j++){ //出力層ユニットループ(1つの特徴マップ分)
						int count_w = 0; //フィルタサイズカウント
						if (j != 0 && j % out_size == 0){
							first_unit += (w_size - 1);
						}
						for (int i = first_unit; i < (int)W.cols(); i++){ //入力層ユニットループ
							if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
								i += (in_size - w_size);
							}

							t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, W_init(cout_init, count_w));
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


		W.setFromTriplets(t.begin(), t.end());

		/*********************/
		/* 岸本修正 ここまで */
		/*********************/

	}

	if (p == 'p'){
		in_size = in_s;
		w_size = w_s;
		n_map = n_m;
		out_size = out_s;
		act = _act;

		try{
			if (in_s % w_s != 0){
				std::cout << in_s << " % " << w_s << " = " << in_s % w_s << std::endl;
				throw "( ・_ゝ・)このPoolingサイズでは割り切れないよ!!( ・_ゝ・)";
			}
		}
		catch (char* error){
			std::cout << error << std::endl;
			exit(1);
		}

		if (dim == 2){
			in_dim = in_size *in_size;
			out_dim = out_size * out_size;
			w_dim = w_size * w_size;
		}

		if (dim == 3){
			in_dim = in_size *in_size * in_size;
			out_dim = out_size * out_size * out_size;
			w_dim = w_size * w_size * w_size;
		}

		//プーリング用の仮想的なスパース行列
		W.resize(pre_out / (w_dim), pre_out);
		std::cout << "layer_p = " << W.rows() << ", " << W.cols() << std::endl; ////////////////////////////

		//スパース行列のnon-zero成分数
		w_num = w_dim * (int)W.rows();
		t.resize(w_num);


		/**** こっから仮想的な重み行列作成 ****/
		int count = 0; //全結合数カウント
		int first_unit = 0; //畳み込み最初の画素

		if (dim == 3)
		{
			for (int j = 0; j < (int)(W.rows()); j++){ //出力層ユニットループ(1つの特徴マップ分)

				int count_w = 0; //フィルタサイズカウント
				if (j == 0){

				}


				else if (j % (in_size / w_size) != 0){
					first_unit += (w_size - 1);
				}
				else if (j % (in_size / w_size) == 0 && j % (out_size * out_size) != 0){
					first_unit += (w_size - 1)*in_size + w_size - 1;
				}
				else if (j % (out_size * out_size) == 0){
					first_unit += (w_size - 1)*in_size + w_size - 1 + (in_size * in_size * (w_size - 1));
				}

				for (int i = first_unit; i < (int)W.cols(); i++){ //入力層ユニットループ
					if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
						i += (in_size - w_size);
					}
					if (count_w != 0 && count_w % (w_size * w_size) == 0){
						i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
					}
					t[count] = Eigen::Triplet<float>(j, i, (float)1 / w_dim);
					count++;
					count_w++;
					if (count_w == w_size * w_size * w_size)	break; //フィルタサイズ分代入したら終了
				}
				if (count == w_num) break;
				first_unit++;
			}
		}
		else if (dim == 2)
		{
			for (int j = 0; j < (int)(W.rows()); j++){ //出力層ユニットループ(1つの特徴マップ分)

				int count_w = 0; //フィルタサイズカウント
				if (j == 0){

				}


				else if (j % (in_size / w_size) != 0){
					first_unit += (w_size - 1);
				}
				else if (j % (in_size / w_size) == 0 && j % (out_size * out_size) != 0){
					first_unit += (w_size - 1)*in_size + w_size - 1;
				}
				else if (j % (out_size * out_size) == 0){
					first_unit += (w_size - 1)*in_size + w_size - 1;
				}

				for (int i = first_unit; i < (int)W.cols(); i++){ //入力層ユニットループ
					if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
						i += (in_size - w_size);
					}
					t[count] = Eigen::Triplet<float>(j, i, (float)1 / w_dim);
					count++;
					count_w++;
					if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
				}
				if (count == w_num) break;
				first_unit++;
			}
		}


		W.setFromTriplets(t.begin(), t.end());


	}

}


void CNN::convn(Eigen::MatrixXf& INPUT, Eigen::MatrixXf& OUTPUT, std::string act_func = "sigmoid"){
	OUTPUT = W * INPUT;

	if (act_func == "sigmoid"){
		sigmoid(OUTPUT, OUTPUT);
	}

	else if (act_func == "ReLU")
	{
		ReLU(OUTPUT, OUTPUT);
	}
}

void CNN::pooling(Eigen::MatrixXf& INPUT, Eigen::MatrixXf& OUTPUT, std::string pool = "average", int p = 2){
	Eigen::MatrixXf stock_output(W.rows(), OUTPUT.cols());
	if (pool == "average"){
		for (int i = 0; i < (int)INPUT.cols(); i++)
		{
			stock_output.block(0, i, stock_output.rows(), 1) = W * INPUT.col(i);
		}
		OUTPUT = stock_output;
	}

	if (pool == "max"){
		for (int batch = 0; batch < (int)INPUT.cols(); batch++)
		{
			int count = 0; //全結合数カウント
			int first_unit = 0; //畳み込み最初の画素

			//スパース行列のnon-zero成分数
			w_num = (int)W.rows();
			t.resize(w_num);
			if (dim == 3)
			{
				for (int j = 0; j < (int)(W.rows()); j++){ //出力層ユニットループ(1つの特徴マップ分)
					float w_max = FLT_MIN;
					int index;
					int count_w = 0; //フィルタサイズカウント
					if (j == 0){

					}


					else if (j % (in_size / w_size) != 0){
						first_unit += (w_size - 1);
					}
					else if (j % (in_size / w_size) == 0 && j % (out_size * out_size) != 0){
						first_unit += (w_size - 1)*in_size + w_size - 1;
					}
					else if (j % (out_size * out_size) == 0){
						first_unit += (w_size - 1)*in_size + w_size - 1 + (in_size * in_size * (w_size - 1));
					}

					for (int i = first_unit; i < (int)W.cols(); i++){ //入力層ユニットループ
						if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
							i += (in_size - w_size);
						}
						if (count_w != 0 && count_w % (w_size * w_size) == 0){
							i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
						}

						if (count_w == 0)
						{
							index = i;
							w_max = INPUT(i, batch);
						}
						else
						{
							if (w_max < INPUT(i, batch))
							{
								index = i;
								w_max = INPUT(i, batch);
							}
						}
						count_w++;
						if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
					}
					t[count] = Eigen::Triplet<float>(j, index, 1.0);
					count++;
					if (count == w_num) break;
					first_unit++;
				}
			}
			else if (dim == 2)
			{
				for (int j = 0; j < (int)(W.rows()); j++){ //出力層ユニットループ(1つの特徴マップ分)
					float w_max = FLT_MIN;
					int index;
					int count_w = 0; //フィルタサイズカウント
					if (j == 0){

					}

					else if (j % (in_size / w_size) != 0){
						first_unit += (w_size - 1);
					}
					else if (j % (in_size / w_size) == 0 && j % (out_size * out_size) != 0){
						first_unit += (w_size - 1)*in_size + w_size - 1;
					}
					else if (j % (out_size * out_size) == 0){
						first_unit += (w_size - 1)*in_size + w_size - 1;
					}

					for (int i = first_unit; i < (int)W.cols(); i++){ //入力層ユニットループ
						if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
							i += (in_size - w_size);
						}

						if (count_w == 0)
						{
							index = i;
							w_max = INPUT(i, batch);
						}
						else
						{
							if (w_max < INPUT(i, batch))
							{
								index = i;
								w_max = INPUT(i, batch);
							}
						}
						t[count] = Eigen::Triplet<float>(j, i, (float)1 / w_dim);
						count_w++;
						if (count_w == w_dim)	break; //フィルタサイズ分代入したら終了
					}
					t[count] = Eigen::Triplet<float>(j, index, 1.0);
					count++;
					if (count == w_num) break;
					first_unit++;
				}
			}

			W.setFromTriplets(t.begin(), t.end());
			stock_output.block(0, batch, stock_output.rows(), 1) = W * INPUT.col(batch);

		}
		OUTPUT = stock_output;
	}

	if (pool == "Lp"){

	}

}