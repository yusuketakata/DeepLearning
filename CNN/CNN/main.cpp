#include"Header.h"
#include"dcnn.h"
#include"cnn.h"
#include "mlp.h"
#include <time.h>
#include "Utils.h"
#include "cnn_info.h"
#include"random_shuffle.h"

int main(int argc, char *argv[]){

	dcnn_info input_info;//入力情報
	if (argc != 2)
	{
		std::cout << "usage : DCNN.exe input_info.txt" << std::endl << "Hit enter-key to exit...";
		getchar();
		exit(1);
	}
	input_info.input(argv[1]);

	Eigen::setNbThreads(input_info.thread);
	Eigen::initParallel();

	//*** 分割したstringを格納するためのダミー ***//
	std::vector<std::string> process_dummy = split(input_info.process, ',');
	std::vector<std::string> activation_dummy = split(input_info.activation, ',');
	std::vector<std::string> n_map_dummy = split(input_info.n_map, ',');
	std::vector<std::string> w_size_dummy = split(input_info.w_size, ',');
	std::vector<std::string> pool_size_dummy = split(input_info.pool_size, ',');

	std::vector<char> process;
	std::vector<std::string> activation;
	std::vector<int> n_map;
	std::vector<int> w_size;
	std::vector<int> pool_size;

	for (int i = 0; i < (int)process_dummy.size(); i++){
		char cpy = process_dummy[i][0];
		process.push_back(cpy);
	}

	for (int i = 0; i < (int)activation_dummy.size(); i++){
		std::string cpy_act = activation_dummy[i];
		activation.push_back(cpy_act);
	}

	activation[(int)activation.size() - 1] = Replace(activation[(int)activation.size() - 1], "\\", "");

	n_map_dummy[n_map_dummy.size() - 1] = Replace(n_map_dummy[n_map_dummy.size() - 1], "\\", "");
	w_size_dummy[w_size_dummy.size() - 1] = Replace(w_size_dummy[w_size_dummy.size() - 1], "\\", "");
	pool_size_dummy[pool_size_dummy.size() - 1] = Replace(pool_size_dummy[pool_size_dummy.size() - 1], "\\", "");

	for (int i = 0; i < (int)n_map_dummy.size(); i++){
		n_map.push_back(std::atoi(n_map_dummy[i].c_str()));
	}
	for (int i = 0; i < (int)w_size_dummy.size(); i++){
		w_size.push_back(std::atoi(w_size_dummy[i].c_str()));
	}
	for (int i = 0; i < (int)pool_size_dummy.size(); i++){
		pool_size.push_back(std::atoi(pool_size_dummy[i].c_str()));
	}

	int in_map = input_info.input_map;
	int in_dim;
	int in = input_info.in_size;
	int out = input_info.out_size;
	if (input_info.dim == 3)
	{
		in_dim = in * in * in * in_map;
	}
	else if (input_info.dim == 2)
	{
		in_dim = in * in * in_map;
	}

	////*** テキスト情報読み込みここまで ***////


	std::cout << "Please Input CV_group Number" << std::endl;
	std::vector<int> group;
	for (int cv = 0; cv < input_info.CV_group; cv++){
		int p;
		std::cin >> p;
		group.push_back(p);
	}

	////////////////////　Cross_Validation ループ ////////////////////////////

	for (int cv_loop = 0; cv_loop < group.size(); cv_loop++)
	{

		///////////////// DCNNのオブジェクトの初期化 ///////////////////////

		std::cout << "Group" + std::to_string(group[cv_loop]) + " Start" << std::endl;
		DCNN dcnn(input_info, process, in_map, n_map, w_size, pool_size, activation, cv_loop, input_info.dropout);

		/**** 入力は(次元数，サンプル数)の行列で与えてください ****/
		/**** 正解は(クラス数, サンプル数)の行列で与えてください ****/

		// 学習データ読み込み ////////
		std::vector<std::string> train_name;
		std::cout << input_info.name_txt << std::endl;
		std::ifstream file(input_info.name_txt + "/group" + std::to_string(group[cv_loop]) + "/train_name.txt");
		std::string buf;
		while (file && getline(file, buf))
		{
			train_name.push_back(buf);
		}

		std::cout << "load train_data Now v(^_^)v" << std::endl;

		std::vector<std::vector<float>> train_X(train_name.size());
		std::vector<std::vector<unsigned char>> train_Y(train_name.size());

		for (size_t n_case = 0; n_case < train_name.size(); n_case++){

			read_vector(train_X[n_case], input_info.dir_i + "/group" + std::to_string(group[cv_loop]) + '/' + train_name[n_case] + ".raw");
			read_vector(train_Y[n_case], input_info.dir_a + "/" + train_name[n_case] + ".raw");

		}

		Eigen::MatrixXf trainX;
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> trainY;

		random_sort(train_X, train_Y, trainX, trainY, train_name.size(), in_dim);


		std::vector<std::vector<float>>().swap(train_X);
		std::vector<std::vector<unsigned char>>().swap(train_Y);

		// 交差検定データ読み込み ////////
		std::vector<std::string> valid_name;
		std::cout << input_info.name_txt << std::endl;
		std::ifstream file_(input_info.name_txt + "/group" + std::to_string(group[cv_loop]) + "/valid_name.txt");
		std::string buf_;
		while (file_ && getline(file_, buf_))
		{
			valid_name.push_back(buf_);
		}

		std::cout << "load valid_data Now v(^_^)v" << std::endl;

		std::vector<std::vector<float>> valid_X(valid_name.size());
		std::vector<std::vector<unsigned char>> valid_Y(valid_name.size());


		for (size_t n_case = 0; n_case < valid_name.size(); n_case++)
		{
			read_vector(valid_X[n_case], input_info.dir_i + "/group" + std::to_string(group[cv_loop]) + '/' + valid_name[n_case] + ".raw");
			read_vector(valid_Y[n_case], input_info.dir_a + "/" + valid_name[n_case] + ".raw");
		}


		Eigen::MatrixXf validX;
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> validY;

		random_sort(valid_X, valid_Y, validX, validY, valid_name.size(), in_dim);


		std::vector<std::vector<float>>().swap(valid_X);
		std::vector<std::vector<unsigned char>>().swap(valid_Y);



		clock_t start, end;
		start = clock();

		std::string dir_o = input_info.dir_o + "/param" + input_info.param + "/group" + std::to_string(group[cv_loop]);

		if (!nari::system::directry_is_exist(dir_o))
			nari::system::make_directry(dir_o);



		dcnn.train(dir_o, trainX, trainY, validX, validY, input_info.epoch, input_info.batch_size, (float)input_info.alpha, input_info.moment, input_info.lamda, input_info.dropout);

		end = clock();
		double ti = (double)(end - start) / CLOCKS_PER_SEC;
		std::cout << "train for : " << ti << "[s]" << std::endl;



		std::cout << "//*** TEST START ***//" << std::endl;


		std::vector<std::string> test_name;
		std::cout << input_info.name_txt << std::endl;
		std::ifstream file__(input_info.name_txt + "/group" + std::to_string(group[cv_loop]) + "/test_name.txt");
		std::string buf__;
		while (file__ && getline(file__, buf__))
		{
			test_name.push_back(buf__);
		}

		for (size_t n_test = 0; n_test < test_name.size(); n_test++)
		{

			std::vector<float> test_X;
			std::cout << "loading test_data Now v(^_^)v" << std::endl;

			read_vector(test_X, input_info.dir_t + "/" + test_name[n_test] + ".raw");

			Eigen::MatrixXf testX(in_dim, test_X.size() / in_dim);
			for (long long yy = 0; yy < test_X.size() / in_dim; yy++){
				for (int xx = 0; xx < in_dim; xx++){
					testX(xx, yy) = test_X[yy * in_dim + xx];
				}
			}

			Eigen::MatrixXf OUTPUT;
			start = clock();

			dcnn.predict(testX, OUTPUT);

			end = clock();
			ti = (double)(end - start) / CLOCKS_PER_SEC;
			std::cout << "predict for : " << ti << "[s]" << std::endl;


			std::string dir_o = input_info.dir_o + "/param" + input_info.param + "/predict";
			if (!nari::system::directry_is_exist(dir_o))
				nari::system::make_directry(dir_o);

			write_raw_and_txt(OUTPUT, dir_o + "/" + test_name[n_test]);

		}
	}
	return 0;
}