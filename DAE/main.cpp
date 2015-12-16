#include "Header.h"
#include "autoencoder.h"
#include "sdae_info.h"
#include "tktlib/raw_io.h"
#include "tktlib/utility.h"
#include "random_shuffle.h"
#include "sdae.h"


void add_noise(const Eigen::MatrixXf& Input, Eigen::MatrixXf& Output, int type, int rate)
{
	///////////////////////////////////////////////////////////////
	// 入力にノイズを加える関数								     //
	// type : 1... pepper noise									 //
	//		: 2... occlusion									 //
	///////////////////////////////////////////////////////////////	

	Output = Input;

	if (type == 1)
	{
		std::cout << "Noise type is solt : rate = " << (float)rate / 10 << std::endl;
		dist = std::uniform_real_distribution<float>(0, 1);

		for (long int yy = 0; yy < Output.cols(); yy++)
		{
			for (int xx = 0; xx < Output.rows(); xx++)
			{
				float no = dist(engine);
				if (no < (rate / 10.))
				{
					Output(xx, yy) = 0;
				}
			}
		}
	}
	else if (type == 2)
	{
		std::cout << "Noise type is occlusion : size = " << rate << std::endl;

		dist_int = std::uniform_int_distribution<>(0, 9 - rate);        // 取りあえず3次元でパッチサイズ9の場合のみに対応

		for (long int sample = 0; sample < Input.cols(); sample++)
		{
			int x = dist_int(engine);
			int y = dist_int(engine);
			int z = dist_int(engine);

			for (int size_z = 0; size_z < rate; size_z++)
			{
				for (int size_y = 0; size_y < rate; size_y++)
				{
					for (int size_x = 0; size_x < rate; size_x++)
					{
						int s = (z + size_z) * 9 * 9 + (y + size_y) * 9 + (x + size_x);
						Output(s, sample) = 0;
					}
				}

			}
		}

	}
	else
	{
		std::cout << "No Noise !!! " << std::endl;
		Output = Input;
	}
}

void main(int argc, char *argv[])
{


	sdae_info input_info;//入力情報
	if (argc != 2)
	{
		std::cout << "usage : Autoencoder.exe input_info.txt" << std::endl << "Hit enter-key to exit...";
		getchar();
		exit(1);
	}
	input_info.input(argv[1]);

	Eigen::setNbThreads(input_info.num_thread);

	// 分割したstringを格納するためのダミー//////////////////
	std::vector<std::string> n_unit_dummy = split(input_info.n_unit, ',');

	std::vector<int> n_unit;

	n_unit_dummy[n_unit_dummy.size() - 1] = Replace(n_unit_dummy[n_unit_dummy.size() - 1], "/", "");

	for (int i = 0; i < n_unit_dummy.size(); i++){
		n_unit.push_back(std::stoi(n_unit_dummy[i]));
	}

	std::cout << "Please Input CV_group Number" << std::endl;
	std::vector<int> group;
	for (int cv = 0; cv < input_info.CV_group; cv++){
		int p;
		std::cin >> p;
		group.push_back(p);
	}

	////////////////////　Cross_Validation ループ ////////////////////////////

	for (int cv_loop = 0; cv_loop < input_info.CV_group; cv_loop++)
	{

		///////////////// Autoencoderのオブジェクトの初期化 ///////////////////////

		std::cout << "Group" + std::to_string(group[cv_loop]) + " Start" << std::endl;

		hidden_layer h;


		if (input_info.ini_switch == 0){ // All Random Initialize

			for (int i = 0; i < input_info.n_layer; i++){
				AE dae(n_unit[i], n_unit[i + 1], input_info.batch_size);
				h.push_back(dae);
			}
		}

		//if (input_info.ini_switch == 1){ // Use pretrained Initialize

		//	for (int i = 0; i < input_info.n_layer; i++){

		//		std::string dir_p_ = input_info.dir_p + "/layer" + std::to_string(i + 1);
		//		std::cout << input_info.dir_p + "/layer" + std::to_string(i + 1) << std::endl;

		//		if (!nari::system::directry_is_exist(input_info.dir_p + "/group" + std::to_string(group[cv_loop]) + "/pretrain/layer" + std::to_string(i + 1))){
		//			AE sdae(n_unit[i], n_unit[i + 1], input_info.batch_size);
		//			h.push_back(sdae);
		//		}
		//		else{
		//			std::vector<double> W(n_unit[i + 1] * n_unit[i]);
		//			std::vector<double> bout(n_unit[i]);
		//			std::vector<double> bhid(n_unit[i + 1]);



		//			read_vector(W, input_info.dir_p + "/group" + std::to_string(group[cv_loop]) + "/pretrain/layer" + std::to_string(i + 1) + "/W.raw");
		//			read_vector(bout, input_info.dir_p + "/group" + std::to_string(group[cv_loop]) + "/pretrain/layer" + std::to_string(i + 1) + "/b.raw");
		//			read_vector(bhid, input_info.dir_p + "/group" + std::to_string(group[cv_loop]) + "/pretrain/layer" + std::to_string(i + 1) + "/c.raw");


		//			RBM rbm(n_unit[i], n_unit[i + 1], input_info.batch_size, W, b, c);
		//			h.push_back(rbm);
		//		}


		//	}
		//}


		///////////////////////////////////////////////////////////////////

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

		for (size_t n_case = 0; n_case < train_name.size(); n_case++)
		{
			read_vector(train_X[n_case], input_info.dir_i + "/" + train_name[n_case] + ".raw");
		}


		Eigen::MatrixXf trainX;  //正解
		Eigen::MatrixXf trainN;  //feed-forward用

		random_sort(train_X, trainX, train_name.size(), n_unit[0]);
		std::vector<std::vector<float>>().swap(train_X);

		// ノイズを加える
		add_noise(trainX, trainN, input_info.type, input_info.rate);

		////////////////////////////////////////////////////////
		// 交差検定用データ読み込み //
		std::vector<std::string> valid_name;
		std::cout << input_info.name_txt << std::endl;
		std::ifstream file_(input_info.name_txt + "/group" + std::to_string(group[cv_loop]) + "/valid_name.txt");
		std::string buf_;
		while (file_ && getline(file_, buf_))
		{
			valid_name.push_back(buf_);
		}

		std::vector<std::vector<float>> valid_X(valid_name.size());
		std::cout << "load validation_data Now v(^_^)v" << std::endl;

		for (size_t n_case = 0; n_case < valid_name.size(); n_case++)
		{
			read_vector(valid_X[n_case], input_info.dir_i + "/" + valid_name[n_case] + ".raw");
		}
		///////////////////////////////////////////////////////////

		Eigen::MatrixXf validX;
		Eigen::MatrixXf validN;

		random_sort(valid_X, validX, valid_name.size(), n_unit[0]);
		std::vector<std::vector<float>>().swap(valid_X);

		add_noise(validX, validN, input_info.type, input_info.rate);

		std::cout << "/////////////// PRETRAINING START ///////////////" << std::endl;

		std::string dir_o;
		dir_o = input_info.dir_o + "/autoencoder/noise_" + std::to_string(input_info.type) + "_" + std::to_string(input_info.rate) + "/param" + input_info.param + "/dae/group" + std::to_string(group[cv_loop]);

		Eigen::MatrixXf layer_train_input = trainN;
		Eigen::MatrixXf layer_train_answer = trainX;
		Eigen::MatrixXf layer_valid_input = validN;
		Eigen::MatrixXf layer_valid_answer = validX;

		for (size_t layer = 0; layer < input_info.n_layer; layer++)
		{
			std::string num = std::to_string(layer + 1);

			if (!nari::system::directry_is_exist(dir_o + "/layer" + num)) nari::system::make_directry(dir_o + "/layer" + num);

			h.pretraining(layer, layer_train_input, layer_train_answer, layer_valid_input, layer_valid_answer, (double)input_info.alpha, input_info.epoch, input_info.batch_size,
				n_unit[layer + 1], dir_o + "/layer" + num);

			std::cout << "DAE" + num + "finish" << std::endl;

			std::cout << "make_hidden_train" << std::endl;


			h.hidden_train_data(layer, layer_train_input);
			h.hidden_train_data(layer, layer_train_answer);
			h.hidden_train_data(layer, layer_valid_input);
			h.hidden_train_data(layer, layer_valid_answer);



			//////// 中間層の特徴量を全データ分保存
			//std::vector<std::string> test_name;
			//std::cout << input_info.name_txt << std::endl;
			//std::ifstream file__(input_info.name_txt + "/all.txt");
			//std::string buf__;
			//while (file__ && getline(file__, buf__))
			//{
			//	test_name.push_back(buf__);
			//}

			//for (size_t n_test = 0; n_test < test_name.size(); n_test++)
			//{

			//	std::vector<float> test_X;

			//	std::cout << "loading" + test_name[n_test] + " Now v(^_^)v" << std::endl;
			//	read_vector(test_X, input_info.dir_t + "/" + test_name[n_test] + ".raw");

			//	Eigen::MatrixXf testX(test_X.size(), 1);
			//	for (size_t i = 0; i < test_X.size(); i++)
			//	{
			//		testX(i, 0) = test_X[i];
			//	}

			//	testX.resize(n_unit[0], test_X.size() / n_unit[0]);
			//	Eigen::MatrixXf testN = testN.Zero(testX.rows(), testX.cols());

			//	std::string dir_o_ = input_info.dir_o + "/autoencoder/noise_" + std::to_string(input_info.type) + "_" + std::to_string(input_info.rate) + "/param" + input_info.param + "/dae/group" + std::to_string(group[cv_loop]) + "/layer" + std::to_string(layer + 1) + "/hidden";
			//	if (!nari::system::directry_is_exist(dir_o_)) nari::system::make_directry(dir_o_);

			//	if (layer == 0)
			//	{
			//		h.hidden_train_data(layer, testX);
			//	}
			//	else
			//	{
			//		for (size_t l = 0; l < layer + 1; l++)
			//		{
			//			h.hidden_train_data(l, testX);
			//		}
			//	}
			//	write_raw_and_txt(testX, dir_o_ + "/" + test_name[n_test]);


			//}

			////// 中間層の特徴量を全データ分保存ここまで //////////////





		}


		std::cout << "/////////////// FINETUNING START ///////////////" << std::endl;
		dir_o = input_info.dir_o + "/autoencoder/noise_" + std::to_string(input_info.type) + "_" + std::to_string(input_info.rate) + "/param" + input_info.param + "/sdae/group" + std::to_string(group[cv_loop]);
		if (!nari::system::directry_is_exist(dir_o)) nari::system::make_directry(dir_o);

		std::vector<SDAE> sdae;
		for (int i = 0; i < input_info.n_layer; i++)
		{
			Eigen::MatrixXf Weight;
			Eigen::MatrixXf bhidden;
			Eigen::MatrixXf bout;
			h.return_param(Weight, bhidden, bout, i);
			SDAE layer(Weight, bhidden);
			sdae.push_back(layer);
		}
		for (int i = input_info.n_layer - 1; i >= 0; i--)
		{
			Eigen::MatrixXf Weight;
			Eigen::MatrixXf bhidden;
			Eigen::MatrixXf bout;
			h.return_param(Weight, bhidden, bout, i);
			SDAE layer(Weight.transpose(), bout);
			sdae.push_back(layer);
		}


		fine_tuning(sdae, trainN, trainX, validN, validX, (float)input_info.alpha, input_info.epoch, input_info.batch_size, dir_o);

		std::cout << "/////////////// TEST START ///////////////" << std::endl;

		std::vector<std::string> test_name;
		std::cout << input_info.name_txt << std::endl;


		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		std::ifstream file__(input_info.name_txt + "/all.txt");
		std::string buf__;
		while (file__ && getline(file__, buf__))
		{
			test_name.push_back(buf__);
		}

		for (size_t n_test = 0; n_test < test_name.size(); n_test++)
		{

			std::vector<float> test_X;

			std::cout << "loading" + test_name[n_test] + " Now v(^_^)v" << std::endl;
			read_vector(test_X, input_info.dir_t + "/" + test_name[n_test] + ".raw");

			Eigen::MatrixXf testX(test_X.size(), 1);
			for (size_t i = 0; i < test_X.size(); i++)
			{
				testX(i, 0) = test_X[i];
			}

			testX.resize(n_unit[0], test_X.size() / n_unit[0]);

			//add_noise(testX, testN, input_info.type, input_info.rate);

			dir_o = input_info.dir_o + "/autoencoder/noise_" + std::to_string(input_info.type) + "_" + std::to_string(input_info.rate) + "/param" + input_info.param + "/sdae/group" + std::to_string(group[cv_loop]);
			if (!nari::system::directry_is_exist(dir_o)) nari::system::make_directry(dir_o);
			test(sdae, testX, dir_o, test_name[n_test]);

		}
	}

}