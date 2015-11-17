#ifndef __CNN_INFO__
#define __CNN_INFO__
#include <naricommon.h>
#include <nariinfocontroller.h>
#include <narifile.h>
#include <string>

struct dcnn_info
{
	std::string param;	    //パラメータ番号
	std::string dir_o;	    //出力フォルダ
	std::string dir_i;	    //学習入力フォルダ
	std::string dir_a;	    //正解フォルダ
	std::string dir_t;	    //テスト入力フォルダ
	std::string dir_p;	    //パラメータ格納フォルダ
	std::string name_txt;	    //症例名テキスト
	std::string n_map;          //各層のmap数
	std::string process;	    //process(convolution or pooling)
	std::string w_size;	    //Filter size
	std::string pool_size;	    //Pooling size
	std::string activation;     //activation function
	int error_funcion;          //誤差関数(2乗誤差 or クロスエントロピー)
	float alpha;                //学習率
	int epoch;                  //epoch
	int dim;                    //Dimension (2 or 3)
	int n_layer;		    //Number of All layer
	int in_size;		    //Input image size
	int out_size;		    //Output size
	int CV_group;               //CVグループ数
	int batch_size;		    //バッチサイズ
	int input_map;              //入力マップ数
	int thread;                 //スレッド数
	float moment;               //モーメンタム係数
	float lamda;                //正則化係数(L2)
	int validation_num;         //何回おきにvalidationするか
	int ini_switch;		    //重みの初期値スイッチ
	int dropout;		    //dropout


	// テキストから入力情報を書き込み
	inline void input(const std::string &path)
	{
		nari::infocontroller info;
		info.load(path);
		param = nari::file::add_delim(info.get_as_str("param"));		  // parameter number
		dir_o = nari::file::add_delim(info.get_as_str("dir_o"));		  // 出力フォルダ
		dir_i = info.get_as_str("dir_i");					  // 入力フォルダ
		dir_a = info.get_as_str("dir_a");					  // 正解フォルダ
		dir_t = info.get_as_str("dir_t");					  // テストフォルダ
		dir_p = info.get_as_str("dir_p");					  // パラメータ格納フォルダ
		CV_group = info.get_as_int("CV_group");					  // 学習データディレクトリ
		name_txt = info.get_as_str("name_txt");					  // テスト症例名テキスト
		ini_switch = info.get_as_int("ini_switch");				  // 重み初期化スイッチ
 
		dim = info.get_as_int("dim");                                             // Dimension
		n_layer = info.get_as_int("n_layer");					  // Number of All layer
		in_size = info.get_as_int("in_size");					  // Input size
		out_size = info.get_as_int("out_size");					  // Output size
		process = nari::file::add_delim(info.get_as_str("process"));		  // process(convolution or pooling or full connect)
		activation = nari::file::add_delim(info.get_as_str("activation"));        //activation function
		n_map = nari::file::add_delim(info.get_as_str("n_map"));		  // 各層のmap数
		w_size = nari::file::add_delim(info.get_as_str("w_size"));		  // Filter size
		pool_size = nari::file::add_delim(info.get_as_str("pool_size"));	  // Pooling size
		epoch = info.get_as_int("epoch");                                         // epoch
		alpha = (float)info.get_as_double("alpha");                               // 学習率
		batch_size = info.get_as_int("batch_size");				  // バッチサイズ
		input_map = info.get_as_int("input_map");                                 // 入力マップ数
		thread = info.get_as_int("thread");                                       // スレッド数
		moment = (float)info.get_as_double("moment");                             // モーメンタム係数
		lamda = (float)info.get_as_double("lamda");                               // 正則化係数(L2)
		error_funcion = info.get_as_int("error_function");                        // 誤差関数(2乗誤差 or クロスエントロピー)
		validation_num = info.get_as_int("validation");                           // 何回おきにvalidationするか
		dropout = info.get_as_int("dropout");					  // dropout


		info.output();
	}
};


#endif