#ifndef __SDAE_INFO__
#define __SDAE_INFO__
#include "naricommon.h"
#include "nariinfocontroller.h"
#include "narifile.h"
#include <string>

struct sdae_info
{
	std::string param;			// パラメータ番号
	std::string dir_o;			// 出力フォルダ
	std::string dir_i;			// 入力フォルダ
	std::string dir_t;			// テストフォルダ
	std::string dir_p;		    // パラメータ格納フォルダ
	std::string name_txt;		// 症例名テキスト
	double alpha;			    // 学習率
	int epoch;		            // epoch
	std::string n_unit;         // 各層のユニット数
	int ini_switch;             // パラメータ初期化スイッチ
	int type;					// ノイズタイプ
	int rate;					// ノイズ割合
	int CV_group;               // CVグループ数
	int n_layer;                // 層数
	int batch_size;				// バッチサイズ
	int num_thread;				// スレッド数
	double lambda;              // 正則化パラメータ


	// テキストから入力情報を書き込み
	inline void input(const std::string &path)
	{
		nari::infocontroller info;
		info.load(path);
		param = nari::file::add_delim(info.get_as_str("param"));		          // パラメータ番号
		dir_o = nari::file::add_delim(info.get_as_str("dir_o"));		          // 出力フォルダ
		dir_i = info.get_as_str("dir_i");									      // 入力フォルダ
		dir_t = info.get_as_str("dir_t");									      // テストフォルダ
		ini_switch = info.get_as_int("ini_switch");							      // 初期化スイッチ
		type = info.get_as_int("type");										      // ノイズタイプ
		rate = info.get_as_int("rate");										      // ノイズ割合
		CV_group = info.get_as_int("CV_group");						     	      // 学習データディレクトリ
		dir_p = nari::file::add_delim(info.get_as_str("dir_p"));		          // パラメータ格納フォルダ
		name_txt = info.get_as_str("name_txt");					                  // テスト症例名テキスト
		n_layer = info.get_as_int("n_layer");									  // 入力データ次元
		alpha = info.get_as_double("alpha");					                  // 学習率
		epoch = info.get_as_int("epoch");									      // 1層目のepoch
		batch_size = info.get_as_int("batch_size");								  // バッチサイズ
		lambda = info.get_as_double("lambda");									  // 正則化パラメータ
		n_unit = nari::file::add_delim(info.get_as_str("n_unit"));		    	  // 各層のユニット数
		num_thread = info.get_as_int("num_thread");

		info.output();
	}
};


#endif