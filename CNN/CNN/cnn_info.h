#ifndef __DCNN_INFO__
#define __DCNN_INFO__
#include <naricommon.h>
#include <nariinfocontroller.h>
#include <narifile.h>
#include <string>

struct dcnn_info
{
	std::string param;			//�p�����[�^�ԍ�
	std::string dir_o;			//�o�̓t�H���_
	std::string dir_i;			//�w�K���̓t�H���_
	std::string dir_a;			//�����t�H���_
	std::string dir_t;			//�e�X�g���̓t�H���_
	std::string dir_p;			//�p�����[�^�i�[�t�H���_
	std::string name_txt;		//�Ǘᖼ�e�L�X�g
	std::string n_map;          //�e�w��map��
	std::string process;	    //process(convolution or pooling)
	std::string w_size;		    //Filter size
	std::string pool_size;		//Pooling size
	std::string activation;     //activation function
	int error_funcion;          //�덷�֐�(2��덷 or �N���X�G���g���s�[)
	float alpha;                //�w�K��
	int epoch;                  //epoch
	int dim;                    //Dimension (2 or 3)
	int n_layer;				//Number of All layer
	int in_size;				//Input image size
	int out_size;				//Output size
	int CV_group;               //CV�O���[�v��
	int batch_size;				//�o�b�`�T�C�Y
	int input_map;              //���̓}�b�v��
	int thread;                 //�X���b�h��
	float moment;               //���[�����^���W��
	float lamda;                //�������W��(L2)
	int validation_num;         //���񂨂���validation���邩
	int ini_switch;		        //�d�݂̏����l�X�C�b�`


	// �e�L�X�g������͏�����������
	inline void input(const std::string &path)
	{
		nari::infocontroller info;
		info.load(path);
		param = nari::file::add_delim(info.get_as_str("param"));		          // parameter number
		dir_o = nari::file::add_delim(info.get_as_str("dir_o"));		          // �o�̓t�H���_
		dir_i = info.get_as_str("dir_i");									      // ���̓t�H���_
		dir_a = info.get_as_str("dir_a");									      // �����t�H���_
		dir_t = info.get_as_str("dir_t");									      // �e�X�g�t�H���_
		dir_p = info.get_as_str("dir_p");									      // �p�����[�^�i�[�t�H���_
		CV_group = info.get_as_int("CV_group");						     	      // �w�K�f�[�^�f�B���N�g��
		name_txt = info.get_as_str("name_txt");					                  // �e�X�g�Ǘᖼ�e�L�X�g
		ini_switch = info.get_as_int("ini_switch");								  // �d�ݏ������X�C�b�`

		dim = info.get_as_int("dim");                                             // Dimension
		n_layer = info.get_as_int("n_layer");									  // Number of All layer
		in_size = info.get_as_int("in_size");									  // Input size
		out_size = info.get_as_int("out_size");									  // Output size
		process = nari::file::add_delim(info.get_as_str("process"));			  // process(convolution or pooling or full connect)
		activation = nari::file::add_delim(info.get_as_str("activation"));        // activation function
		n_map = nari::file::add_delim(info.get_as_str("n_map"));				  // �e�w��map��
		w_size = nari::file::add_delim(info.get_as_str("w_size"));		          // Filter size
		pool_size = nari::file::add_delim(info.get_as_str("pool_size"));		  // Pooling size
		epoch = info.get_as_int("epoch");                                         // epoch
		alpha = (float)info.get_as_double("alpha");                               // �w�K��
		batch_size = info.get_as_int("batch_size");								  // �o�b�`�T�C�Y
		input_map = info.get_as_int("input_map");                                 // ���̓}�b�v��
		thread = info.get_as_int("thread");                                       // �X���b�h��
		moment = (float)info.get_as_double("moment");                             // ���[�����^���W��
		lamda = (float)info.get_as_double("lamda");                               // �������W��(L2)
		error_funcion = info.get_as_int("error_function");                        // �덷�֐�(2��덷 or �N���X�G���g���s�[)
		validation_num = info.get_as_int("validation");                           // ���񂨂���validation���邩

		info.output();
	}
};


#endif