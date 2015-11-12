#ifndef __SDAE_INFO__
#define __SDAE_INFO__
#include "naricommon.h"
#include "nariinfocontroller.h"
#include "narifile.h"
#include <string>

struct sdae_info
{
	std::string param;			// �p�����[�^�ԍ�
	std::string dir_o;			// �o�̓t�H���_
	std::string dir_i;			// ���̓t�H���_
	std::string dir_t;			// �e�X�g�t�H���_
	std::string dir_p;		    // �p�����[�^�i�[�t�H���_
	std::string name_txt;		// �Ǘᖼ�e�L�X�g
	double alpha;			    // �w�K��
	int epoch;		            // epoch
	std::string n_unit;         // �e�w�̃��j�b�g��
	int ini_switch;             // �p�����[�^�������X�C�b�`
	int type;					// �m�C�Y�^�C�v
	int rate;					// �m�C�Y����
	int CV_group;               // CV�O���[�v��
	int n_layer;                // �w��
	int batch_size;				// �o�b�`�T�C�Y
	int num_thread;				// �X���b�h��
	double lambda;              // �������p�����[�^


	// �e�L�X�g������͏�����������
	inline void input(const std::string &path)
	{
		nari::infocontroller info;
		info.load(path);
		param = nari::file::add_delim(info.get_as_str("param"));		          // �p�����[�^�ԍ�
		dir_o = nari::file::add_delim(info.get_as_str("dir_o"));		          // �o�̓t�H���_
		dir_i = info.get_as_str("dir_i");									      // ���̓t�H���_
		dir_t = info.get_as_str("dir_t");									      // �e�X�g�t�H���_
		ini_switch = info.get_as_int("ini_switch");							      // �������X�C�b�`
		type = info.get_as_int("type");										      // �m�C�Y�^�C�v
		rate = info.get_as_int("rate");										      // �m�C�Y����
		CV_group = info.get_as_int("CV_group");						     	      // �w�K�f�[�^�f�B���N�g��
		dir_p = nari::file::add_delim(info.get_as_str("dir_p"));		          // �p�����[�^�i�[�t�H���_
		name_txt = info.get_as_str("name_txt");					                  // �e�X�g�Ǘᖼ�e�L�X�g
		n_layer = info.get_as_int("n_layer");									  // ���̓f�[�^����
		alpha = info.get_as_double("alpha");					                  // �w�K��
		epoch = info.get_as_int("epoch");									      // 1�w�ڂ�epoch
		batch_size = info.get_as_int("batch_size");								  // �o�b�`�T�C�Y
		lambda = info.get_as_double("lambda");									  // �������p�����[�^
		n_unit = nari::file::add_delim(info.get_as_str("n_unit"));		    	  // �e�w�̃��j�b�g��
		num_thread = info.get_as_int("num_thread");

		info.output();
	}
};


#endif