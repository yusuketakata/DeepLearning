#pragma once
#include"Header.h"
#include "Utils.h"
#include "mlp.h"


class CNN
{
private:
public:
	Eigen::MatrixXf W_init; //Initial filter
	Eigen::MatrixXf W_stock; //�X�p�[�X�s��̏d�ݏC���p
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
	CNN(std::string, int, int, int, int, int, char, int, int, std::string, int, int, int); //2�w�ڈȍ~Convolution

	void convn(Eigen::MatrixXf&, Eigen::MatrixXf&, std::string);
	void pooling(Eigen::MatrixXf&, Eigen::MatrixXf&, std::string, int);

};


// 1�w�ڂ̃R���X�g���N�^
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
				throw "( �E_�T�E)���̃t�B���^�T�C�Y�͂���������!!( �E_�T�E)";
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
					//W_init(j, i) = 0.01;
					//if (j==0)
					//{
					//	W_init(j,i) = 0.01;
					//}
					//else if (j==1)
					//{
					//	W_init(j,i) = 0.02;
					//}
					//}else if (j==2)
					//{
					//	W_init(j,i) = 0.03;
					//}else if (j==3)
					//{
					//	W_init(j,i) = 0.04;
					//}
				}
			}
		}
		else if (ini_switch == 1){
			std::vector<float> W_dummy;
			read_vector(W_dummy, dir_p + "/group" + std::to_string(cv_loop + 1) + "/hidden_W" + std::to_string(layer) + ".raw");
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

		int count = 0; //�S�������J�E���g
		int cout_init = 0;
		int first_unit = 0; //��ݍ��ݍŏ��̉�f
		if (dim == 3)
		{
			for (int k = 0; k < n_map; k++){ //�����}�b�v���[�v
				first_unit = 0;
				for (int pre_map_loop = 0; pre_map_loop < in_map; pre_map_loop++){ //���͑w�̓����}�b�v���[�v
					first_unit += (pre_map_loop * in_dim - first_unit);
					for (int j = 0; j < ((int)(W.rows()) / n_map); j++){ //�o�͑w���j�b�g���[�v(1�̓����}�b�v��)
						int count_w = 0; //�t�B���^�T�C�Y�J�E���g
						if (j != 0 && j % out_size == 0){
							first_unit += (w_size - 1);
						}
						if (j != 0 && j % (out_size * out_size) == 0){
							first_unit += in_size * (w_size - 1);
						}
						for (int i = first_unit; i < (int)W.cols(); i++){ //���͑w���j�b�g���[�v
							if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
								i += (in_size - w_size);
							}
							if (count_w != 0 && count_w % (w_size * w_size) == 0){
								i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
							}
							//std::cout << cout_init << "," << count_w << std::endl;
							//std::cout << cout_init;
							t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, W_init(cout_init, count_w));
							count++;
							count_w++;
							if (count_w == w_size * w_size * w_size)	break; //�t�B���^�T�C�Y�����������I��
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
			for (int k = 0; k < n_map; k++){ //�����}�b�v���[�v
				first_unit = 0;
				for (int pre_map_loop = 0; pre_map_loop < in_map; pre_map_loop++){ ///���͑w�̓����}�b�v���[�v
					first_unit += (pre_map_loop * in_dim - first_unit);
					for (int j = 0; j < ((int)(W.rows()) / n_map); j++){ //�o�͑w���j�b�g���[�v(1�̓����}�b�v��)
						int count_w = 0; //�t�B���^�T�C�Y�J�E���g
						if (j != 0 && j % out_size == 0){
							first_unit += (w_size - 1);
						}
						for (int i = first_unit; i < (int)W.cols(); i++){ //���͑w���j�b�g���[�v
							if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
								i += (in_size - w_size);
							}
							//std::cout << cout_init << "," << count_w << std::endl;
							//std::cout << cout_init;
							t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, W_init(cout_init, count_w));
							count++;
							count_w++;
							if (count_w == w_dim)	break; //�t�B���^�T�C�Y�����������I��
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

		//�v�[�����O�p�̉��z�I�ȃX�p�[�X�s��
		W.resize(in_dim * n_map / (w_dim), in_dim * n_map);
		std::cout << "layer_p = " << W.rows() << ", " << W.cols() << std::endl; /////////////

		//�X�p�[�X�s���non-zero������
		w_num = w_dim * (int)W.rows();
		t.resize(w_num);


		/**** �������牼�z�I�ȏd�ݍs��쐬 ****/
		int count = 0; //�S�������J�E���g
		int first_unit = 0; //��ݍ��ݍŏ��̉�f

		for (int j = 0; j < (int)(W.rows()); j++){ //�o�͑w���j�b�g���[�v(1�̓����}�b�v��)

			int count_w = 0; //�t�B���^�T�C�Y�J�E���g
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

			for (int i = first_unit; i < (int)W.cols(); i++){ //���͑w���j�b�g���[�v
				if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
					i += (in_size - w_size);
				}
				if (count_w != 0 && count_w % (w_size * w_size) == 0){
					i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
				}
				//std::cout << i << "," << j << std::endl;
				t[count] = Eigen::Triplet<float>(j, i, (float)1 / w_dim);
				//std::cout << t.size() << std::endl;/////
				//t[count] = Triplet<double>( j , i, 1 );
				count++;
				count_w++;
				if (count_w == w_size * w_size * w_size)	break; //�t�B���^�T�C�Y�����������I��
			}
			if (count == w_num) break;
			first_unit++;
		}


		W.setFromTriplets(t.begin(), t.end());


	}


}

// 2�w�ڈȍ~�̃R���X�g���N�^
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
				throw "( �E_�T�E)���̃t�B���^�T�C�Y�͂���������!!( �E_�T�E)";
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
		/* �ݖ{�C�� : ���͂̓����}�b�v������������ꍇ�ɑΉ�  */
		/******************************************************/
		W_init = W_init.Zero(n_map * pre_map, w_dim);

		int fan_in;
		fan_in = n_map * w_size * w_size;
		dist = std::uniform_real_distribution<float>(-(float)sqrt(3.0 / (float)fan_in), (float)sqrt(3.0 / (float)fan_in));
		if (ini_switch == 0){
			for (int j = 0; j < W_init.rows(); j++){
				for (int i = 0; i < W_init.cols(); i++){
					W_init(j, i) = dist(engine);
					//W_init(j, i) = 0.01;
					//if (j==0)
					//{
					//	W_init(j,i) = 0.03;
					//}
					//else if (j==1)
					//{
					//	W_init(j,i) = 0.04;
					//}else if (j==2)
					//{
					//	W_init(j,i) = 0.05;
					//}else if (j==3)
					//{
					//	W_init(j,i) = 0.06;
					//}
				}
			}
		}
		else if (ini_switch == 1){

			std::vector<float> W_dummy;
			read_vector(W_dummy, dir_p + "/group" + std::to_string(cv_loop + 1) + "/hidden_W" + std::to_string(layer) + ".raw");

			for (int j = 0; j < W_init.cols(); j++){
				for (int i = 0; i < W_init.rows(); i++){
					W_init(i, j) = W_dummy[j * W_init.rows() + i];

				}
			}
			std::cout << "param get" << std::endl;

		}
		//b = Eigen::VectorXf::Zero(n_map);
		W_stock = W_init;

		W.resize(out_dim * n_map, pre_out);
		std::cout << "layer_c = " << W.rows() << ", " << W.cols() << std::endl; ////////////////
		w_num = w_dim *  (int)W.rows() * pre_map;

		t.resize(w_num);

		//std::cout << in_size << "," << in_dim << std::endl;

		int count = 0; //�S�������J�E���g
		int cout_init = 0;
		int first_unit = 0; //��ݍ��ݍŏ��̉�f
		if (dim == 3)
		{
			for (int k = 0; k < n_map; k++){ //�����}�b�v���[�v
				first_unit = 0;
				for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //�O�w�̓����}�b�v���[�v
					first_unit += (pre_map_loop * in_dim - first_unit);
					for (int j = 0; j < ((int)(W.rows()) / n_map); j++){ //�o�͑w���j�b�g���[�v(1�̓����}�b�v��)
						int count_w = 0; //�t�B���^�T�C�Y�J�E���g
						if (j != 0 && j % out_size == 0){
							first_unit += (w_size - 1);
						}
						if (j != 0 && j % (out_size * out_size) == 0){
							first_unit += in_size * (w_size - 1);
						}
						for (int i = first_unit; i < (int)W.cols(); i++){ //���͑w���j�b�g���[�v
							if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
								i += (in_size - w_size);
							}
							if (count_w != 0 && count_w % (w_size * w_size) == 0){
								i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
							}
							//std::cout << cout_init << "," << count_w << std::endl;
							//std::cout << cout_init;
							t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, W_init(cout_init, count_w));
							count++;
							count_w++;
							if (count_w == w_size * w_size * w_size)	break; //�t�B���^�T�C�Y�����������I��
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
			for (int k = 0; k < n_map; k++){ //�����}�b�v���[�v
				first_unit = 0;
				for (int pre_map_loop = 0; pre_map_loop < pre_map; pre_map_loop++){ //�O�w�̓����}�b�v���[�v
					first_unit += (pre_map_loop * in_dim - first_unit);
					for (int j = 0; j < ((int)(W.rows()) / n_map); j++){ //�o�͑w���j�b�g���[�v(1�̓����}�b�v��)
						int count_w = 0; //�t�B���^�T�C�Y�J�E���g
						if (j != 0 && j % out_size == 0){
							first_unit += (w_size - 1);
						}
						for (int i = first_unit; i < (int)W.cols(); i++){ //���͑w���j�b�g���[�v
							if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
								i += (in_size - w_size);
							}
							//std::cout << cout_init << "," << count_w << std::endl;
							//std::cout << cout_init;
							t[count] = Eigen::Triplet<float>(j + (k * out_dim), i, W_init(cout_init, count_w));
							count++;
							count_w++;
							if (count_w == w_dim)	break; //�t�B���^�T�C�Y�����������I��
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
		/* �ݖ{�C�� �����܂� */
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
				throw "( �E_�T�E)����Pooling�T�C�Y�ł͊���؂�Ȃ���!!( �E_�T�E)";
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

		//�v�[�����O�p�̉��z�I�ȃX�p�[�X�s��
		W.resize(pre_out / (w_dim), pre_out);
		std::cout << "layer_p = " << W.rows() << ", " << W.cols() << std::endl; ////////////////////////////

		//�X�p�[�X�s���non-zero������
		w_num = w_dim * (int)W.rows();
		t.resize(w_num);


		/**** �������牼�z�I�ȏd�ݍs��쐬 ****/
		int count = 0; //�S�������J�E���g
		int first_unit = 0; //��ݍ��ݍŏ��̉�f

		if (dim == 3)
		{
			for (int j = 0; j < (int)(W.rows()); j++){ //�o�͑w���j�b�g���[�v(1�̓����}�b�v��)

				int count_w = 0; //�t�B���^�T�C�Y�J�E���g
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

				for (int i = first_unit; i < (int)W.cols(); i++){ //���͑w���j�b�g���[�v
					if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
						i += (in_size - w_size);
					}
					if (count_w != 0 && count_w % (w_size * w_size) == 0){
						i += ((in_size * (in_size - w_size)) + (in_size - w_size - 1) + 1);
					}
					t[count] = Eigen::Triplet<float>(j, i, (float)1 / w_dim);
					count++;
					count_w++;
					if (count_w == w_size * w_size * w_size)	break; //�t�B���^�T�C�Y�����������I��
				}
				if (count == w_num) break;
				first_unit++;
			}
		}
		else if (dim == 2)
		{
			for (int j = 0; j < (int)(W.rows()); j++){ //�o�͑w���j�b�g���[�v(1�̓����}�b�v��)

				int count_w = 0; //�t�B���^�T�C�Y�J�E���g
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

				for (int i = first_unit; i < (int)W.cols(); i++){ //���͑w���j�b�g���[�v
					if (count_w != 0 && count_w % w_size == 0 && count_w % (w_size * w_size) != 0){
						i += (in_size - w_size);
					}
					t[count] = Eigen::Triplet<float>(j, i, (float)1 / w_dim);
					count++;
					count_w++;
					if (count_w == w_dim)	break; //�t�B���^�T�C�Y�����������I��
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
			int count = 0; //�S�������J�E���g
			int first_unit = 0; //��ݍ��ݍŏ��̉�f

			//�X�p�[�X�s���non-zero������
			w_num = (int)W.rows();
			t.resize(w_num);
			if (dim == 3)
			{
				for (int j = 0; j < (int)(W.rows()); j++){ //�o�͑w���j�b�g���[�v(1�̓����}�b�v��)
					float w_max = FLT_MIN;
					int index;
					int count_w = 0; //�t�B���^�T�C�Y�J�E���g
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

					for (int i = first_unit; i < (int)W.cols(); i++){ //���͑w���j�b�g���[�v
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
						if (count_w == w_dim)	break; //�t�B���^�T�C�Y�����������I��
					}
					t[count] = Eigen::Triplet<float>(j, index, 1.0);
					count++;
					if (count == w_num) break;
					first_unit++;
				}
			}
			else if (dim == 2)
			{
				for (int j = 0; j < (int)(W.rows()); j++){ //�o�͑w���j�b�g���[�v(1�̓����}�b�v��)
					float w_max = FLT_MIN;
					int index;
					int count_w = 0; //�t�B���^�T�C�Y�J�E���g
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

					for (int i = first_unit; i < (int)W.cols(); i++){ //���͑w���j�b�g���[�v
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
						if (count_w == w_dim)	break; //�t�B���^�T�C�Y�����������I��
					}
					t[count] = Eigen::Triplet<float>(j, index, 1.0);
					count++;
					if (count == w_num) break;
					first_unit++;
				}
			}

			W.setFromTriplets(t.begin(), t.end());
			//std::cout << "W  " << W << std::endl;
			stock_output.block(0, batch, stock_output.rows(), 1) = W * INPUT.col(batch);

		}
		OUTPUT = stock_output;
	}

	if (pool == "Lp"){

	}

}