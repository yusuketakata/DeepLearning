#pragma once

#include "Header.h"
#include "tktlib\eigen-eigen-10219c95fe65\Eigen\Core"
#include "tktlib\raw_io.h"

class AE
{

private:
public:
	Eigen::MatrixXf W;			//weight
	Eigen::MatrixXf bhid;		//hidden bias
	Eigen::MatrixXf bout;		//output bias
	Eigen::MatrixXf output;		// encode output

	int n_visible;
	int n_hidden;
	int batch_size;

	AE(size_t, size_t, size_t);

	
	Eigen::MatrixXf encoder(const Eigen::MatrixXf&);
	Eigen::MatrixXf decoder(const Eigen::MatrixXf&);
	void param_update(const Eigen::MatrixXf&, const Eigen::MatrixXf&, float, double&);
	void valid_test(const Eigen::MatrixXf&, const Eigen::MatrixXf&, float, double&);
};

AE::AE(size_t n_visible, size_t n_hidden, size_t batch_size)
{
	///////////////////////////////////////////////////////////////
	// �p�����[�^�̏����l���^�����Ȃ��ꍇ�̃R���X�g���N�^      //
	// �����͓��̓��j�b�g�̐��C�B�ꃆ�j�b�g�̐�                  //
	// W : -0.01 ~ 0.01 �̈�l�����ɂ���ď�����                 //
	// b :  0�ŏ�����                                            //
	// c :  0�ŏ�����                                            //
	///////////////////////////////////////////////////////////////	

	W = Eigen::MatrixXf::Ones(n_hidden, n_visible);
	bhid = Eigen::MatrixXf::Zero(n_hidden, 1);
	bout = Eigen::MatrixXf::Zero(n_visible, 1);

	int fan_in;
	fan_in = (int)(n_hidden + n_visible + 1);

	dist = std::uniform_real_distribution<float>(-(float)sqrt(6.0 / (float)fan_in), (float)sqrt(6.0 / (float)fan_in));

	for (int j = 0; j < W.cols(); j++){
		for (int i = 0; i < W.rows(); i++){
			W(i, j) = dist(engine);
		}
	}
}



Eigen::MatrixXf AE::encoder(const Eigen::MatrixXf& visible)
{
	///////////////////////////////////////////////////////////////
	// encode ���s���֐�									     //
	// visible : ���̓T���v��                                    //
	///////////////////////////////////////////////////////////////	

	Eigen::MatrixXf Hidden;
	Eigen::MatrixXf I = I.Ones(1, visible.cols());
	Eigen::MatrixXf B = bhid * I;
	Eigen::MatrixXf X = ((W * visible).array() + B.array()).matrix();

	Hidden = 1 / (1 + exp((-X).array()));

	return Hidden;

}

Eigen::MatrixXf AE::decoder(const Eigen::MatrixXf& hidden)
{
	///////////////////////////////////////////////////////////////
	// decode ���s���֐�									     //
	// visible : ���̓T���v��                                    //
	///////////////////////////////////////////////////////////////	

	Eigen::MatrixXf Output;
	Eigen::MatrixXf I = I.Ones(1, hidden.cols());
	Eigen::MatrixXf B = bout * I;
	Eigen::MatrixXf X = ((W.transpose() * hidden).array() + B.array()).matrix();

	Output = 1 / (1 + exp((-X).array()));

	return Output;

}

void AE::param_update(const Eigen::MatrixXf& input, const Eigen::MatrixXf& answer, float lr, double& error)
{
	Eigen::MatrixXf dW = dW.Zero(W.rows(), W.cols());
	Eigen::MatrixXf dbhid = dbhid.Zero(bhid.rows(), 1);
	Eigen::MatrixXf dbout = dbout.Zero(bout.rows(), 1);

	Eigen::MatrixXf Hidden;
	Eigen::MatrixXf Output;

	Hidden = encoder(input);
	Output = decoder(Hidden);

	//error = -(answer.array() * (Output.array() + FLT_MIN).log() + (1. - answer.array()) * (1. - Output.array() + FLT_MIN).log()).sum();

	// ���덷��error���v�Z // 
	error = ((answer.array() - Output.array()).array() * (answer.array() - Output.array()).array()).sum();


	//*** calculate gradient ***//

	// ���z���v�Z����ۂɓ�d�v�Z��h�����߂̕ϐ� //
	Eigen::MatrixXf Diff = answer - Output;
	Eigen::MatrixXf WW = W * Diff;
	Eigen::MatrixXf H = Hidden.array() * (1 - Hidden.array());
	Eigen::MatrixXf MM = WW.array() * H.array();

	dW = MM * input.transpose() + (Diff * Hidden.transpose()).transpose();
	dW = (lr / (float)answer.cols()) * dW;
	dbhid = MM.rowwise().sum();
	dbhid = (lr / (float)answer.cols()) * dbhid;


	dbout = Diff.rowwise().sum();
	dbout = (lr / (float)answer.cols()) * dbout;

	////*** param update ***//

	W += dW;
	bhid += dbhid;
	bout += dbout;

}

void AE::valid_test(const Eigen::MatrixXf& input, const Eigen::MatrixXf& answer, float lr, double& error)
{
	Eigen::MatrixXf dW = dW.Zero(W.rows(), W.cols());
	Eigen::MatrixXf dbhid = dbhid.Zero(bhid.rows(), 1);
	Eigen::MatrixXf dbout = dbout.Zero(bout.rows(), 1);

	Eigen::MatrixXf Hidden;
	Eigen::MatrixXf Output;

	Hidden = encoder(input);
	Output = decoder(Hidden);

	// �N���X�G���g���s�[
	//error = -(answer.array() * (Output.array() + FLT_MIN).log() + (1. - answer.array()) * (1. - Output.array() + FLT_MIN).log()).sum();
	// ���덷
	error = ((answer.array() - Output.array()).array() * (answer.array() - Output.array()).array()).sum() / input.cols();
}