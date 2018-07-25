#pragma once
#include "functions.h"
#include <stdio.h>
#include <stdlib.h>



//Optimizer Start
class SGD {
public:
	float self_lr;
	SGD() {

	}
	SGD(float lr) {
		self_lr = lr;
	}
	void update(float *params, float *grads, int Size)
	{
		int i;
		for (i = 0; i < Size; i++) params[i] -= self_lr * grads[i];
	}
	~SGD()
	{

	}
};
//Optimizer Start

class Affine {
public:
	float *self_weight, *self_bias, *self_in;
	float *dW, *db;
	int self_inNode, self_outNode;
	float *temp;
	Affine()
	{

	}
	Affine(float *weight, float *bias, int inNode, int outNode)
	{
		self_weight = weight;
		self_bias = bias;
		self_inNode = inNode;
		self_outNode = outNode;
		self_in = new float[inNode];
		dW = new float[inNode*outNode];
		db = new float[outNode];
	}
	float* forward(float *in)
	{
		int i, j;
		for (i = 0; i < self_inNode; i++)
			self_in[i] = in[i];
		delete[] in;
		temp = new float[self_outNode];
		for (i = 0; i < self_outNode; i++) {
			temp[i] = 0;
			for (j = 0; j < self_inNode; j++) {
				temp[i] += self_in[j] * self_weight[j*self_outNode + i];
			}
			temp[i] += self_bias[i];
		}
		return temp;
	}
	float* backward(float *grad)
	{
		float *temp = new float[self_inNode];
		for (int i = 0; i < self_inNode; i++) temp[i] = 0;
		for (int i = 0; i < self_inNode; i++)
		{
			for (int j = 0; j < self_outNode; j++)
			{
				temp[i] += grad[j] * self_weight[i*self_outNode + j];
				dW[i*self_outNode + j] = self_in[i] * grad[j];
				db[j] = grad[j];
			}
		}
		delete[] grad;

		return temp;
	}
	~Affine()
	{
		
	}
};

class SoftmaxwithLoss
{
public:
	int Self_Size;
	float *self_pre;
	int *self_label;
	float self_loss;
	SoftmaxwithLoss()
	{

	}
	SoftmaxwithLoss(int Size)
	{
		Self_Size = Size;
		self_pre = new float[Self_Size];
		self_label = new int[Self_Size];
	}
	float forward(float *out, unsigned char *label)
	{
		for (int i = 0; i < Self_Size; i++) self_label[i] = label[i];
		softmax(out, Self_Size);
		for (int i = 0; i < Self_Size; i++) self_pre[i] = out[i];
		self_loss = cross_entropy_error(out, Self_Size, label);
		return self_loss;
	}
	float* backward(float Loss)
	{
		float *grad = new float[Self_Size];
		for (int i = 0; i < Self_Size; i++)
		{
			grad[i] = (self_pre[i] - self_label[i])*Loss;
		}
		return grad;
	}
	~SoftmaxwithLoss()
	{

	}
};
class Convolution
{
public:
	float *Self_weight, *Self_bias, *Self_in, *dW, *db;
	int Self_stride, Self_pad;
	int Self_FilterSize, Self_FilterNum, Self_FilterChannel;
	int Self_xSize, Self_ySize;
	int out_xSize, out_ySize;
	Convolution()
	{

	}
	Convolution(float *weight, float *bias, int stride, int pad, int FilterSize, int FilterNum, int FilterChannel, int img_xSize, int img_ySize) {
		Self_weight = weight;
		Self_bias = bias;
		dW = new float[FilterSize*FilterSize*FilterNum*FilterChannel];
		db = new float[FilterNum];
		Self_in = new float[(img_xSize + 2 * pad) * (img_ySize + 2 * pad)* FilterChannel];
		Self_FilterSize = FilterSize;
		Self_FilterNum = FilterNum;
		Self_FilterChannel = FilterChannel;
		Self_stride = stride;
		Self_pad = pad;
		Self_xSize = img_xSize;
		Self_ySize = img_ySize;
		out_xSize = 1 + (Self_xSize + 2 * Self_pad - Self_FilterSize) / Self_stride;
		out_ySize = 1 + (Self_ySize + 2 * Self_pad - Self_FilterSize) / Self_stride;
	}
	float* Forward(float *in)
	{
		float *out = new float[Self_FilterNum * out_xSize * out_ySize];
		int i, j, k, l, r, c;
		int image_r, image_c;
		float img_value, fil_value;
		float sum = 0.0;
		for (i = 0; i < Self_ySize + 2 * Self_pad; i++)
		{
			for (j = 0; j < Self_xSize + 2 * Self_pad; j++)
			{
				for (k = 0; k < Self_FilterChannel; k++) 
				{	
					Self_in[k*(Self_xSize + 2 * Self_pad)*(Self_ySize + 2 * Self_pad) + i*(Self_xSize + 2 * Self_pad) + j] = \
						((i < Self_pad) | (i >= Self_ySize + Self_pad) | (j < Self_pad) | (j >= Self_xSize + Self_pad)) ? \
						0 : in[k*Self_xSize*Self_ySize + (i-Self_pad) * Self_xSize + j-Self_pad];
				}
			}
		}
		for (i = 0; i < Self_FilterNum; i++)
		{
			for (j = 0; j < out_ySize; j++)
			{
				for (k = 0; k < out_xSize; k++)
				{
					for (r = 0; r < Self_FilterSize; r++)
						for (c = 0; c < Self_FilterSize; c++)
						{
							sum = 0.0;
							image_r = j*Self_stride + r;
							image_c = k*Self_stride + c;
							for (l = 0; l < Self_FilterChannel; l++)
							{
								img_value = Self_in[l*(Self_ySize + 2 * Self_pad)*(Self_xSize + 2 * Self_pad) + image_r * (Self_xSize + 2 * Self_pad) + image_c];
								fil_value = Self_weight[i*Self_FilterSize*Self_FilterSize*Self_FilterChannel + l * Self_FilterSize*Self_FilterSize + r * Self_FilterSize + c];
								sum += img_value * fil_value;
							}
						} 
					out[i*out_xSize*out_ySize + j * out_xSize + k] = sum + Self_bias[i];
				}
			}
		}
		delete[]in;
		return out;
	}
	float* backward(float *dout)
	{
		int i, j, k, l, r, c;
		for (i = 0; i < Self_FilterNum; i++)
		{
			db[i] = 0;
			for (j = 0; j < out_ySize; j++)
			{
				for (k = 0; k < out_xSize; k++)
				{
					db[i] += dout[i * out_ySize * out_xSize + j * out_xSize + k];
				}
			}
		}
		float *temp_in = new float[Self_FilterChannel*out_xSize*out_ySize*Self_FilterSize*Self_FilterSize];
		for (i = 0; i < out_ySize; i++)
		{
			for (j = 0; j < out_xSize; j++)
			{
				for (k = 0; k < Self_FilterChannel; k++)
				{
					for (r = 0; r < Self_FilterSize; r++)
					{
						for (c = 0; c < Self_FilterSize; c++)
						{
							temp_in[i*out_xSize*Self_FilterChannel*Self_FilterSize*Self_FilterSize + j * Self_FilterChannel*Self_FilterSize*Self_FilterSize + k * Self_FilterSize*Self_FilterSize + r * Self_FilterSize + c] = \
								Self_in[(c + j * Self_stride) + (r + i * Self_stride)*(Self_xSize + 2 * Self_pad) + k * (Self_xSize + 2 * Self_pad)*(Self_ySize + 2 * Self_pad)];
						}
					}
				}
			}
		}

		float *temp_d = new float[out_xSize * out_ySize * Self_FilterNum];
		for (i = 0; i < out_ySize; i++)
		{
			for (j = 0; j < out_xSize; j++)
			{
				for (l = 0; l < Self_FilterNum; l++)
				{
					temp_d[l*out_xSize*out_ySize + i*out_xSize+j] = dout[l*out_xSize*out_ySize + i*out_xSize + j];
				}
			}
		}

		for (i = 0; i<Self_FilterSize*Self_FilterSize*Self_FilterChannel; i++)
			for (j = 0; j < Self_FilterNum; j++)
			{
				dW[j*Self_FilterSize*Self_FilterSize*Self_FilterChannel + i] = 0;
				for (k = 0; k < out_xSize*out_ySize; k++)
					dW[j*Self_FilterSize*Self_FilterSize*Self_FilterChannel + i] += temp_d[j*out_xSize*out_ySize + k] * temp_in[k*Self_FilterChannel*Self_FilterSize*Self_FilterSize+i];
			}

		float *temp_dx = new float[out_xSize*out_ySize*Self_FilterChannel*Self_FilterSize*Self_FilterSize];

		for (i = 0; i < out_xSize*out_ySize; i++)
		{
			for (j = 0; j < Self_FilterChannel*Self_FilterSize*Self_FilterSize; j++)
			{
				temp_dx[i*Self_FilterChannel*Self_FilterSize*Self_FilterSize + j] = 0;
				for (k = 0; k < Self_FilterNum; k++)
					temp_dx[i*Self_FilterChannel*Self_FilterSize*Self_FilterSize + j] += \
						Self_weight[k * Self_FilterChannel*Self_FilterSize*Self_FilterSize+j] * temp_d[k*out_ySize*out_xSize + i];
			}
		}
		float *dx = (float *)calloc((Self_xSize + 2 * Self_pad)*(Self_ySize + 2 * Self_pad)*Self_FilterChannel, sizeof(float));
		for (i = 0; i < out_ySize; i++)
		{
			for (j = 0; j < out_xSize; j++)
			{
				for (k = 0; k < Self_FilterChannel; k++)
				{
					for (l = 0; l < Self_FilterSize; l++)
					{
						for (r = 0; r < Self_FilterSize; r++)
						{
							dx[k * (Self_xSize + 2 * Self_pad)*(Self_ySize + 2 * Self_pad) + (l + i * Self_stride)*(Self_xSize + 2 * Self_pad) + r + j * Self_stride] += \
								temp_dx[(k*Self_FilterSize*Self_FilterSize + l * Self_FilterSize + r) * out_xSize*out_ySize + i * out_xSize + j];
						}
					}
				}
			}
		}
		delete[] temp_d;
		delete[] temp_in;
		delete[] temp_dx;
		delete[] dout;
		return dx;
	}
	~Convolution()
	{
		
	}
};

class Pooling {
public:
	int self_h, self_w, self_stride, self_pad;
	float *x, *dx;
	int *arg_max;
	int out_h, out_w;
	int Height, Width, Channel;
	Pooling()
	{

	}
	Pooling(int pool_h, int pool_w, int stride, int pad, int channel, int height, int width) {
		self_h = pool_h;
		self_w = pool_w;
		self_stride = stride;
		self_pad = pad;
		Channel = channel;
		Height = height;
		Width = width;
		out_h = (height - self_h) / self_stride + 1;
		out_w = (width - self_w) / self_stride + 1;
		arg_max = new int[out_h*out_w*Channel];
	}
	float* Forward(float *in)
	{	
		x = new float[out_h*out_w*Channel];
		int i, j, k, r, c, max_arg=0;
		float max;
		for (i = 0; i < Channel; i++)
		{
			for (j = 0; j < out_h; j++)
			{
				for (k = 0; k < out_w; k++)
				{
					max = -11111;
					for (r = 0; r < self_h; r++)
					{
						for (c = 0; c < self_w; c++)
						{
							if(r+j*self_stride<Height && c+k*self_stride < Width)
								if (max < in[i*Height*Width + (r + j * self_stride)*Width + c + k * self_stride])
								{
									max = in[i*Height*Width + (r + j * self_stride)*Width + c + k * self_stride];
									max_arg = r * self_w + c;
								}
						}
					}
					x[i*out_h*out_w + j * out_w + k] = max;
					arg_max[i*out_h*out_w + j * out_w + k] = max_arg;
				}
			}
		}
		delete[] in;
		return x;
	}
	float* backward(float* dout)
	{
		dx = (float*)calloc(Height*Width*Channel, sizeof(float));
		int p_r, p_c, p;
		int i, j, k, r, c;
		for (i = 0; i < Channel; i++)
		{
			for (j = 0; j < Height; j++)
			{
				for (k = 0; k < Width; k++)
				{
					if (j%self_stride == 0 && k%self_stride == 0)
					{
						if (j / self_stride < out_h && k / self_stride < out_w)
						{
							p = arg_max[i*out_h*out_w + (j / self_stride) * out_w + k / self_stride];
							p_c = p % self_w;
							p_r = p / self_w;
							for (r = 0; r<self_w; r++)
								for (c = 0; c < self_h; c++) {
									if (p_c == c && p_r == r && j + c<Height && k + r<Width)
										dx[i*Width*Height + (j + c) * Width + k + r] += dout[i*out_w*out_h + j / self_stride * out_w + k / self_stride];
								}
						}						
					}	
				}
			}
		}
	
		delete[] dout;
		return dx;
	}
	~Pooling()
	{
		// delete[] arg_max;
	}
};

class Relu {
public:
	int *mask, Size;
	Relu() {

	}
	Relu(int size) {
		Size = size;
		mask = (int *)calloc(size, sizeof(int));
	}
	void forward(float *x)
	{
		for (int i = 0; i < Size; i++)
		{
			x[i] = (x[i] > 0) ? x[i] : 0;
			mask[i] = 1;
		}
	}
	void backward(float *dout)
	{
		for (int i = 0; i < Size; i++)
		{
			dout[i] = (mask[i] == 1) ? dout[i] : 0;
		}
	}
	~Relu() 
	{
		// delete[] mask;
	}
};

class BatchNormalization {
public: 
	float Mean=0, Var=0, Std=0, Gamma, Beta, *temp;
	int Size;
	BatchNormalization() {

	}

	BatchNormalization(int size, float gamma, float beta) 
	{
		temp = new float[size];
		Size = size;
		Gamma = gamma;
		Beta = beta;
	}
	float* forward(float* data) 
	{
		int i;
		for (i = 0; i < Size; i++) Mean += data[i];
		Mean /= Size;
		for (i = 0; i < Size; i++) temp[i] = data[i] - Mean;
		for (i = 0; i < Size; i++) Var += temp[i] * temp[i];
		Std = (float)sqrt((double)Var + 1e-308);
		for (i = 0; i < Size; i++) data[i] = temp[i] / Std * Gamma + Beta;
		return data;
	}
	float* backward(float *dout)
	{
		float dstd=0, dvar=0, dmean=0;
		for (int i = 0; i < Size; i++) dout[i] *= Gamma;
		for (int i = 0; i < Size; i++) dstd += -(dout[i] * temp[i]) / (Std*Std);

		dvar = (float)0.5*dstd / Std;
		for (int i = 0; i < Size; i++) dout[i] = (dout[i]/Std) + 2 * temp[i] * dvar;
		for (int i = 0; i < Size; i++) dmean += dout[i];
		for (int i = 0; i < Size; i++) dout[i]=(dout[i]-dmean);

		return dout;
	}
	~BatchNormalization() 
	{
		// delete[] temp;
	}
};
//Layers End



//Network Start
class MultilayerNet {
public:
	int self_inputSize, *self_hidden_list, self_hidden_num, self_outputSize;
	float **weight_conv, **bias_conv, *weight_affine, *bias_affine;
	float Loss, scale;
	int total_weight, Accuracy=0;
	int i, *a, k = 0, l, m, j, w, h;

	BatchNormalization *Batch;
	Relu *relu;
	Convolution *Conv;
	Pooling *Pool;
	Affine affine;
	SoftmaxwithLoss Softmax_Loss;
	FILE *fin, *fout;

	MultilayerNet() {

	}
	MultilayerNet(int inputSize, int hiddenSize_list[], int hidden_num, int OutputSize) {
		self_inputSize = inputSize;
		self_hidden_list = hiddenSize_list;
		self_hidden_num = hidden_num;
		self_outputSize = OutputSize;

		Batch = new BatchNormalization[hidden_num + 1];
		relu = new Relu[hidden_num];
		Conv = new Convolution[hidden_num];
		Pool = new Pooling[hidden_num];

		a = new int[self_hidden_num];

		for (i = 0; i < hidden_num; i++) k += hiddenSize_list[i];

		weight_conv = (float**)malloc(sizeof(float*)*self_hidden_num);
		bias_conv = (float**)malloc(sizeof(float*)*self_hidden_num);

		l = IMG_WIDTH;
		m = IMG_HEIGHT;
		for (i = 0; i < self_hidden_num; i++)
		{
			a[i] = (i == 0) ? FILTER_SIZE * FILTER_SIZE*RGB*self_hidden_list[i] : FILTER_SIZE * FILTER_SIZE*self_hidden_list[i - 1] * self_hidden_list[i];
			weight_conv[i] = (float *)calloc(a[i], sizeof(float));
			bias_conv[i] = (float *)calloc(self_hidden_list[i],sizeof(float));
			l = (l + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
			l = (l - POOL) / STRIDE + 1;
			m = (m + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
			m = (m - POOL) / STRIDE + 1;
		}
		weight_affine = (float *)calloc(l*m*self_hidden_list[i-1] * CLASSIFY, sizeof(float));
		bias_affine = (float *)calloc(CLASSIFY, sizeof(float));
		l = IMG_WIDTH;
		m = IMG_HEIGHT;
		if (fin = fopen("params.txt", "rb"))
		{
			for (i = 0; i < self_hidden_num; i++)
			{
				a[i] = (i == 0) ? FILTER_SIZE * FILTER_SIZE*RGB*self_hidden_list[i] : FILTER_SIZE * FILTER_SIZE*self_hidden_list[i - 1] * self_hidden_list[i];				
				l = (l + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
				l = (l - POOL) / STRIDE + 1;
				m = (m + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
				m = (m - POOL) / STRIDE + 1;
				fread(weight_conv[i], sizeof(float), a[i], fin);
				fread(bias_conv[i], sizeof(float), self_hidden_list[i], fin);
			}
			fread(weight_affine, sizeof(float), l*m*self_hidden_list[i - 1] * CLASSIFY, fin);
			fread(bias_affine, sizeof(float), CLASSIFY, fin);
			fclose(fin);
		}
		else
		{
			scale = (float)sqrt(2.0 / (double)(self_inputSize + k + self_outputSize));
			for (i = 0; i < self_hidden_num; i++)
			{
				l = (l + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
				l = (l - POOL) / STRIDE + 1;
				m = (m + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
				m = (m - POOL) / STRIDE + 1;
				a[i] = (i == 0) ? FILTER_SIZE * FILTER_SIZE*RGB*self_hidden_list[i] : FILTER_SIZE * FILTER_SIZE*self_hidden_list[i - 1] * self_hidden_list[i];
				for (k = 0; k < a[i]; k++) weight_conv[i][k] = ((float)rand() / (float)RAND_MAX) * scale;
			}
			for (j = 0; j < l*m*self_hidden_list[i - 1] * CLASSIFY; j++) weight_affine[j] = ((float)rand() / (float)RAND_MAX)*scale;			
		}
		i = 0;
		j = RGB;
		w = IMG_WIDTH;
		h = IMG_HEIGHT;
		for (i = 0; i < hidden_num; i++)
		{
			Relu relu_temp(w*h*j);
			BatchNormalization batch_temp(w*h*j, 1, 0);
			Convolution conv_temp(weight_conv[i], bias_conv[i], STRIDE, PAD, FILTER_SIZE, self_hidden_list[i], j, w, h);
			w = (w + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
			h = (h + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
			j = self_hidden_list[i];
			Pooling pool_temp(POOL, POOL, STRIDE, PAD, j, h, w);
			w = (w - POOL) / STRIDE + 1;
			h = (h - POOL) / STRIDE + 1;
			if (i != 0) relu[i-1] = relu_temp;
			Batch[i] = batch_temp;
			Conv[i] = conv_temp;
			Pool[i] = pool_temp;
		}
		Relu relu_temp(w*h*j);
		relu[i-1] = relu_temp;
		SoftmaxwithLoss soft(CLASSIFY);
		BatchNormalization batch_temp(w*h*j, 1, 0);
		Affine affine_temp(weight_affine, bias_affine, w*h*j, CLASSIFY);
		Softmax_Loss = soft;
		Batch[i] = batch_temp;
		affine = affine_temp;
	}
	float* predict(float *x)
	{
		float *x_temp;
		for (int i = 0; i < self_hidden_num; i++)
		{
			x = Batch[i].forward(x);
			x = Conv[i].Forward(x);
			x = Pool[i].Forward(x);
			relu[i].forward(x);
			x_temp = x;
		}
		x_temp = Batch[self_hidden_num].forward(x_temp);
		x_temp=affine.forward(x_temp);
		return x_temp;
	}
	float loss(float *x, unsigned char *t, int *acc)
	{
		int temp = *acc;
		x=predict(x);
		float LOSS = Softmax_Loss.forward(x, t);
		*acc = temp + accuracy(x, t);
		delete[] x;
		delete[] t;
		return LOSS;
	}
	int accuracy(float *x, unsigned char *t)
	{
		float max = 0;
		int arg_max = 0;
		for (int i = 0; i < CLASSIFY; i++) {
			if (max < x[i])
			{
				max = x[i];
				arg_max = i;
			}
		}
		for (int i = 0; i < CLASSIFY; i++) {
			if ((t[i] == 1) & (arg_max == i)) return 1;
		}
		return 0;
	}
	void gradient(float *x, unsigned char *t)
	{
		Loss = loss(x, t, &Accuracy);
		x = Softmax_Loss.backward(Loss);
		x = affine.backward(x);
		x = Batch[self_hidden_num].backward(x);
		float *x_temp = x;
		for (int i = self_hidden_num - 1; i >= 0; i--)
		{
			relu[i].backward(x);
			x = Pool[i].backward(x);
			x = Conv[i].backward(x);
			x = Batch[i].backward(x);
			x_temp = x;
		}
		free(x);
	}
	~MultilayerNet() {
		
	}
};
// Network End



//Trainer Start
class Trainer {
public:
	float self_lr = LR;
	int list[CONV_NUM] = { 10 };
	int chk = 0;
	MultilayerNet network;
	SGD sgd;
	Trainer() {
		MultilayerNet NETWORK(IMG_HEIGHT*IMG_WIDTH*RGB, list, CONV_NUM, CLASSIFY);
		network = NETWORK;
		SGD SGD(self_lr);
		sgd = SGD;
	}
	void Train(float *x_train, unsigned char *t_train) {
		network.gradient(x_train, t_train);
		int j = 0 , w = 0, h = 0;
		j = RGB;
		w = IMG_WIDTH;
		h = IMG_HEIGHT;
		for (int i = 0; i < network.self_hidden_num; i++)
		{
			sgd.update(network.weight_conv[i], network.Conv[i].dW, FILTER_SIZE*FILTER_SIZE*j*network.self_hidden_list[i]);
			sgd.update(network.bias_conv[i], network.Conv[i].db, network.self_hidden_list[i]);
			j = network.self_hidden_list[i];
			w = (w + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
			w = (w - POOL) / STRIDE + 1;
			h = (h + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
			h = (h - POOL) / STRIDE + 1;
		}
		sgd.update(network.weight_affine, network.affine.dW, w*h*j*CLASSIFY);
		sgd.update(network.bias_affine, network.affine.db, CLASSIFY);
		chk++;
		if (chk % MINIBATCH_SIZE == 0)
		{
			printf("Accuracy : %.1f%%\n", (float)network.Accuracy / chk * 100);
			network.Accuracy = 0;
			chk = 0;
		}
	}
	~Trainer()
	{
		for (int i = 0; i < network.self_hidden_num; i++) {
			free(network.Conv[i].dW);
			free(network.Conv[i].db);
			free(network.Conv[i].Self_in);
			free(network.Pool[i].arg_max);
			free(network.relu[i].mask);
			free(network.Batch[i].temp);
			free(network.weight_conv[i]);
			free(network.bias_conv[i]);
		}
		free(network.Batch[network.self_hidden_num].temp);
		free(network.a);
		free(network.weight_conv);
		free(network.bias_conv);
		free(network.weight_affine);
		free(network.bias_affine);
		free(network.affine.self_in);
		free(network.affine.dW);
		free(network.affine.db);
		free(network.Softmax_Loss.self_pre);
		free(network.Softmax_Loss.self_label);
	}
};
//Trainer End


//Tester Start
class Tester
{
public:
	int list[CONV_NUM] = { 10 };
	int chk = 0;
	int acc = 0;
	int pre_label;
	MultilayerNet network;
	Tester() 
	{
		MultilayerNet NETWORK(IMG_HEIGHT*IMG_WIDTH*RGB, list, CONV_NUM, CLASSIFY);
		network = NETWORK;
	}
	void test(float *x, unsigned char label)
	{
		
		network.predict(x);
		softmax(x, CLASSIFY);
		chk++;
		pre_label=arg_max(x, CLASSIFY);
		if (pre_label == label) acc++;
		if (chk == MINIBATCH_SIZE)
		{
			printf("Accuracy: %.1f%%\n", (float)acc / chk * 100);
			chk = 0;
			acc = 0;
		}
	}
	~Tester() 
	{

	}
};
//Tester End
