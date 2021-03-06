#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "Timer.h"
#include "train.cpp"
#include "stdafx.h"

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

int main()
{
	char order[6], num;
	int dummy;
	unsigned char dummy3;
	unsigned char *dummy2;
	FILE *img, *lbl;
	float **image;
	unsigned char **label;
	int image_size, label_size;

	CPerfCounter counter;
	float elapsedTime=0;

	//CHOOSE INSTRUCTION
	printf("명령을 입력해주세요(train/test): ");
	scanf("%s", order);

	//TRAIN
	if (!strcmp(order, "train"))
	{
		Trainer train;

		//MNIST IMAGE OPEN
		img = fopen("train_image.idx3-ubyte", "rb");
		fread(&dummy, sizeof(int), 1, img);
		fread(&image_size, sizeof(int), 1, img);
		image_size = ReverseInt(image_size);
		fread(&dummy, sizeof(int), 1, img);
		fread(&dummy, sizeof(int), 1, img);
		image = (float**)malloc(image_size * sizeof(float*));

		//MNIST LABEL OPEN
		lbl = fopen("train_label.idx1-ubyte", "rb");
		fread(&dummy, sizeof(int), 1, lbl);
		fread(&label_size, sizeof(int), 1, lbl);
		label_size = ReverseInt(label_size);
		label = (unsigned char**)malloc(label_size * sizeof(unsigned char*));

		//Train
		for (int i = 0; i < image_size; i++)
		{
			counter.Start();
			image[i] = (float *)malloc(sizeof(float)*IMG_WIDTH*IMG_HEIGHT);
			dummy2 = (unsigned char*)malloc(sizeof(unsigned char)*IMG_WIDTH*IMG_HEIGHT);
			fread(dummy2, sizeof(unsigned char), IMG_WIDTH*IMG_HEIGHT, img);
			for (int j = 0; j < IMG_WIDTH*IMG_HEIGHT; j++) image[i][j] = (float)dummy2[j];
			free(dummy2);

			label[i] = (unsigned char*)malloc(sizeof(unsigned char)*CLASSIFY);
			fread(&dummy3, sizeof(unsigned char), 1, lbl);
			for (int j = 0; j < CLASSIFY; j++) label[i][j] = (j == dummy3) ? 1 : 0;
			train.Train(image[i], label[i]);
			counter.Stop();
			elapsedTime += counter.GetElapsedTime();			
			counter.Reset();
		}
		printf("\n\nTime for train data is %.1fs\n", elapsedTime);
		fclose(img);
		fclose(lbl);
		//Write Parameters
		train.network.fout = fopen("params.txt", "wb");
		for (int i = 0; i < train.network.self_hidden_num; i++)
		{
			train.network.l = IMG_WIDTH;
			train.network.m = IMG_HEIGHT;
			for (i = 0; i < train.network.self_hidden_num; i++)
			{
				train.network.l = (train.network.l + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
				train.network.l = (train.network.l - POOL) / STRIDE + 1;
				train.network.m = (train.network.m + 2 * PAD - FILTER_SIZE) / STRIDE + 1;
				train.network.m = (train.network.m - POOL) / STRIDE + 1;
				train.network.a[i] = (train.network.i == 0) ? FILTER_SIZE * FILTER_SIZE*RGB*train.network.self_hidden_list[i] : FILTER_SIZE * FILTER_SIZE*train.network.self_hidden_list[i - 1] * train.network.self_hidden_list[i];
				fwrite(train.network.weight_conv[i], sizeof(float), train.network.a[i], train.network.fout);
				fwrite(train.network.bias_conv[i], sizeof(float), train.network.self_hidden_list[i], train.network.fout);
			}
			fwrite(train.network.weight_affine, sizeof(float), train.network.l*train.network.m*train.network.self_hidden_list[i - 1] * CLASSIFY, train.network.fout);
			fwrite(train.network.bias_affine, sizeof(float), CLASSIFY, train.network.fout);
		}
		fclose(train.network.fout);
		fclose(img);
		fclose(lbl);
		free(image);
		free(label);
	}

	//TEST
	else if (!strcmp(order, "test"))
	{
		Tester test;
		//MNIST IMAGE OPEN
		img = fopen("test_image.idx3-ubyte", "rb");
		fread(&dummy, sizeof(int), 1, img);
		fread(&image_size, sizeof(int), 1, img);
		image_size = ReverseInt(image_size);
		fread(&dummy, sizeof(int), 1, img);
		fread(&dummy, sizeof(int), 1, img);
		image = (float**)malloc(image_size * sizeof(float*));

		//MNIST LABEL OPEN
		lbl = fopen("test_label.idx1-ubyte", "rb");
		fread(&dummy, sizeof(int), 1, lbl);
		fread(&label_size, sizeof(int), 1, lbl);
		label_size = ReverseInt(label_size)	;

		//Train
		for (int i = 0; i < image_size; i++)
		{
			counter.Start();
			image[i] = (float *)malloc(sizeof(float)*IMG_WIDTH*IMG_HEIGHT);
			dummy2 = (unsigned char*)malloc(sizeof(unsigned char)*IMG_WIDTH*IMG_HEIGHT);
			fread(dummy2, sizeof(unsigned char), IMG_WIDTH*IMG_HEIGHT, img);
			for (int j = 0; j < IMG_WIDTH*IMG_HEIGHT; j++) image[i][j] = (float)dummy2[j];
			free(dummy2);

			fread(&dummy3, sizeof(unsigned char), 1, lbl);
			test.test(image[i], dummy3);
			counter.Stop();
			elapsedTime += counter.GetElapsedTime();
			counter.Reset();
			
		}
		printf("\nTime for test data is %.1fs\n", elapsedTime);
		fclose(img);
		fclose(lbl);
		free(image);
	}
	return 0;
}
