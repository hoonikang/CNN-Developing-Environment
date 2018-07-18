#pragma once
#include "functions.h"

int arg_max(float *in, int Size)
{
	int temp = 0;
	float cmp = 0;
	int i;
	for (i = 0; i < Size; i++) {
		if (cmp < in[i])
		{
			cmp = in[i];
			temp = i;
		}
	}
	return temp;
}

void softmax(float *in, int Size)
{
	float sum = 0;
	int i;
	int max_arg=arg_max(in, Size);
	for (i = 0; i < Size; i++)
	{
		sum += (float)exp((double)(in[i]-in[max_arg]));
	}
	float max = in[max_arg];
	for (i = 0; i < Size; i++)
		in[i] = (float)exp((double)(in[i] - max)) / sum;
}
float cross_entropy_error(float *prob, int Size, unsigned char *Label)
{
	float sum = 0;
	for (int i = 0; i < Size; i++)
		sum += -(float)Label[i] * (float)log((double)prob[i]+1e-307);
	return sum;
}
float softmax_with_loss(float *in, int Size, unsigned char *Label)
{
	softmax(in, Size);
	int max_arg = arg_max(in, Size);
	return cross_entropy_error(in, Size, Label);
}

