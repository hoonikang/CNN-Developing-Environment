#pragma once
#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define RGB 1
#define FILTER_SIZE 3
#define STRIDE 1
#define PAD 1
#define POOL 2
#define CLASSIFY 10
#define MINIBATCH_SIZE 5000
#define LR 0.0000000001
#define CONV_NUM 1

#include <math.h>


int arg_max(float *in, int Size);
void softmax(float *in, int Size);
float cross_entropy_error(float *prob, int Size, unsigned char *Label);
float softmax_with_loss(float *in, int Size, unsigned char *Label);
