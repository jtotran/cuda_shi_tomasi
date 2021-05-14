#ifndef SHI_TOMASI_H
#define SHI_TOMASI_H

#include <cuda.h> 
#include <cuda_runtime_api.h>
#include <math.h>
#include <omp.h>
#include <stddef.h> 
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h> 

#include <thrust/device_vector.h> 
#include <thrust/host_vector.h> 
#include <thrust/sort.h> 
#include <thrust/copy.h> 

typedef struct dataWrapperT {
    float data;
    size_t x;
    size_t y;
} dataWrapperT;

struct sortDataWrapperT {
    __host__ __device__
    bool operator() (const struct dataWrapperT &a, const struct dataWrapperT &b) {
        return a.data > b.data;
    }
};

__host__
float *computeKernel(float sigma, int *w, int *a); 
__host__
float *computeGderiv(float sigma); 

__global__
void convolve(float *image, float *result_image, float *kernel, int image_width, int image_height, int kernal_width, int
        kernel_height, int block_width);

__device__
float minEigenvalue(float a, float b, float c, float d); 
__global__ 
void computeEigenvalues(float *hgrad, float *vgrad, int image_width, int image_height, int window_size, float
        *eigenvalues); 

__global__
void computeCorners(float *corners, int *corner_indices, int image_width, int image_height, int block_width);
__device__ 
void drawBox(float *image, int index, int image_width, int image_height, int radius); 
__global__
void preDraw(float *image, int *corner_indices, int image_width, int image_height, int corner_arr_size, int radius); 

#endif 
