#include "image_template.h" 
#include "shi_tomasi.cuh"

__host__ 
float *computeKernel(float sigma, int *w, int *a) {
    float *G;
    int temp_a = round(2.5 * (sigma) - 0.5);
    float sum = 0;

    int temp_w = 2 * temp_a + 1;
    G = (float *)malloc(sizeof(float) * temp_w); 

    for(int i = 0; i < temp_w; i++) {
        G[i] = exp((-1 * (i - temp_a) * (i - temp_a)) / (2 * sigma * sigma));
        sum = sum + G[i]; 
    }

    for(int i = 0; i < temp_w; i++) {
        G[i] = G[i] / sum;
    }

    *w = temp_w; 
    *a = temp_a; 
    return G;
}

__host__
float *computeGderiv(float sigma) {
    float *Gderiv;
    int a = round(2.5 * (sigma) - 0.5);
    float sum = 0;

    int w = 2 * a + 1;
    Gderiv = (float *)malloc(sizeof(float) * w); 

    for(int i = 0; i < w; i++) {
        Gderiv[i] = -1 * (i - a) * exp((-1 * (i - a) * (i - a)) / (2 * sigma * sigma));
        sum = sum - i * Gderiv[i]; 
    }
    for(int i = 0; i < w; i++) {
        Gderiv[i] = Gderiv[i] / sum; 
    }

    float temp = 0;
    for(int i = 0; i < (w / 2); i++) {
        temp = Gderiv[w - 1 - i];
        Gderiv[w - 1 - i] = Gderiv[i];
        Gderiv[i] = temp; 
    }

    return Gderiv;
}

__global__ 
void convolve(float *image, float *result_image, float *kernel, int image_width, int image_height, int kernel_width, int kernel_height, int block_width) {
    extern __shared__ float shr_image[]; 
    int localidx = threadIdx.x, localidy = threadIdx.y; 
    int globalidx = localidx + blockIdx.x * blockDim.x;
    int globalidy = localidy + blockIdx.y * blockDim.y; 

    shr_image[localidx * block_width + localidy] = image[globalidx * image_width + globalidy];  

    __syncthreads(); 

    int sum = 0.0;
    int offseti = 0;
    int offsetj = 0;
    for(int k = 0; k < kernel_height; k++) {
        for(int m = 0; m < kernel_width; m++) {
            offseti = -1 * (kernel_height / 2) + k;
            offsetj = -1 * (kernel_width / 2) + m;
            if((localidx + offseti) > -1 && (localidx + offseti) < block_width && (localidy + offsetj) > -1 &&
                (localidy + offsetj) < block_width) {
                sum = sum + shr_image[(localidx + offseti) * block_width + (localidy + offsetj)] * kernel[k *
                    kernel_width + m];
                continue; 
            }
            if(!((globalidx + offseti) > -1 && (globalidx + offseti) < image_height && (globalidy + offsetj) > -1 &&
                        (globalidy + offsetj) < image_width)) {
                continue;
            }
            sum = sum + image[(globalidx + offseti) * image_width + (globalidy + offsetj)] * kernel[k * kernel_width + m]; 
        }
    }
    result_image[globalidx * image_width + globalidy] = sum; 
     
    return; 
} 

__device__
float minEigenvalue(float a, float b, float c, float d) {
    float ev_one = (a + d) / 2 + pow(((a + d) * (a + d)) / 4 - (a * d - b * c), 0.5);
    float ev_two = (a + d) / 2 - pow(((a + d) * (a + d)) / 4 - (a * d - b * c), 0.5);

    if(ev_one >= ev_two) { 
        return ev_two;
    }
    else { 
        return ev_one; 
    }

    return ev_one;
}

__global__ 
void computeEigenvalues(float *hgrad, float *vgrad, int image_width, int image_height, int window_size, float
        *eigenvalues) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y; 

    int w = floorf(window_size / 2); 
    int offseti, offsetj;
    float ixx_sum = 0, iyy_sum = 0, ixiy_sum = 0; 

    for(int k = 0; k < window_size; k++) {
        for(int m = 0; m < window_size; m++) {
            offseti = -1 * w + k;
            offsetj = -1 * w + m;
            if (i + offseti < 0 || i + offseti >= image_height || j + offsetj < 0 || j + offsetj >= image_width) {
                continue; 
            }
            ixx_sum += hgrad[(i + offseti) * image_width + (j + offsetj)] * hgrad[(i + offseti) * image_width + (j +
                    offsetj)];
            iyy_sum += vgrad[(i + offseti) * image_width + (j + offsetj)] * vgrad[(i + offseti) * image_width + (j +
                    offsetj)];
            ixiy_sum += hgrad[(i + offseti) * image_width + (j + offsetj)] * vgrad[(i + offseti) * image_width + (j +
                    offsetj)];
        }
    }
    eigenvalues[i * image_width + j] = minEigenvalue(ixx_sum, ixiy_sum, ixiy_sum, iyy_sum); 

    return;
}

__global__ 
void computeCorners(float *corners, int *corner_indices, int image_width, int image_height, int block_width) {
    /* Original Implementation */
    /*
    extern __shared__ struct dataWrapperT shr_wrapper[]; 
    int localidx = threadIdx.x, localidy = threadIdx.y; 
    int globalidx = localidx + blockIdx.x * blockDim.x;
    int globalidy = localidy + blockIdx.y * blockDim.y; 

    shr_wrapper[localidx * block_width + localidy] = wrapped_eigs[globalidx * image_width + globalidy];  

    __syncthreads(); 

    for(int s=((blockDim.x * blockDim.y) / 2); s > 0; s >>= 1) {
        if((localidx * block_width + localidx) >= s) {
            continue; 
        }
        if(shr_wrapper[localidx * block_width + localidy].data > shr_wrapper[(localidx * block_width + localidy) +
                s].data) {
            shr_wrapper[localidx * block_width + localidy] = shr_wrapper[(localidx * block_width + localidy)]; 
            __syncthreads();
        } 
        else {
            shr_wrapper[localidx * block_width + localidy] = shr_wrapper[(localidx * block_width + localidy) + s]; 
            __syncthreads();
        }
    }

    if(localidx == 0 && localidy == 0) {
        max_block_eigs[blockIdx.x * (image_width / block_width) + blockIdx.y] = shr_wrapper[localidx * block_width + localidy];  
    }

    return;
    */ 
    /* Implementation with help from Dr. Pallipuram */ 
    int locali = threadIdx.x;
    int localj = threadIdx.y;
    int globali = threadIdx.x + blockIdx.x * blockDim.x;
    int globalj = threadIdx.y + blockIdx.y * blockDim.y; 
    
    extern __shared__ float A[];

    A[locali * block_width + localj] = corners[globali * image_width + globalj];
    __syncthreads(); 
    A[locali * block_width + localj + block_width * block_width] = globali * image_width + globalj;  
    __syncthreads(); 

    int t = threadIdx.x * block_width + threadIdx.y;
    for(int stride = (blockDim.x * blockDim.y)/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if(t < stride) {
            if(A[t + stride] > A[t]) {
                A[t] = A[t + stride]; 
                A[t + block_width * block_width] = A[t + stride + block_width * block_width]; 
            }
        }
    }

    if(locali == 0 && localj == 0) {
        corner_indices[blockIdx.x * gridDim.y + blockIdx.y] = A[0 + block_width * block_width]; 
    }

    return;
} 

__device__
void drawBox(float *image, int index, int image_width, int image_height, int radius) {

    int k = 0;
    int m = 0;
    int i = index / image_height;
    int j = index % image_width;
    for(k = -1 * radius; k <= radius; k++) {
        for(m = -1 * radius; m <= radius; m++) {
            if((i + k) < 0 || (i + k) >= image_height || (j + m) < 0 || (j + m) >= image_width) {
                continue;
            }
            image[(i + k) * image_width + (j + m)] = 0;
        }
    }
    return; 
}

__global__
void preDraw(float *image, int *corner_indices, int image_width, int image_height, int corner_arr_size, int radius) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y; 

    if(i < 0 || i >= corner_arr_size || j < 0 || j >= corner_arr_size) {
        return;
    }

    drawBox(image, corner_indices[i * corner_arr_size + j], image_width, image_height, radius); 
    return;
}

__host__ 
int shiTomasi(int argc, char *argv[]) {
    struct timeval comp_start, comp_end;
    unsigned long comp_time;

    char *filepath = argv[1];
    float sigma = atof(argv[2]);
    /* Sensitivity is not used in this implementation */
    /* float sensitivity = 0.1;  */
    int windowsize = atoi(argv[3]);
    int block_width = atoi(argv[4]);

    float *h_image = NULL;
    float *d_image = NULL; 
    int width, height;

    float *h_kern_gaus = NULL, *h_kern_deriv = NULL;
    float *d_kern_gaus = NULL, *d_kern_deriv = NULL;
    int kernel_width, a; 

    float *d_temp_horiz = NULL, *d_horiz = NULL;
    float *d_temp_vert = NULL, *d_vert = NULL;
    float *h_horiz = NULL, *h_vert = NULL;

    float *d_eigenvalues = NULL;
    float *h_eigenvalues = NULL;

    int *d_corner_indices = NULL;
    int *h_corner_indices = NULL;

    read_image_template<float>(filepath, &h_image, &width, &height); 
    h_kern_gaus = computeKernel(sigma, &kernel_width, &a);
    h_kern_deriv = computeGderiv(sigma); 

    cudaMalloc((void **) & d_image, sizeof(float) * width * height); 
    cudaMalloc((void **) & d_kern_gaus, sizeof(float) * kernel_width * 1); 
    cudaMalloc((void **) & d_kern_deriv, sizeof(float) * 1 * kernel_width); 
    cudaMalloc((void **) & d_temp_horiz, sizeof(float) * width * height); 
    cudaMalloc((void **) & d_horiz, sizeof(float) * width * height); 
    cudaMalloc((void **) & d_temp_vert, sizeof(float) * width * height); 
    cudaMalloc((void **) & d_vert, sizeof(float) * width * height); 
    cudaMalloc((void **) & d_corner_indices, sizeof(int) * (width / block_width) * (height / block_width));

    int radius = width * 0.0025;

    h_horiz = (float *)malloc(sizeof(float) * width * height);
    h_vert = (float *)malloc(sizeof(float) * width * height);
    h_eigenvalues = (float *)malloc(sizeof(float) * width * height);
    h_corner_indices = (int *)malloc(sizeof(int) * (width / block_width) * (height / block_width));   cudaMalloc((void **) & d_eigenvalues, sizeof(float) * width * height); 

    dim3 dimBlock(block_width, block_width); 
    dim3 dimGrid(width / block_width, height / block_width);

    cudaMemcpy(d_image, h_image, sizeof(float) * width * height, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_kern_gaus, h_kern_gaus, sizeof(float) * kernel_width * 1, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_kern_deriv, h_kern_deriv, sizeof(float) * 1 * kernel_width, cudaMemcpyHostToDevice); 

    gettimeofday(&comp_start, NULL);
    convolve<<<dimGrid, dimBlock, sizeof(float) * block_width * block_width>>>(d_image, d_temp_horiz, d_kern_gaus,
            width, height, 1, kernel_width, block_width); 
    convolve<<<dimGrid, dimBlock, sizeof(float) * block_width * block_width>>>(d_temp_horiz, d_horiz, d_kern_deriv,
            width, height, kernel_width, 1, block_width); 
    convolve<<<dimGrid, dimBlock, sizeof(float) * block_width * block_width>>>(d_image, d_temp_vert, d_kern_gaus,
            width, height, kernel_width, 1, block_width);  
    convolve<<<dimGrid, dimBlock, sizeof(float) * block_width * block_width>>>(d_temp_vert, d_vert, d_kern_deriv,
            width, height, 1, kernel_width, block_width); 

    computeEigenvalues<<<dimGrid, dimBlock>>>(d_horiz, d_vert, width, height, windowsize, d_eigenvalues);

    computeCorners<<<dimGrid, dimBlock, sizeof(float) * block_width * block_width * 2>>>(d_eigenvalues,
            d_corner_indices, width, height, block_width); 

    preDraw<<<dimGrid, dimBlock>>>(d_image, d_corner_indices, width, height, (width / block_width), radius);
    cudaThreadSynchronize();
    gettimeofday(&comp_end, NULL);

    cudaMemcpy(h_image, d_image, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    /*
    char horiz_file_name[] = "horizontal.pgm";
    char vert_file_name[] = "vertical.pgm";
    char eigen_file_name[] = "eigenvalues.pgm";
    write_image_template<float>(horiz_file_name, h_horiz, width, height); 
    write_image_template<float>(vert_file_name, h_vert, width, height);
    write_image_template<float>(eigen_file_name, h_eigenvalues, width, height);
    */
    char final_file_name[] = "shi_tomasi.pgm";
    write_image_template<float>(final_file_name, h_image, width, height);

    comp_time = ((comp_end.tv_sec * 1000000 + comp_end.tv_usec) - (comp_start.tv_sec * 1000000 + 
                comp_start.tv_usec));
    printf("%d,%d,%ld\n", width, block_width, comp_time);

    free(h_image);
    free(h_kern_gaus);
    free(h_kern_deriv);
    free(h_horiz);
    free(h_vert);
    free(h_eigenvalues);
    free(h_corner_indices);

    cudaFree(d_image);
    cudaFree(d_kern_gaus);
    cudaFree(d_kern_deriv);
    cudaFree(d_temp_horiz);
    cudaFree(d_horiz);
    cudaFree(d_temp_vert);
    cudaFree(d_vert);
    cudaFree(d_eigenvalues);
    cudaFree(d_corner_indices);
 
    return 0; 
}


int main(int argc, char *argv[]) {
    cudaSetDevice(0);
    int err = shiTomasi(argc, argv); 

    return 0; 
}
