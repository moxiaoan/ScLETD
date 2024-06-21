#ifndef _ANISOTROPIC_H_
#define _ANISOTROPIC_H_

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "rocblas.h"

#define BLOCK 64
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 4
#define NELASTIC 1
typedef double Dtype;

int Approx, ela_size;
int flag;
double *fieldE;
double *tmpy_fftre, *fftre;
double *Epsilon2d;
double *Sigma2d;
double *C4D;
double *n11, *n22, *n33;
double *BN;
double *tmpy_RE1, *tmpy_RE2, *tmpy_fftRE, *bn_RE, *bn_IM, *BN_RE, *BN_IM, *eta_RE, *eta_IM, *fftRE;
double *Elas;
Dtype alpha1, beta1, beta2;
Dtype *Gx_re, *Gx_im, *Gy_re, *Gy_im, *Gz_re, *Gz_im, *temp1, *temp2, *temp3, *temp4, *temp5, *temp6, *temp7, *temp8, *Itemp1, *Itemp2, *Itemp3, *Itemp4;
Dtype *conv_big_temp1, *conv_big_temp2, *conv_big_temp3, *conv_big_temp4, *conv_big_temp5, *conv_big_temp6, *conv_big_temp7, *conv_big_temp8;
//convolution variables
double *conv_s_right, *conv_s_bottom, *conv_s_back;
double *conv_r_right, *conv_r_bottom, *conv_r_back;
double *conv_s_left, *conv_s_top, *conv_s_front;
double *conv_r_left, *conv_r_top, *conv_r_front;
double *conv_S_right, *conv_S_bottom, *conv_S_back;
double *conv_R_right, *conv_R_bottom, *conv_R_back;
double *conv_S_left, *conv_S_top, *conv_S_front;
double *conv_R_left, *conv_R_top, *conv_R_front;
double *conv_e, *conv_big_e_re, *conv_big_e_im;
MPI_Request *conv_ireq_left_right, *conv_ireq_top_bottom, *conv_ireq_front_back;
int conv_lr_size, conv_tb_size, conv_fb_size;
MPI_Datatype conv_left_right, conv_top_bottom, conv_front_back;

double *conv_x_re, *conv_x_im, *conv_y_re, *conv_y_im, *conv_z_re, *conv_z_im;
double *conv_big_x_re, *conv_big_x_im, *conv_big_y_re, *conv_big_y_im, *conv_big_z_re, *conv_big_z_im;
Dtype *conv_temp1, *conv_temp2, *conv_temp3, *conv_temp4, *conv_temp5, *conv_temp6, *conv_temp7, *conv_temp8, *conv_temp9, *conv_temp10, *conv_temp11, *conv_temp12;

hipEvent_t st, ed;
hipEvent_t st2, ed2;
hipEvent_t st3, ed3;
hipDeviceProp_t props;

rocblas_handle handle;
rocblas_int x_m, x_n, x_k;
rocblas_int y_m, y_n, y_k;
rocblas_int z_m, z_n, z_k;
rocblas_int Gx_m, Gx_n, Gx_k;
rocblas_int Gy_m, Gy_n, Gy_k;
rocblas_int Gz_m, Gz_n, Gz_k;

rocblas_int conv_x_m, conv_x_n, conv_x_k, conv_y_m, conv_y_n, conv_y_k, conv_z_m, conv_z_n, conv_z_k;
rocblas_int conv_big_x_m, conv_big_x_n, conv_big_x_k, conv_big_y_m, conv_big_y_n, conv_big_y_k, conv_big_z_m, conv_big_z_n, conv_big_z_k;

#endif
