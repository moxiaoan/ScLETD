#ifndef _SCLETD_HIP_H_
#define _SCLETD_HIP_H_

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "rocblas.h"
#include <hipfft.h>
#include <complex.h>
#include "add.h"

//#define ScLETD_DEBUG
#define LEN 256
#define MODE 3
#define BLOCK 64
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 4

int deviceId;
hipDeviceProp_t props;
hipEvent_t st, ed;
hipEvent_t st2, ed2;
float dgemm_time, F1_time, F2_time, fieldE1_time, updateU_new_time, calc_time;
float trans_unpack_time;
float trans_mu_time;
float trans_enlarge_time;
float trans_pack_time;
float trans_Memcpy_time;
float trans_MPI_time;
float trans_time;
float timer;
float timer2;
float conv_transfer_time, conv_calculate_time, elastic_calculate_time;

double *n11, *n22, *n33;
double *Epsilon2d[NELASTIC], *Sigma2d[NELASTIC], *C4D;
double *BN;
double *tmpy_RE1, *tmpy_fftRE[NELASTIC][NELASTIC], *tmpy_fftIM[NELASTIC], *BN_fftRE[NELASTIC], *BN_fftIM[NELASTIC], *bn_RE, *bn_IM, *BN_RE, *BN_IM, *eta_RE, *eta_IM, *fftRE[NELASTIC][NELASTIC];
double *Elas, *elas, *Elas1;
double *S_left, *S_right, *S_top, *S_bottom, *S_front, *S_back;
double *R_left, *R_right, *R_top, *R_bottom, *R_front, *R_back;

//DCU variables
  double *fieldEr;
  Stype *field2B, *field2B2;
  double *fieldE, *fielde, *field, *field1, *field2, *field3;
  double *fieldEu_top, *fieldEu_bottom, *fieldEu_front, *fieldEu_back;
  double *fieldEu_left, *fieldEu_right;
  double *fieldEr_front;
  double *fieldEr_back;
  double *fieldEr_top, *fieldEs_top;
  double *fieldEr_bottom, *fieldEs_bottom;
  double *fieldEr_left, *fieldEs_left;
  double *fieldEr_right, *fieldEs_right;
  double *fieldEe_front, *fieldEe_back;
  double *fieldEe_top, *fieldEe_bottom;
  double *fieldEe_left, *fieldEe_right;
  //declare update
  Dtype *ddx, *ddy, *ddz;
  //declare F1
  Dtype *f1;
  //declare F2
  Dtype *f2;
  Dtype *lambda;
  //declare dgemm
  Dtype *mpxi, *mpyi, *mpzi, *mpx, *mpy, *mpz;
  rocblas_handle handle;
  rocblas_int x_m, x_n, x_k;
  rocblas_int y_m, y_n, y_k;
  rocblas_int z_m, z_n, z_k;
  rocblas_int Gx_m, Gx_n, Gx_k;
  rocblas_int Gy_m, Gy_n, Gy_k;
  rocblas_int Gz_m, Gz_n, Gz_k;
  
  rocblas_int conv_x_m, conv_x_n, conv_x_k, conv_y_m, conv_y_n, conv_y_k, conv_z_m, conv_z_n, conv_z_k;
  rocblas_int conv_big_x_m, conv_big_x_n, conv_big_x_k, conv_big_y_m, conv_big_y_n, conv_big_y_k, conv_big_z_m, conv_big_z_n, conv_big_z_k;

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

//void init_KL();
//void write_chk(void);
//void write_field2B(int irun);
//void read_chk(void);
//void ac_calc_F1(unsigned short int *f);
#endif 
