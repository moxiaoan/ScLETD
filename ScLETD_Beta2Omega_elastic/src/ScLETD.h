#ifndef _SCLETD_H_
#define _SCLETD_H_

#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <mm_malloc.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#define f_A (2376.0 / 2372.0) // 1
#define f_B ((7128.0 + 12.0 * 2372.0) / 2372.0) // 15
#define f_C ((4752.0 + 12.0 * 2372.0) / 2372.0) // 14
#define PI 3.141592653589793

typedef double Dtype;
typedef unsigned short int Stype;

#define ORI 64
#define DIM 3
#define NELASTIC 4

int flag;
int Approx;
int NX, NY, NZ;
double epsilon2d[NELASTIC][DIM * DIM];
double sigma2d[NELASTIC][DIM * DIM];
double C4d[DIM * DIM * DIM * DIM];
double *tmpy_re, *tmpy_im, *tmpy_fftre[NELASTIC][NELASTIC], *tmpy_fftim[NELASTIC], *Bn_fftre[NELASTIC], *Bn_fftim[NELASTIC], *fftre[NELASTIC][NELASTIC];
double *s_left, *s_right, *s_top, *s_bottom, *s_front, *s_back;
double *r_left, *r_right, *r_top, *r_bottom, *r_front, *r_back;
double *e_left, *e_right, *e_top, *e_bottom, *e_front, *e_back;
MPI_Request *ireq_left_right, *ireq_top_bottom, *ireq_front_back;
//double *n11, *n22, *n33;
//double *Epsilon2d, *Sigma2d, *C4D;
//double *BN[NELASTIC][NELASTIC];
//double *tmpy_RE1, *tmpy_fftRE[NELASTIC], *tmpy_fftIM[NELASTIC], *BN_fftRE[NELASTIC], *BN_fftIM[NELASTIC];
//double *Elas, *elas;
//double *S_left, *S_right, *S_top, *S_bottom, *S_front, *S_back;
//double *R_left, *R_right, *R_top, *R_bottom, *R_front, *R_back;
double ElasticScale;
int ELASTIC;


int tt;
int out_tt, out_iter;
double functime, runtime, walltime;
int ot2;
int ot1;
int ioutput;
int nout;
double rad;
double percent;
int rotation;
double *theta;
int ANISOTROPIC;
int nac, nch;
int iter;
int left, right, top, bottom, front, back;
int nx, ny, nz;

char processor_name[MPI_MAX_PROCESSOR_NAME];
int  namelen;
int nprocs, procs[3], myrank, prank, cart_id[3];
int periodic;
double *MPX, *MPY, *MPZ, *MPXI, *MPYI, *MPZI, *DDX, *DDY, *DDZ;
double *MPX_b, *MPY_b, *MPZ_b, *MPXI_b, *MPYI_b, *MPZI_b, *DDX_b, *DDY_b, *DDZ_b;
double A1, A2, A3, A4, A5, C1;
double dt, t_total;
double epn2;
double xmin, ymin, zmin;
double xmax, ymax, zmax;

float *field2_all;
unsigned char* field2_cp;

//new
int lr_size, tb_size, fb_size;
int block_x=2, block_y=2, block_z=2;        // 每个下采样空间三个维度的节点数
int stride_x = 2, stride_y = 2, stride_z = 2;
int block_size = block_x*block_y*block_z;
int is_gather_rank = 0;                            // 对应该进程的汇总节点

int simulate_size;
int simulate_ds_size;
int one_func_simulate_ds_size;
int gatherd_size;
int compressed_size;
int recv_count=0, send_count=0;



double hx, hy, hz;
double kkx, kky, kkz;
int restart;
char work_dir[1024];
char data_dir[1024];
char detect_result_dir[1024];

int restart_iter;
char restart_dir[1024];
int ini_num;
int nghost;
int ix1, ix2, ix3, ix4, iy1, iy2, iy3, iy4, iz1, iz2, iz3, iz4;
int lnx, lny, lnz, gnx, gny, gnz;
double alpha, beta, alpha1, beta1, beta2;
int stage;
MPI_Datatype left_right, top_bottom, front_back;
MPI_Status *status;
MPI_Comm XYZ_COMM;
MPI_Comm YZ_COMM;
MPI_Comm R_COMM;


//用于多尺度检测的MPI

MPI_Request *send_request = (MPI_Request *)calloc(1, sizeof(MPI_Request));
MPI_Request *recv_request = (MPI_Request *)calloc(block_x*block_y*block_z-1, sizeof(MPI_Request));

MPI_Status *send_status = (MPI_Status *)calloc(1, sizeof(MPI_Status));    
MPI_Status *recv_status = (MPI_Status *)calloc(block_x*block_y*block_z-1, sizeof(MPI_Status));  


int color1, key1;
int irun, simulate_checkpoint, nchk, chk;
int counts;
//offset
size_t offset, offset_Er;
size_t u_fb, u_tb, u_lr;
size_t e_fb, e_tb, e_lr;
size_t offset2;
size_t ela_size;

// CNN path
 char meta_path[1024];
 char ckpt_path[1024] ;

//cpu variables
struct Allen_Cahn
{
  double u;
  double LE, KE;
  double lambda[3][3];
  double lambda_check[3][3];
  //double lambdar[3][3];
  //double lambdar1[3][3];
  Stype  *field2b;
  double *field2b_c;
  double *fieldE;
  double *fieldEs_left, *fieldEr_left, *fieldEs_right, *fieldEr_right;
  double *fieldEs_top, *fieldEr_top, *fieldEs_bottom, *fieldEr_bottom;
  double *fieldEr_front, *fieldEr_back;
  double *fieldEs_front, *fieldEs_back;
  MPI_Request *ireq_left_right_fieldE, *ireq_top_bottom_fieldE, *ireq_front_back_fieldE;
} * ac;
double *lambdar, *lambdar1;

  Dtype *ac_fieldE;
  Dtype *ac_fieldE_ds;        // 保存有down sample后的模拟结果
  Dtype *ac_fieldE_gathered;  // 保存有从其它进程获取到的模拟结果
  Dtype *ac_fieldE_gathered_reoder;  // 保存有从其它进程获取到的模拟结果
  int x, y, z, n, b_x, b_y, b_z, x_, y_, z_, n_, new_idx;
  Dtype dat;  // 保存有从其它进程获取到的模拟结果
  unsigned int *ac_fieldE_ds_cp;          // 保存有经过压缩后的数据
  unsigned int *ac_fieldE_ds_cp_gathered; // 保存有收集到的压缩后的数据

  int *ori, *Ori, *f, *F;

  Stype *temp_ac_fieldE;
  char detect_result_name[1024];


#endif 
