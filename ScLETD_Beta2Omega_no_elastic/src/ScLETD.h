#ifndef _SCLETD_H_
#define _SCLETD_H_

#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <mm_malloc.h>
/*#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif*/
#include <math.h>
#include <time.h>
#include "mpi.h"
//#include "hip_complex.h"
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

#define f_A (2376.0 / 2372.0) // 1
#define f_B ((7128.0 + 12.0 * 2372.0) / 2372.0) // 15
#define f_C ((4752.0 + 12.0 * 2372.0) / 2372.0) // 14
#define PI 3.141592653589793

typedef double Dtype;
typedef unsigned short int Stype;


#define DIM 3
int NX, NY, NZ;
#define NELASTIC 1
double epsilon2d[NELASTIC][DIM * DIM];
double sigma2d[NELASTIC][DIM * DIM];
double C4d[DIM * DIM * DIM * DIM];
double *Bn[NELASTIC][NELASTIC];
double *tmpy_re, *tmpy_fftre[NELASTIC], *tmpy_fftim[NELASTIC];
double *s_left, *s_right, *s_top, *s_bottom, *s_front, *s_back;
double *r_left, *r_right, *r_top, *r_bottom, *r_front, *r_back;
double *e_left, *e_right, *e_top, *e_bottom, *e_front, *e_back;
MPI_Request *ireq_left_right, *ireq_top_bottom, *ireq_front_back;

double *BN[NELASTIC][NELASTIC];
double *tmpy_RE1, *tmpy_fftRE[NELASTIC], *tmpy_fftIM[NELASTIC];
double *Elas;
double *S_left, *S_right, *S_top, *S_bottom, *S_front, *S_back;
double *R_left, *R_right, *R_top, *R_bottom, *R_front, *R_back;

double ElasticScale;
int ELASTIC;
int ioutput;
int nout;
double rad;
double percent;
int rotation;
double theta[3];
int ANISOTROPIC;
int ETD2;
//int *fieldgx, *fieldgy, *fieldgz;
int nac, nch;
int iter;
int left, right, top, bottom, front, back;
int nx, ny, nz;
char processor_name[MPI_MAX_PROCESSOR_NAME];
int  namelen;
int deviceId;
hipDeviceProp_t props;

int nprocs, procs[3], myrank, prank, cart_id[3];
int periodic;
double *MPX, *MPY, *MPZ, *MPXI, *MPYI, *MPZI, *DDX, *DDY, *DDZ;
double *MPX_b, *MPY_b, *MPZ_b, *MPXI_b, *MPYI_b, *MPZI_b, *DDX_b, *DDY_b, *DDZ_b;
double A1, A2, A3, A4, A5, C1;
double dt, t_total;
double epn2;
double xmin, ymin, zmin;
double xmax, ymax, zmax;
//new
int lr_size, tb_size, fb_size;

double hx, hy, hz;
double kkx, kky, kkz;
int restart;
char work_dir[1024];
char data_dir[1024];
int restart_iter;
char restart_dir[1024];
int ini_num;
int nghost;
int ix1, ix2, ix3, ix4, iy1, iy2, iy3, iy4, iz1, iz2, iz3, iz4;
int lnx, lny, lnz, gnx, gny, gnz;
double alpha, beta;
int stage;
MPI_Datatype left_right, top_bottom, front_back;
MPI_Status *status;
MPI_Comm XYZ_COMM;
MPI_Comm YZ_COMM;
MPI_Comm R_COMM;
int color1, key1;
int irun, checkpoint,nchk, chk;
int counts;
//offset
size_t offset, offset_Er;
size_t u_fb, u_tb, u_lr;
size_t e_fb, e_tb, e_lr;
size_t offset2;
//offset of 64 fft
size_t offset3;
int elas_x, elas_y, elas_z;
//time
hipEvent_t st, ed;
hipEvent_t st2, ed2;
//double functime, runtime, walltime;
float elastic_copyin_time;
float skipin_copyin_time;
float elastic_copyout_time;
float elastic_Memcpy_time;
float elastic_multiply_BN_time;
float trilinear_interpolation_time;
float fft_forward_A_time, fft_forward_B_time, fft_forward_C_time, fft_forward_D_time;
float dgemm_time, copy_time, F1_time, F2_time, fieldE1_time, Tu_time, updateU_new_time, calc_time;
float forward_mpi_time, backward_mpi_time, elastic_calculate_time, elastic_transfer_time;
float trans_unpack_time;
float trans_mu_time;
float trans_enlarge_time;
float trans_pack_time;
float trans_Memcpy_time;
float trans_MPI_time;
float trans_time;
float timer;
float timer2;

//cpu variables
struct Allen_Cahn
{
  double u;
  double LE, KE;
  double lambda[3][3];
  double lambda_check[3][3];
  double lambdar[3][3];
  double lambdar1[3][3];
  Stype *field2b;
  double *fielde, *fieldE;
  double *fieldEs_left, *fieldEr_left, *fieldEs_right, *fieldEr_right;
  double *fieldEs_top, *fieldEr_top, *fieldEs_bottom, *fieldEr_bottom;
  double *fieldEr_front, *fieldEr_back;
  double *fieldEs_front, *fieldEs_back;
  MPI_Request *ireq_left_right_fieldE, *ireq_top_bottom_fieldE, *ireq_front_back_fieldE;
} * ac;

//DCU variables
  double *fieldEr;
  Stype *field2B;
  double *fieldE, *fielde;
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
  Dtype *phiE;
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

void init_KL();
double EF();
double AC_f(int n);
double CH_f(int n);
void write_chk(void);
void write_field2B(int irun);
void read_chk(void);
// L(-F(C,E)+KC)
#define SWTICH_AC_FIELD1(f)                              \
  {                                                      \
    for (m = 0; m < nac; m++)                            \
    {                                                    \
      ac[m].u = ac[m].fieldE[k * nx * ny + j * nx + i];  \
    }                                                    \
/*    for (m = 0; m < nch; m++)                            \
    {                                                    \
      ch[m].c = ch[m].fieldCI[k * nx * ny + j * nx + i]; \
    }*/                                                    \
    f = ac[n].LE * ac[n].KE * ac[n].u;                   \
    f -= ac[n].LE * AC_f(n);                             \
  }

// L(F(C,E)-KC)
/*#define SWITCH_CH_FIELD1(f)                              \
  {                                                      \
    for (m = 0; m < nac; m++)                            \
    {                                                    \
      ac[m].u = ac[m].fieldE[k * nx * ny + j * nx + i];  \
    }                                                    \
    for (m = 0; m < nch; m++)                            \
    {                                                    \
      ch[m].c = ch[m].fieldCI[k * nx * ny + j * nx + i]; \
    }                                                    \
    f = -ch[n].LCI * ch[n].KCI * ch[n].c;                \
    f += ch[n].LCI * CH_f(n);                            \
  }
*/
/*#define NAME(prefix, suffix) prefix##suffix
// ? - L(F(C,E))
#define SWITCH_CH_FIELDMU(suffix)                               \
  {                                                             \
    for (m = 0; m < nac; m++)                                   \
    {                                                           \
      ac[m].u = ac[m].NAME(fieldEe_, suffix)[l_e];              \
    }                                                           \
    for (m = 0; m < nch; m++)                                   \
    {                                                           \
      ch[m].c = ch[m].NAME(fieldCIe_, suffix)[l_e];             \
    }                                                           \
    NAME(fieldmu_, suffix)                                      \
    [l_mu] = ch[n].LCI * epn2 * NAME(fieldmu_, suffix)[l_mu];   \
    NAME(fieldmu_, suffix)                                      \
    [l_mu] -= ch[n].LCI * (CH_f(n));                            \
    if (ELASTIC == CH_FUNCTION)                                 \
    {                                                           \
      NAME(fieldmu_, suffix)                                    \
      [l_mu] -= ch[n].LCI * (ch[n].NAME(felase_, suffix)[l_e]); \
    }                                                           \
  }*/
#endif 
