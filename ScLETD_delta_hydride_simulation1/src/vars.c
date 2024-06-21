#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include "mpi.h"
#include "ScLETD.h"
#include "anisotropic_hip.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

/*
fieldgx, fieldgy, fieldgz are global coordinate
of field variable tensor points
*/
void
global_index (void)
{
  int i, j, k, gx, gy, gz;
  double x, y, z, cx, cy, cz;

  if (cart_id[0] == 0) {
    gx = 0;
  }
  else {
    gx = cart_id[0] * nx - 2 * nghost * cart_id[0];
  }
  if (cart_id[1] == 0) {
    gy = 0;
  }
  else {
    gy = cart_id[1] * ny - 2 * nghost * cart_id[1];
  }
  if (cart_id[2] == 0) {
    gz = 0;
  }
  else {
    gz = cart_id[2] * nz - 2 * nghost * cart_id[2];
  }
  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
        fieldgx[k * nx * ny + j * nx + i] = gx + i;
        fieldgy[k * nx * ny + j * nx + i] = gy + j;
        fieldgz[k * nx * ny + j * nx + i] = gz + k;
      }
    }
  }
}

// initialize compute range and control range of subdomain
// ix1,ix2, .. ,ix3,ix4
// iy1,iy2, .. ,iy3,iy4
// iz1,iz2, .. ,iz3,iz4
void
init_para (void)
{
  if (left < 0) {
    ix1 = 0;
    ix2 = ix1;
    ix3 = ix1 + nx - nghost;
    ix4 = nx;
  }
  else if (right < 0) {
    ix1 = 0;
    ix2 = ix1 + nghost;
    ix3 = ix1 + nx;
    ix4 = nx;
  }
  else {
    ix1 = 0;
    ix2 = ix1 + nghost;
    ix3 = ix1 + nx - nghost;
    ix4 = nx;
  }
  lnx = ix4 - ix1;
  if (periodic) {
    gnx = lnx * procs[0] - 2 * nghost * procs[0];
  } else {
    gnx = lnx * procs[0] - 2 * nghost * (procs[0] - 1);
  }

  if (top < 0) {
    iy1 = 0;
    iy2 = iy1;
    iy3 = iy1 + ny - nghost;
    iy4 = ny;
  }
  else if (bottom < 0) {
    iy1 = 0;
    iy2 = iy1 + nghost;
    iy3 = iy1 + ny;
    iy4 = ny;
  }
  else {
    iy1 = 0;
    iy2 = iy1 + nghost;
    iy3 = iy1 + ny - nghost;
    iy4 = ny;
  }
  lny = iy4 - iy1;
  if (periodic) {
    gny = lny * procs[0] - 2 * nghost * procs[0];
  } else {
    gny = lny * procs[0] - 2 * nghost * (procs[0] - 1);
  }

  if (front < 0) {
    iz1 = 0;
    iz2 = iz1;
    iz3 = iz1 + nz - nghost;
    iz4 = nz;
  }
  else if (back < 0) {
    iz1 = 0;
    iz2 = iz1 + nghost;
    iz3 = iz1 + nz;
    iz4 = nz;
  }
  else {
    iz1 = 0;
    iz2 = iz1 + nghost;
    iz3 = iz1 + nz - nghost;
    iz4 = nz;
  }
  lnz = iz4 - iz1;
  if (periodic) {
    gnz = lnz * procs[0] - 2 * nghost * procs[0];
  } else {
    gnz = lnz * procs[0] - 2 * nghost * (procs[0] - 1);
  }
  global_index();
}




void
init_vars (void)
{
  alpha = 1.0;
  beta = 0.0;
  alpha1 = -1.0;
  beta1 = -1.0;
  beta2 = 1.0;
  init_KL();
  offset = nx * ny * nz;
  ela_size = Approx * Approx * Approx;
  Gx_m = nx - 2 * nghost;
  Gx_n = (ny - 2 * nghost) * (nz - 2 * nghost);
  Gx_k = Approx;
  Gy_m = ny - 2 * nghost;
  Gy_n = (nz - 2 * nghost) * Approx;
  Gy_k = Approx;
  Gz_m = nz - 2 * nghost;
  Gz_n = Approx * Approx;
  Gz_k = Approx;
  conv_x_m = (nx - 2 * nghost) + Approx - 1;
  conv_x_n = Approx * Approx;
  conv_x_k = Approx;
  conv_y_m = (ny - 2 * nghost) + Approx - 1;
  conv_y_n = Approx * ((nx - 2 * nghost) + Approx - 1);
  conv_y_k = Approx;
  conv_z_m = (nz - 2 * nghost) + Approx - 1;
  conv_z_n = ((ny - 2 * nghost) + Approx - 1) * ((nz - 2 * nghost) + Approx - 1);
  conv_z_k = Approx;
  conv_big_x_m = (nx - 2 * nghost) + Approx - 1;
  conv_big_x_n = ((ny - 2 * nghost) + Approx - 1) * ((nz - 2 * nghost) + Approx - 1);
  conv_big_x_k = (nx - 2 * nghost) + Approx - 1;
  conv_big_y_m = (ny - 2 * nghost) + Approx - 1;
  conv_big_y_n = ((ny - 2 * nghost) + Approx - 1) * ((nz - 2 * nghost) + Approx - 1);
  conv_big_y_k = (ny - 2 * nghost) + Approx - 1;
  conv_big_z_m = (nz - 2 * nghost) + Approx - 1;
  conv_big_z_n = ((ny - 2 * nghost) + Approx - 1) * ((nz - 2 * nghost) + Approx - 1);
  conv_big_z_k = (nz - 2 * nghost) + Approx - 1;
  conv_lr_size = Approx * (ny - 2 * nghost + Approx * 2) * (nz - 2 * nghost + Approx * 2);
  conv_tb_size = (nx - 2 * nghost) * Approx * (nz - 2 * nghost + Approx * 2);
  conv_fb_size = (nx - 2 * nghost) * (ny - 2 * nghost) * Approx;
}

void
alloc_vars (void)
{
  NX = nx - 2 * nghost;
  NY = ny - 2 * nghost;
  NZ = nz - 2 * nghost;
  int n;
  // global coordinate
  fieldgx = (int *) _mm_malloc (sizeof (int) * nx * ny * nz, 256);
  fieldgy = (int *) _mm_malloc (sizeof (int) * nx * ny * nz, 256);
  fieldgz = (int *) _mm_malloc (sizeof (int) * nx * ny * nz, 256);
  // Allen-Cahn function
  ac = (struct Allen_Cahn *) _mm_malloc (nac * sizeof (struct Allen_Cahn), 256);
  for (n = 0; n < nac; n++) {
    // field variable
    ac[n].fieldE = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ac[n].fieldE1 = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ac[n].fieldE2 = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ac[n].fieldEt = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ac[n].fieldEp = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ac[n].fieldE1p = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ac[n].fieldEr = (double *) _mm_malloc (sizeof (double) * (nx+2*2) * (ny+2*2) * (nz+2*2), 256);
    // send or receive buffer of ghost
    ac[n].fieldEs_left = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEr_left = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEs_right = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEr_right = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEs_top = (double *) _mm_malloc (sizeof (double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEr_top = (double *) _mm_malloc (sizeof (double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEs_bottom = (double *) _mm_malloc (sizeof (double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEr_bottom = (double *) _mm_malloc (sizeof (double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEr_front = (double *) _mm_malloc (sizeof (double) * nx * ny * (nghost + 2), 256);
    ac[n].fieldEr_back = (double *) _mm_malloc (sizeof (double) * nx * ny * (nghost + 2), 256);
    // enlarge buffer of receive
    ac[n].fieldEe_left = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + 4) * (nz + 4), 256);
    ac[n].fieldEe_right = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + 4) * (nz + 4), 256);
    ac[n].fieldEe_top = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (nghost + 2) * (nz + 4), 256);
    ac[n].fieldEe_bottom = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (nghost + 2) * (nz + 4), 256);
    ac[n].fieldEe_front = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (ny + 4) * (nghost + 2), 256);
    ac[n].fieldEe_back = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (ny + 4) * (nghost + 2), 256);
    // boundary condition of 1 order
    ac[n].fieldEu_left = (double *) _mm_malloc (sizeof (double) * ny * nz, 256);
    ac[n].fieldEu_right = (double *) _mm_malloc (sizeof (double) * ny * nz, 256);
    ac[n].fieldEu_top = (double *) _mm_malloc (sizeof (double) * nx * nz, 256);
    ac[n].fieldEu_bottom = (double *) _mm_malloc (sizeof (double) * nx * nz, 256);
    ac[n].fieldEu_front = (double *) _mm_malloc (sizeof (double) * nx * ny, 256);
    ac[n].fieldEu_back = (double *) _mm_malloc (sizeof (double) * nx * ny, 256);
    // boundary condition of 2 order
    ac[n].fieldEmu_left = (double *) _mm_malloc (sizeof (double) * ny * nz, 256);
    ac[n].fieldEmu_right = (double *) _mm_malloc (sizeof (double) * ny * nz, 256);
    ac[n].fieldEmu_top = (double *) _mm_malloc (sizeof (double) * nx * nz, 256);
    ac[n].fieldEmu_bottom = (double *) _mm_malloc (sizeof (double) * nx * nz, 256);
    ac[n].fieldEmu_front = (double *) _mm_malloc (sizeof (double) * nx * ny, 256);
    ac[n].fieldEmu_back = (double *) _mm_malloc (sizeof (double) * nx * ny, 256);
    // elastic variable
    ac[n].felas = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ac[n].felase_left = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + 4) * (nz + 4), 256);
    ac[n].felase_right = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + 4) * (nz + 4), 256);
    ac[n].felase_top = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (nghost + 2) * (nz + 4), 256);
    ac[n].felase_bottom = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (nghost + 2) * (nz + 4), 256);
    ac[n].felase_front = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (ny + 4) * (nghost + 2), 256);
    ac[n].felase_back = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (ny + 4) * (nghost + 2), 256);
    // Interpolation approximation
    ac[n].phiE = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ac[n].phiE2 = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    // mpi communication
    ac[n].ireq_left_right_fieldE = (MPI_Request *) calloc (4, sizeof (MPI_Request));
    ac[n].ireq_top_bottom_fieldE = (MPI_Request *) calloc (4, sizeof (MPI_Request));
    ac[n].ireq_front_back_fieldE = (MPI_Request *) calloc (4, sizeof (MPI_Request));

    ac[n].ireq_left_right_felas = (MPI_Request *) calloc (4, sizeof (MPI_Request));
    ac[n].ireq_top_bottom_felas = (MPI_Request *) calloc (4, sizeof (MPI_Request));
    ac[n].ireq_front_back_felas = (MPI_Request *) calloc (4, sizeof (MPI_Request));

    // gradient tmp variable
    ac[n].gradx = (double *)_mm_malloc(sizeof(double) * nx * ny * nz, 256);
    ac[n].grady = (double *)_mm_malloc(sizeof(double) * nx * ny * nz, 256);
    ac[n].gradz = (double *)_mm_malloc(sizeof(double) * nx * ny * nz, 256);
    // nolinear term variable
    ac[n].f1 = (double *)_mm_malloc(sizeof(double) * nx * ny * nz, 256);
    ac[n].f2 = (double *)_mm_malloc(sizeof(double) * nx * ny * nz, 256);
    ac[n].f3 = (double *)_mm_malloc(sizeof(double) * nx * ny * nz, 256);

  // elatic variable
  ac[n].Bn = (double *)_mm_malloc(sizeof(double) * NX * NY * NZ * nac, 256);
  ac[n].elas_re = (double *)_mm_malloc(sizeof(double) * NX * NY * NZ, 256);
  ac[n].elas_im = (double *)_mm_malloc(sizeof(double) * NX * NY * NZ, 256);
  ac[n].theta_re = (double *)_mm_malloc(sizeof(double) * NX * NY * NZ, 256);
  ac[n].theta_im = (double *)_mm_malloc(sizeof(double) * NX * NY * NZ, 256);
  ac[n].elas = (double *)_mm_malloc(sizeof(double) * NX * NY * NZ, 256);
  ac[n].elas_field = (double *)_mm_malloc(sizeof(double) * NX * NY * NZ, 256);
  }

  tmpy_fftre = (double *)_mm_malloc(sizeof(double) * Approx * Approx * Approx * 8, 256);
  fftre = (double *)_mm_malloc(sizeof(double) * Approx * Approx * Approx * 8, 256);
  conv_r_right = (double *)_mm_malloc(sizeof(double) * Approx * (NY + 2 * Approx) * (NZ + 2 * Approx), 256);
  conv_r_bottom = (double *)_mm_malloc(sizeof(double) * NX * Approx * (NZ + 2 * Approx), 256);
  conv_r_back = (double *)_mm_malloc(sizeof(double) * NX * NY * Approx, 256);
  conv_s_right = (double *)_mm_malloc(sizeof(double) * Approx * (NY + 2 * Approx) * (NZ + 2 * Approx), 256);
  conv_s_bottom = (double *)_mm_malloc(sizeof(double) * NX * Approx * (NZ + 2 * Approx), 256);
  conv_s_back = (double *)_mm_malloc(sizeof(double) * NX * NY * Approx, 256);
  conv_s_left = (double *)_mm_malloc(sizeof(double) * Approx * (NY + 2 * Approx) * (NZ + 2 * Approx), 256);
  conv_s_top = (double *)_mm_malloc(sizeof(double) * NX * Approx * (NZ + 2 * Approx), 256);
  conv_s_front = (double *)_mm_malloc(sizeof(double) * NX * NY * Approx, 256);
  conv_r_left = (double *)_mm_malloc(sizeof(double) * Approx * (NY + 2 * Approx) * (NZ + 2 * Approx), 256);
  conv_r_top = (double *)_mm_malloc(sizeof(double) * NX * Approx * (NZ + 2 * Approx), 256);
  conv_r_front = (double *)_mm_malloc(sizeof(double) * NX * NY * Approx, 256);
  conv_ireq_left_right = (MPI_Request *)calloc(4, sizeof(MPI_Request));
  conv_ireq_top_bottom = (MPI_Request *)calloc(4, sizeof(MPI_Request));
  conv_ireq_front_back = (MPI_Request *)calloc(4, sizeof(MPI_Request));
  rocblas_create_handle(&handle);
  hipMalloc (&tmpy_fftRE, Approx * Approx * Approx * 8 * sizeof(double));
  hipMalloc (&fftRE, Approx * Approx * Approx * 8 * sizeof(double));
  hipMalloc (&BN, NX * NY * NZ * sizeof(double));
  hipMalloc (&Elas, nx * ny * nz * sizeof(double));
  hipMalloc (&tmpy_RE1, NX * NY * NZ*sizeof(double));
  hipMalloc (&tmpy_RE2, NX * NY * NZ*sizeof(double));
  hipMalloc (&Gx_re, NX * Approx  * sizeof(double));
  hipMalloc (&Gx_im, NX * Approx  * sizeof(double));
  hipMalloc (&Gy_re, NY * Approx  * sizeof(double));
  hipMalloc (&Gy_im, NY * Approx  * sizeof(double));
  hipMalloc (&Gz_re, NZ * Approx  * sizeof(double));
  hipMalloc (&Gz_im, NZ * Approx  * sizeof(double));
  hipMalloc (&temp1, Approx * NY * NZ * sizeof(double));
  hipMalloc (&temp2, Approx * NY * NZ * sizeof(double));
  hipMalloc (&temp3, Approx * Approx * NZ * sizeof(double));
  hipMalloc (&temp4, Approx * Approx * NZ * sizeof(double));
  hipMalloc (&temp5, Approx * Approx * NZ * sizeof(double));
  hipMalloc (&temp6, Approx * Approx * NZ * sizeof(double));
  hipMalloc (&temp7, Approx * Approx * Approx * sizeof(double));
  hipMalloc (&temp8, Approx * Approx * Approx * sizeof(double));
  hipMalloc (&Itemp1, Approx * NY * NZ  * sizeof(double));
  hipMalloc (&Itemp2, Approx * NY * NZ  * sizeof(double));
  hipMalloc (&Itemp3, NX * NY * NZ * sizeof(double));
  hipMalloc (&Itemp4, NX * NY * NZ * sizeof(double));
  hipMalloc (&conv_S_left, Approx * (NY + Approx * 2) * (NZ + Approx * 2) * sizeof (double));
  hipMalloc (&conv_R_right, Approx * (NY + Approx * 2) * (NZ + Approx * 2) * sizeof (double));
  hipMalloc (&conv_R_left, Approx * (NY + Approx * 2) * (NZ + Approx * 2) * sizeof (double));
  hipMalloc (&conv_S_right, Approx * (NY + Approx * 2) * (NZ + Approx * 2) * sizeof (double));
  hipMalloc (&conv_S_top, NX * Approx * (NZ + Approx * 2) * sizeof (double));
  hipMalloc (&conv_R_bottom, NX * Approx * (NZ + Approx * 2) * sizeof (double));
  hipMalloc (&conv_R_top, NX * Approx * (NZ + Approx * 2) * sizeof (double));
  hipMalloc (&conv_S_bottom, NX * Approx * (NZ + Approx * 2) * sizeof (double));
  hipMalloc (&conv_S_front, NX * NY * Approx* sizeof (double));
  hipMalloc (&conv_R_back, NX * NY * Approx* sizeof (double));
  hipMalloc (&conv_R_front, NX * NY * Approx* sizeof (double));
  hipMalloc (&conv_S_back, NX * NY * Approx* sizeof (double));
  hipMalloc (&conv_big_x_re, (NX + Approx - 1) * (NX + Approx - 1) * sizeof(double));
  hipMalloc (&conv_big_x_im, (NX + Approx - 1) * (NX + Approx - 1) * sizeof(double));
  hipMalloc (&conv_big_y_re, (NX + Approx - 1) * (NX + Approx - 1) * sizeof(double));
  hipMalloc (&conv_big_y_im, (NX + Approx - 1) * (NX + Approx - 1) * sizeof(double));
  hipMalloc (&conv_big_z_re, (NX + Approx - 1) * (NX + Approx - 1) * sizeof(double));
  hipMalloc (&conv_big_z_im, (NX + Approx - 1) * (NX + Approx - 1) * sizeof(double));
  hipMalloc (&conv_x_re, (NX + Approx - 1) * Approx * sizeof(double));
  hipMalloc (&conv_x_im, (NX + Approx - 1) * Approx * sizeof(double));
  hipMalloc (&conv_y_re, (NX + Approx - 1) * Approx * sizeof(double));
  hipMalloc (&conv_y_im, (NX + Approx - 1) * Approx * sizeof(double));
  hipMalloc (&conv_z_re, (NX + Approx - 1) * Approx * sizeof(double));
  hipMalloc (&conv_z_im, (NX + Approx - 1) * Approx * sizeof(double));
  hipMalloc (&conv_temp1, Approx * (NY + Approx - 1) * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp2, Approx * (NY + Approx - 1) * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp3, Approx * (NY + Approx - 1) * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp4, Approx * (NY + Approx - 1) * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp5, Approx * Approx * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp6, Approx * Approx * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp7, Approx * Approx * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp8, Approx * Approx * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp9, (NX + Approx - 1)* (NY + Approx - 1) * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp10, (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp11, (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&conv_temp12, (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1) * sizeof (double));
  hipMalloc (&BN_RE, (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1) * 8 * sizeof (double));
  hipMalloc (&BN_IM, (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1) * 8 * sizeof (double));
  hipMalloc (&eta_RE, (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1) * 8 * sizeof (double));
  hipMalloc (&eta_IM, (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1) * 8 * sizeof (double));
  hipMalloc (&bn_RE, (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1) * nac * sizeof (double));
  hipMalloc (&bn_IM, (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1) * nac * sizeof (double));
  hipMalloc (&conv_e, (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1) * sizeof (double));
  hipMemset(conv_e, 0.0, sizeof(double) * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1));
// hipMalloc (&conv_e, (NX + Approx * 2) * (NY + Approx * 2) * (NZ + Approx * 2) * sizeof (double));
// hipMemset(conv_e, 0.0, sizeof(double) * (NX + Approx * 2) * (NY + Approx * 2) * (NZ + Approx * 2));
  hipMalloc(&fieldE, nac * nx * ny * nz * sizeof(Dtype));

  // Cahn-Hilliard function
  ch = (struct Cahn_Hilliard *) _mm_malloc (nch * sizeof (struct Cahn_Hilliard), 256);
  for (n = 0; n < nch; n++) {
    // field variable
    ch[n].fieldCI = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].fieldCI1 = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].fieldCI2 = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].fieldCIt = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].fieldCIp = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].fieldCI1p = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].fieldCIr = (double *) _mm_malloc (sizeof (double) * (nx+2*2) * (ny+2*2) * (nz+2*2), 256);
    // c_alpha and c_delta enlarge field variable
    ch[n].c_alpha_r = (double *) _mm_malloc (sizeof (double) * (nx+2*2) * (ny+2*2) * (nz+2*2), 256);
    ch[n].c_delta_r = (double *) _mm_malloc (sizeof (double) * (nx+2*2) * (ny+2*2) * (nz+2*2), 256);
    // send or receive buffer of ghost
    ch[n].fieldCIr = (double *) _mm_malloc (sizeof (double) * (nx+2*2) * (ny+2*2) * (nz+2*2), 256);
    ch[n].fieldCIs_left = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ch[n].fieldCIr_left = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ch[n].fieldCIs_right = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ch[n].fieldCIr_right = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ch[n].fieldCIs_top = (double *) _mm_malloc (sizeof (double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ch[n].fieldCIr_top = (double *) _mm_malloc (sizeof (double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ch[n].fieldCIs_bottom = (double *) _mm_malloc (sizeof (double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ch[n].fieldCIr_bottom = (double *) _mm_malloc (sizeof (double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ch[n].fieldCIr_front = (double *) _mm_malloc (sizeof (double) * nx * ny * (nghost + 2), 256);
    ch[n].fieldCIr_back = (double *) _mm_malloc (sizeof (double) * nx * ny * (nghost + 2), 256);
    // enlarge buffer
    ch[n].fieldCIe_left = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + 4) * (nz + 4), 256);
    ch[n].fieldCIe_right = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + 4) * (nz + 4), 256);
    ch[n].fieldCIe_top = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (nghost + 2) * (nz + 4), 256);
    ch[n].fieldCIe_bottom = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (nghost + 2) * (nz + 4), 256);
    ch[n].fieldCIe_front = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (ny + 4) * (nghost + 2), 256);
    ch[n].fieldCIe_back = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (ny + 4) * (nghost + 2), 256);
    // boundary condition 1 order laplace
    ch[n].fieldCIu_left = (double *) _mm_malloc (sizeof (double) * ny * nz, 256);
    ch[n].fieldCIu_right = (double *) _mm_malloc (sizeof (double) * ny * nz, 256);
    ch[n].fieldCIu_top = (double *) _mm_malloc (sizeof (double) * nx * nz, 256);
    ch[n].fieldCIu_bottom = (double *) _mm_malloc (sizeof (double) * nx * nz, 256);
    ch[n].fieldCIu_front = (double *) _mm_malloc (sizeof (double) * nx * ny, 256);
    ch[n].fieldCIu_back = (double *) _mm_malloc (sizeof (double) * nx * ny, 256);

    // boundary condition 2 order laplace
    ch[n].fieldCImu_left = (double *) _mm_malloc (sizeof (double) * ny * nz, 256);
    ch[n].fieldCImu_right = (double *) _mm_malloc (sizeof (double) * ny * nz, 256);
    ch[n].fieldCImu_top = (double *) _mm_malloc (sizeof (double) * nx * nz, 256);
    ch[n].fieldCImu_bottom = (double *) _mm_malloc (sizeof (double) * nx * nz, 256);
    ch[n].fieldCImu_front = (double *) _mm_malloc (sizeof (double) * nx * ny, 256);
    ch[n].fieldCImu_back = (double *) _mm_malloc (sizeof (double) * nx * ny, 256);

    // elastic variables
    ch[n].felas = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].felase_left = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + 4) * (nz + 4), 256);
    ch[n].felase_right = (double *) _mm_malloc (sizeof (double) * (nghost + 2) * (ny + 4) * (nz + 4), 256);
    ch[n].felase_top = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (nghost + 2) * (nz + 4), 256);
    ch[n].felase_bottom = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (nghost + 2) * (nz + 4), 256);
    ch[n].felase_front = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (ny + 4) * (nghost + 2), 256);
    ch[n].felase_back = (double *) _mm_malloc (sizeof (double) * (nx + 4) * (ny + 4) * (nghost + 2), 256);

    // Interpolation approximation
    ch[n].phiCI = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].phiCI2 = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);

    ch[n].ireq_left_right_fieldCI = (MPI_Request *) calloc (4, sizeof (MPI_Request));
    ch[n].ireq_top_bottom_fieldCI = (MPI_Request *) calloc (4, sizeof (MPI_Request));
    ch[n].ireq_front_back_fieldCI = (MPI_Request *) calloc (4, sizeof (MPI_Request));
    ch[n].ireq_left_right_felas = (MPI_Request *) calloc (4, sizeof (MPI_Request));
    ch[n].ireq_top_bottom_felas = (MPI_Request *) calloc (4, sizeof (MPI_Request));
    ch[n].ireq_front_back_felas = (MPI_Request *) calloc (4, sizeof (MPI_Request));

    // c_alpha and c_delta field varialbes
    ch[n].c_alpha = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].c_delta = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].ft = (double *) _mm_malloc (sizeof (double) * nx * ny * nz, 256);
    ch[n].ftr = (double *) _mm_malloc (sizeof (double) * (nx+2*2) * (ny+2*2) * (nz+2*2), 256);
  }
  MPX = (double *) _mm_malloc (sizeof (double) * nx * nx, 256);
  MPY = (double *) _mm_malloc (sizeof (double) * ny * ny, 256);
  MPZ = (double *) _mm_malloc (sizeof (double) * nz * nz, 256);
  MPXI = (double *) _mm_malloc (sizeof (double) * nx * nx, 256);
  MPYI = (double *) _mm_malloc (sizeof (double) * ny * ny, 256);
  MPZI = (double *) _mm_malloc (sizeof (double) * nz * nz, 256);
  DDX = (double *) _mm_malloc (sizeof (double) * nx, 256);
  DDY = (double *) _mm_malloc (sizeof (double) * ny, 256);
  DDZ = (double *) _mm_malloc (sizeof (double) * nz, 256);
  MPX_b = (double *) _mm_malloc (sizeof (double) * nx * nx * 4, 256);
  MPY_b = (double *) _mm_malloc (sizeof (double) * ny * ny * 4, 256);
  MPZ_b = (double *) _mm_malloc (sizeof (double) * nz * nz * 4, 256);
  MPXI_b = (double *) _mm_malloc (sizeof (double) * nx * nx * 4, 256);
  MPYI_b = (double *) _mm_malloc (sizeof (double) * ny * ny * 4, 256);
  MPZI_b = (double *) _mm_malloc (sizeof (double) * nz * nz * 4, 256);
  DDX_b = (double *) _mm_malloc (sizeof (double) * nx * 4, 256);
  DDY_b = (double *) _mm_malloc (sizeof (double) * ny * 4, 256);
  DDZ_b = (double *) _mm_malloc (sizeof (double) * nz * 4, 256);
  status = (MPI_Status *) calloc (4, sizeof (MPI_Status));
}

// initialize exp operator
void
ac_init_phi(double epn2, double LE, double KE, double *phi, double *phi2) {
  int i, j, k, l;
  double Hijk, Gijk, tmp;
  for (j = iy1; j < iy4; j++) {
    for (i = ix1; i < ix4; i++) {
      for (k = iz1; k < iz4; k++) {
        l = j * nz * nx + i * nz + k;
        tmp = kkz * DDZ[k] + kky * DDY[j] + kkx * DDX[i];
        Gijk = tmp;
        Hijk = -LE * (tmp * epn2 - KE);
        if (fabs(Hijk) > 1.0e-8) {
          tmp = exp (-dt * Hijk);
          phi[l] = (1.0 - tmp) / Hijk;
          phi2[l] = (1.0 - phi[l] / dt) / Hijk;
        }
        else {
          phi[l] = dt;
          phi2[l] = dt / 2;
        }
      }
    }
  }
}

// initialize exp operator
void
ch_init_phi(double epn2, double LCI, double KCI, double *phi, double *phi2) {
  int i, j, k, l;
  double Hijk, Gijk, tmp;
  for (j = iy1; j < iy4; j++) {
    for (i = ix1; i < ix4; i++) {
      for (k = iz1; k < iz4; k++) {
        l = j * nz * nx + i * nz + k;
        tmp = DDZ[k] + DDY[j] + DDX[i];
        Hijk = -LCI * (tmp * epn2 - KCI);
        if (fabs(Hijk) > 1.0e-8) {
          tmp = exp (-dt * Hijk);
          phi[l] = (1.0 - tmp) / Hijk;
          phi2[l] = (1.0 - phi[l] / dt) / Hijk;
        }
        else {
          phi[l] = dt;
          phi2[l] = dt / 2;
        }
      }
    }
  }
}

// read foriour and laplace matrix
void
read_matrices (void)
{
  int n;
  char filename[1024];
  int i, j, k, l, id, mp_ofst, d_ofst;
  FILE *file;

  if (cart_id[1] == 0 && cart_id[2] == 0) { 
    for (k = 0; k < 4; k++) {
      mp_ofst = k * nx * nx;
      d_ofst = k * nx;

      sprintf (filename, "d%d%d.dat", k, nx);
      file = fopen (filename, "r");
      for (l = 0; l < nx; l++) {
        fscanf (file, "%lf", DDX_b + d_ofst + l);
        DDX_b[d_ofst + l] = DDX_b[d_ofst + l];
      }
      fclose (file);

      sprintf (filename, "v%d%d.dat", k, nx);
      file = fopen (filename, "r");
      for (j = 0; j < nx; j++) {
        for (i = 0; i < nx; i++) {
          l = i * nx + j;
          fscanf (file, "%lf", MPX_b + mp_ofst + l);
        }
      }
      fclose (file);

      sprintf (filename, "vi%d%d.dat", k, nx);
      file = fopen (filename, "r");
      for (j = 0; j < nx; j++) {
        for (i = 0; i < nx; i++) {
          l = i * nx + j;
          fscanf (file, "%lf", MPXI_b + mp_ofst + l);
        }
      }
      fclose (file);

      mp_ofst = k * ny * ny;
      d_ofst = k * ny;

      sprintf (filename, "d%d%d.dat", k, ny);
      file = fopen (filename, "r");
      for (l = 0; l < ny; l++) {
        fscanf (file, "%lf", DDY_b + d_ofst + l);
      }
      fclose (file);

      sprintf (filename, "v%d%d.dat", k, ny);
      file = fopen (filename, "r");
      for (j = 0; j < ny; j++) {
        for (i = 0; i < ny; i++) {
          l = i * ny + j;
          fscanf (file, "%lf", MPY_b + mp_ofst + l);
        }
      }
      fclose (file);

      sprintf (filename, "vi%d%d.dat", k, ny);
      file = fopen (filename, "r");
      for (j = 0; j < ny; j++) {
        for (i = 0; i < ny; i++) {
          l = i * ny + j;
          fscanf (file, "%lf", MPYI_b + mp_ofst + l);
        }
      }
      fclose (file);

      mp_ofst = k * nz * nz;
      d_ofst = k * nz;

      sprintf (filename, "d%d%d.dat", k, nz);
      file = fopen (filename, "r");
      for (l = 0; l < nz; l++) {
        fscanf (file, "%lf", DDZ_b + d_ofst + l);
      }
      fclose (file);

      sprintf (filename, "v%d%d.dat", k, nz);
      file = fopen (filename, "r");
      for (j = 0; j < nz; j++) {
        for (i = 0; i < nz; i++) {
          l = i * nz + j;
          fscanf (file, "%lf", MPZ_b + mp_ofst + l);
        }
      }
      fclose (file);

      sprintf (filename, "vi%d%d.dat", k, nz);
      file = fopen (filename, "r");
      for (j = 0; j < nz; j++) {
        for (i = 0; i < nz; i++) {
          l = i * nz + j;
          fscanf (file, "%lf", MPZI_b + mp_ofst + l);
        }
      }
      fclose (file);
    }
  }

  MPI_Bcast (&DDX_b[0], nx * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast (&MPX_b[0], nx * nx * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast (&MPXI_b[0], nx * nx * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast (&DDY_b[0], ny * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast (&MPY_b[0], ny * ny * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast (&MPYI_b[0], ny * ny * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast (&DDZ_b[0], nz * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast (&MPZ_b[0], nz * nz * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast (&MPZI_b[0], nz * nz * 4, MPI_DOUBLE, 0, YZ_COMM);

  if (left < 0) {
    k = 1;
  }
  else {
    k = 2;
  }

  if (right < 0) {
    l = 1;
  }
  else {
    l = 2;
  }

  id = (k - 1) + (l - 1) * 2;
  mp_ofst = id * nx * nx;
  d_ofst = id * nx;

  for (l = 0; l < nx; l++) {
    DDX[l] = DDX_b[d_ofst + l] / hx / hx;
  }
  for (l = 0; l < nx * nx; l++) {
    MPX[l] = MPX_b[mp_ofst + l];
    MPXI[l] = MPXI_b[mp_ofst + l];
  }

  if (top < 0) {
    k = 1;
  }
  else {
    k = 2;
  }

  if (bottom < 0) {
    l = 1;
  }
  else {
    l = 2;
  }

  id = (k - 1) + (l - 1) * 2;
  mp_ofst = id * ny * ny;
  d_ofst = id * ny;

  for (l = 0; l < ny; l++) {
    DDY[l] = DDY_b[d_ofst + l] / hy / hy;
  }
  for (l = 0; l < ny * ny; l++) {
    MPY[l] = MPY_b[mp_ofst + l];
    MPYI[l] = MPYI_b[mp_ofst + l];
  }

  if (front < 0) {
    k = 1;
  }
  else {
    k = 2;
  }

  if (back < 0) {
    l = 1;
  }
  else {
    l = 2;
  }

  id = (k - 1) + (l - 1) * 2;
  mp_ofst = id * nz * nz;
  d_ofst = id * nz;

  for (l = 0; l < nz; l++) {
    DDZ[l] = DDZ_b[d_ofst + l] / hz / hz;
  }
  for (l = 0; l < nz * nz; l++) {
    MPZ[l] = MPZ_b[mp_ofst + l];
    MPZI[l] = MPZI_b[mp_ofst + l];
  }
  for (n = 0; n < nac; n++) {
    ac_init_phi (ac[n].epn2, ac[n].LE, ac[n].KE, ac[n].phiE, ac[n].phiE2);
  }
  for (n = 0; n < nch; n++) {
    ch_init_phi (ch[n].epn2, ch[n].LCI, ch[n].KCI, ch[n].phiCI, ch[n].phiCI2);
  }
}

// dealloc all variables
void
dealloc_vars (void)
{
  int l;
  int n;
  _mm_free (fieldgx);
  _mm_free (fieldgy);
  _mm_free (fieldgz);
  for (n = 0; n < nac; n++) {
    _mm_free (ac[n].fieldE);
    _mm_free (ac[n].fieldE1);
    _mm_free (ac[n].fieldE2);
    _mm_free (ac[n].fieldEt);
    _mm_free (ac[n].fieldEp);
    _mm_free (ac[n].fieldE1p);
    _mm_free (ac[n].fieldEr);
    _mm_free (ac[n].fieldEs_left);
    _mm_free (ac[n].fieldEr_left);
    _mm_free (ac[n].fieldEs_right);
    _mm_free (ac[n].fieldEr_right);
    _mm_free (ac[n].fieldEs_top);
    _mm_free (ac[n].fieldEr_top);
    _mm_free (ac[n].fieldEs_bottom);
    _mm_free (ac[n].fieldEr_bottom);
    _mm_free (ac[n].fieldEr_front);
    _mm_free (ac[n].fieldEr_back);
    _mm_free (ac[n].fieldEe_left);
    _mm_free (ac[n].fieldEe_right);
    _mm_free (ac[n].fieldEe_top);
    _mm_free (ac[n].fieldEe_bottom);
    _mm_free (ac[n].fieldEe_front);
    _mm_free (ac[n].fieldEe_back);
    _mm_free (ac[n].fieldEu_left);
    _mm_free (ac[n].fieldEu_right);
    _mm_free (ac[n].fieldEu_top);
    _mm_free (ac[n].fieldEu_bottom);
    _mm_free (ac[n].fieldEu_front);
    _mm_free (ac[n].fieldEu_back);

    _mm_free (ac[n].fieldEmu_left);
    _mm_free (ac[n].fieldEmu_right);
    _mm_free (ac[n].fieldEmu_top);
    _mm_free (ac[n].fieldEmu_bottom);
    _mm_free (ac[n].fieldEmu_front);
    _mm_free (ac[n].fieldEmu_back);



    _mm_free (ac[n].felas);
    _mm_free (ac[n].felase_left);
    _mm_free (ac[n].felase_right);
    _mm_free (ac[n].felase_top);
    _mm_free (ac[n].felase_bottom);
    _mm_free (ac[n].felase_front);
    _mm_free (ac[n].felase_back);

    _mm_free (ac[n].phiE);
    _mm_free (ac[n].phiE2);
    for (l = 0; l < 4; l++) {
      MPI_Request_free (&ac[n].ireq_left_right_fieldE[l]);
      MPI_Request_free (&ac[n].ireq_top_bottom_fieldE[l]);
      MPI_Request_free (&ac[n].ireq_front_back_fieldE[l]);

      MPI_Request_free (&ac[n].ireq_left_right_felas[l]);
      MPI_Request_free (&ac[n].ireq_top_bottom_felas[l]);
      MPI_Request_free (&ac[n].ireq_front_back_felas[l]);
    }
    free (ac[n].ireq_left_right_fieldE);
    free (ac[n].ireq_top_bottom_fieldE);
    free (ac[n].ireq_front_back_fieldE);
    free (ac[n].ireq_left_right_felas);
    free (ac[n].ireq_top_bottom_felas);
    free (ac[n].ireq_front_back_felas);

    _mm_free(ac[n].gradx);
    _mm_free(ac[n].grady);
    _mm_free(ac[n].gradz);
    _mm_free(ac[n].f1);
    _mm_free(ac[n].f2);
    _mm_free(ac[n].f3);

  _mm_free(ac[n].Bn);
  _mm_free(ac[n].elas_field);
  _mm_free(ac[n].elas_re);
  _mm_free(ac[n].elas_im);
  _mm_free(ac[n].elas);
  _mm_free(ac[n].theta_re);
  _mm_free(ac[n].theta_im);
  }

  _mm_free(ac);

  for (n = 0; n < nch; n++) {
    _mm_free (ch[n].fieldCI);
    _mm_free (ch[n].fieldCI1);
    _mm_free (ch[n].fieldCI2);
    _mm_free (ch[n].fieldCIt);
    _mm_free (ch[n].fieldCIp);
    _mm_free (ch[n].fieldCI1p);
    _mm_free (ch[n].fieldCIs_left);
    _mm_free (ch[n].fieldCIr_left);
    _mm_free (ch[n].fieldCIs_right);
    _mm_free (ch[n].fieldCIr_right);
    _mm_free (ch[n].fieldCIr);
    _mm_free (ch[n].c_alpha_r);
    _mm_free (ch[n].c_delta_r);
    _mm_free (ch[n].fieldCIs_top);
    _mm_free (ch[n].fieldCIr_top);
    _mm_free (ch[n].fieldCIs_bottom);
    _mm_free (ch[n].fieldCIr_bottom);
    _mm_free (ch[n].fieldCIr_front);
    _mm_free (ch[n].fieldCIr_back);
    _mm_free (ch[n].fieldCIe_left);
    _mm_free (ch[n].fieldCIe_right);
    _mm_free (ch[n].fieldCIe_top);
    _mm_free (ch[n].fieldCIe_bottom);
    _mm_free (ch[n].fieldCIe_front);
    _mm_free (ch[n].fieldCIe_back);
    _mm_free (ch[n].fieldCIu_left);
    _mm_free (ch[n].fieldCIu_right);
    _mm_free (ch[n].fieldCIu_top);
    _mm_free (ch[n].fieldCIu_bottom);
    _mm_free (ch[n].fieldCIu_front);
    _mm_free (ch[n].fieldCIu_back);
    _mm_free (ch[n].fieldCImu_left);
    _mm_free (ch[n].fieldCImu_right);
    _mm_free (ch[n].fieldCImu_top);
    _mm_free (ch[n].fieldCImu_bottom);
    _mm_free (ch[n].fieldCImu_front);
    _mm_free (ch[n].fieldCImu_back);

    _mm_free (ch[n].felas);
    _mm_free (ch[n].felase_left);
    _mm_free (ch[n].felase_right);
    _mm_free (ch[n].felase_top);
    _mm_free (ch[n].felase_bottom);
    _mm_free (ch[n].felase_front);
    _mm_free (ch[n].felase_back);
    _mm_free (ch[n].phiCI);
    _mm_free (ch[n].phiCI2);
    for (l = 0; l < 4; l++) {
      MPI_Request_free (&ch[n].ireq_left_right_fieldCI[l]);
      MPI_Request_free (&ch[n].ireq_top_bottom_fieldCI[l]);
      MPI_Request_free (&ch[n].ireq_front_back_fieldCI[l]);
      MPI_Request_free (&ch[n].ireq_left_right_felas[l]);
      MPI_Request_free (&ch[n].ireq_top_bottom_felas[l]);
      MPI_Request_free (&ch[n].ireq_front_back_felas[l]);
    }
    free (ch[n].ireq_left_right_fieldCI);
    free (ch[n].ireq_top_bottom_fieldCI);
    free (ch[n].ireq_front_back_fieldCI);
    free (ch[n].ireq_left_right_felas);
    free (ch[n].ireq_top_bottom_felas);
    free (ch[n].ireq_front_back_felas);
    _mm_free (ch[n].c_alpha);
    _mm_free (ch[n].c_delta);
    _mm_free (ch[n].ft);
    _mm_free (ch[n].ftr);
  }
  _mm_free (ch);

  _mm_free (MPX);
  _mm_free (MPY);
  _mm_free (MPZ);
  _mm_free (MPXI);
  _mm_free (MPYI);
  _mm_free (MPZI);
  _mm_free (DDX);
  _mm_free (DDY);
  _mm_free (DDZ);
  _mm_free (MPX_b);
  _mm_free (MPY_b);
  _mm_free (MPZ_b);
  _mm_free (MPXI_b);
  _mm_free (MPYI_b);
  _mm_free (MPZI_b);
  _mm_free (DDX_b);
  _mm_free (DDY_b);
  _mm_free (DDZ_b);
  free (status);
  hipFree(fieldE);
  _mm_free(tmpy_fftre);
  _mm_free(fftre);
  rocblas_destroy_handle(handle);
  hipFree(tmpy_fftRE);
  hipFree(fftRE);
  hipFree(BN);
  hipFree(Elas);
  hipFree(tmpy_RE1);
  hipFree(tmpy_RE2);
  hipFree(Gx_re);
  hipFree(Gx_im);
  hipFree(Gy_re);
  hipFree(Gy_im);
  hipFree(Gz_re);
  hipFree(Gz_im);
  hipFree(temp1);
  hipFree(temp2);
  hipFree(temp3);
  hipFree(temp4);
  hipFree(temp5);
  hipFree(temp6);
  hipFree(temp7);
  hipFree(temp8);
  hipFree(Itemp1);
  hipFree(Itemp2);
  hipFree(Itemp3);
  hipFree(Itemp4);
  _mm_free(conv_r_right);
  _mm_free(conv_r_bottom);
  _mm_free(conv_r_back);
  _mm_free(conv_s_left);
  _mm_free(conv_s_top);
  _mm_free(conv_s_front);
  _mm_free(conv_s_right);
  _mm_free(conv_s_bottom);
  _mm_free(conv_s_back);
  _mm_free(conv_r_left);
  _mm_free(conv_r_top);
  _mm_free(conv_r_front);
  for (l = 0; l < 4; l++)
  {
    MPI_Request_free(&conv_ireq_left_right[l]);
    MPI_Request_free(&conv_ireq_top_bottom[l]);
    MPI_Request_free(&conv_ireq_front_back[l]);
  }

  free(conv_ireq_left_right);
  free(conv_ireq_top_bottom);
  free(conv_ireq_front_back);
  hipFree(conv_S_left);
  hipFree(conv_R_right);
  hipFree(conv_S_top);
  hipFree(conv_R_bottom);
  hipFree(conv_S_front);
  hipFree(conv_R_back);
  hipFree(conv_R_left);
  hipFree(conv_S_right);
  hipFree(conv_R_top);
  hipFree(conv_S_bottom);
  hipFree(conv_R_front);
  hipFree(conv_S_back);
  hipFree(conv_big_x_re);
  hipFree(conv_big_x_im);
  hipFree(conv_big_y_re);
  hipFree(conv_big_y_im);
  hipFree(conv_big_z_re);
  hipFree(conv_big_z_im);

  hipFree(conv_x_re);
  hipFree(conv_x_im);
  hipFree(conv_y_re);
  hipFree(conv_y_im);
  hipFree(conv_z_re);
  hipFree(conv_z_im);
  hipFree(conv_temp1);
  hipFree(conv_temp2);
  hipFree(conv_temp3);
  hipFree(conv_temp4);
  hipFree(conv_temp5);
  hipFree(conv_temp6);
  hipFree(conv_temp7);
  hipFree(conv_temp8);
  hipFree(conv_temp9);
  hipFree(conv_temp10);
  hipFree(conv_temp11);
  hipFree(conv_temp12);
  hipFree(BN_RE);
  hipFree(BN_IM);
  hipFree(eta_RE);
  hipFree(eta_IM);
  hipFree(bn_RE);
  hipFree(bn_IM);
  hipFree(conv_e);
}

