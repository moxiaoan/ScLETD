#include "ScLETD.h"
#include "ScLETD_hip.h"
#include <string.h>
#include <errno.h>
#include <hipfft.h>
#include <math.h>
#include <stdio.h>
//#include "hip_complex.h"
#include <stdlib.h>
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif
#include <time.h>

void
write_Bn(double *f, int p, int q)
{
  FILE *fp;
  int i, j, k;
  char fname[1024];
  sprintf(fname, "./Bn/Bn_omega_%d%d_%d%d%d", p, q, cart_id[0], cart_id[1], cart_id[2]);
  fp = fopen (fname, "w");
  for (k = 0; k < NZ; k++) {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
        fprintf (fp, "%+1.15lf\n", f[k * NX * NY + j * NX + i]);
      }
    }
  }
  fclose (fp);
}

void
write_Bn_im(double *f, double *d, int p)
{
  FILE *fp;
  int i, j, k;
  char fname[1024];
  sprintf(fname, "./data_relative_error/%02d/eta%d_000001_%02d%02d%02d.dat", cart_id[2], p, cart_id[0], cart_id[1], cart_id[2]);
  fp = fopen (fname, "w");
#if 1
  for (k = 0; k < NZ; k++) {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
        fprintf (fp, "%+1.15lf\n", f[k * NX * NY + j * NX + i]);
      }
    }
  }
#endif
#if 0
  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        fprintf (fp, "%+1.15lf\n", f[k * nx * ny + j * nx + i]);
      }
    }
  }
#endif
  fclose (fp);
/*
  sprintf(fname, "./Bn/Bnfft_im_%d_%d%d%d", p, cart_id[0], cart_id[1], cart_id[2]);
  fp = fopen (fname, "w");
  for (k = 0; k < NZ; k++) {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
        fprintf (fp, "%+1.15lf\n", d[k * NX * NY + j * NX + i]);
      }
    }
  }
  fclose (fp);
*/
}

static void
write_small(double *f)
{
  FILE *fp;
  int i, j, k;
  char fname[1024];
  // z section
  sprintf(fname, "./Bn/Bnz%d%d%d", cart_id[0], cart_id[1], cart_id[2]);
  //if (cart_id[2] == (procs[2]/2)) {
  if (cart_id[2] == 0)
  {
    fp = fopen(fname, "w");
    if (fp == NULL)
    {
      printf("fopen error %s!\n", strerror(errno));
      exit(1);
    }
    k = 0;
    for (j = 0; j < NY; j++)
    {
      for (i = 0; i < NX; i++)
      {
        fprintf(fp, "%+1.15lf ", f[k * NX * NY + j * NX + i]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
  // x section
  sprintf(fname, "./Bn/Bnx%d%d%d", cart_id[1], cart_id[2], cart_id[0]);
  if (cart_id[0] == (procs[0] / 2))
  {
    fp = fopen(fname, "w");
    if (fp == NULL)
    {
      printf("fopen error %s!\n", strerror(errno));
      exit(1);
    }
    i = 0;
    for (k = 0; k < NZ; k++)
    {
      for (j = 0; j < NY; j++)
      {
        fprintf(fp, "%+1.15lf ", f[k * NX * NY + j * NX + i]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
  sprintf(fname, "./Bn/Bny%d%d%d", cart_id[0], cart_id[2], cart_id[1]);
  if (cart_id[1] == (procs[1] / 2))
  {
    fp = fopen(fname, "w");
    if (fp == NULL)
    {
      printf("fopen error %s!\n", strerror(errno));
      exit(1);
    }
    j = 0;
    for (k = 0; k < NZ; k++)
    {
      for (i = 0; i < NX; i++)
      {
        fprintf(fp, "%+1.15lf ", f[k * NX * NY + j * NX + i]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}

static void
write_large(char *str, double *f)
{
  FILE *fp;
  int i, j, k;
  char fname[1024];
  // z section
  sprintf(fname, "./output/%s_z%d%d%d", str, cart_id[0], cart_id[1], cart_id[2]);
  //if (cart_id[2] == (procs[2]/2)) {
  if (cart_id[2] == 0)
  {
    fp = fopen(fname, "w");
    if (fp == NULL)
    {
      printf("fopen error %s!\n", strerror(errno));
      exit(1);
    }
    k = 6;
    for (j = 0; j < ny; j++)
    {
      for (i = 0; i < nx; i++)
      {
        fprintf(fp, "%+1.15lf ", f[k * nx * ny + j * nx + i]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
  // x section
  sprintf(fname, "./output/%s_x%d%d%d", str, cart_id[1], cart_id[2], cart_id[0]);
  if (cart_id[0] == (procs[0] / 2))
  {
    fp = fopen(fname, "w");
    if (fp == NULL)
    {
      printf("fopen error %s!\n", strerror(errno));
      exit(1);
    }
    i = 6;
    for (k = 0; k < nz; k++)
    {
      for (j = 0; j < ny; j++)
      {
        fprintf(fp, "%+1.15lf ", f[k * nx * ny + j * nx + i]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
  sprintf(fname, "./output/%s_y%d%d%d", str, cart_id[0], cart_id[2], cart_id[1]);
  if (cart_id[1] == (procs[1] / 2))
  {
    fp = fopen(fname, "w");
    if (fp == NULL)
    {
      printf("fopen error %s!\n", strerror(errno));
      exit(1);
    }
    j = 6;
    for (k = 0; k < nz; k++)
    {
      for (i = 0; i < nx; i++)
      {
        fprintf(fp, "%+1.15lf ", f[k * nx * ny + j * nx + i]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}

static void
check_max(double *v)
{
  int i, j, k;
  double tmp, maxtmp, mintmp, mmax, mmin;
  maxtmp = -1.0e30;
  mintmp = 1.0e30;
  for (k = 0; k < NZ; k++)
  {
    for (j = 0; j < NY; j++)
    {
      for (i = 0; i < NX; i++)
      {
        tmp = v[k * NX * NY + j * NX + i];
        if (tmp > maxtmp)
        {
          maxtmp = tmp;
        }
        if (tmp < mintmp)
        {
          mintmp = tmp;
        }
      }
    }
  }
  MPI_Reduce(&maxtmp, &mmax, 1, MPI_DOUBLE, MPI_MAX, prank, MPI_COMM_WORLD);
  MPI_Reduce(&mintmp, &mmin, 1, MPI_DOUBLE, MPI_MIN, prank, MPI_COMM_WORLD);

  if (myrank == prank)
  {
    printf("max:\t\t%+1.15lf,\t\t%+1.15lf\n", mmax, mmin);
  }
}

// local variables
void
elastic_malloc()
{
  int p, q;
  for (p = 0; p < NELASTIC; p++)
  {
  	for (q = 0; q < NELASTIC; q++)
        {
  		tmpy_fftre[p][q] = (double *)_mm_malloc(sizeof(double) * Approx * Approx * Approx * 8, 256);
  		fftre[p][q] = (double *)_mm_malloc(sizeof(double) * Approx * Approx * Approx * 8, 256);
        }
  }
  tmpy_re = (double *)_mm_malloc(sizeof(double) * NX * NY * NZ, 256);
  s_left = (double *)_mm_malloc(sizeof(double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
  s_right = (double *)_mm_malloc(sizeof(double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
  s_top = (double *)_mm_malloc(sizeof(double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
  s_bottom = (double *)_mm_malloc(sizeof(double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
  s_front = (double *)_mm_malloc(sizeof(double) * nx * ny * (nghost + 2), 256);
  s_back = (double *)_mm_malloc(sizeof(double) * nx * ny * (nghost + 2), 256);
  r_left = (double *)_mm_malloc(sizeof(double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
  r_right = (double *)_mm_malloc(sizeof(double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
  r_top = (double *)_mm_malloc(sizeof(double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
  r_bottom = (double *)_mm_malloc(sizeof(double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
  r_front = (double *)_mm_malloc(sizeof(double) * nx * ny * (nghost + 2), 256);
  r_back = (double *)_mm_malloc(sizeof(double) * nx * ny * (nghost + 2), 256);
  e_left = (double *)_mm_malloc(sizeof(double) * (nghost + 2) * (ny + 4) * (nz + 4), 256);
  e_right = (double *)_mm_malloc(sizeof(double) * (nghost + 2) * (ny + 4) * (nz + 4), 256);
  e_top = (double *)_mm_malloc(sizeof(double) * (nx + 4) * (nghost + 2) * (nz + 4), 256);
  e_bottom = (double *)_mm_malloc(sizeof(double) * (nx + 4) * (nghost + 2) * (nz + 4), 256);
  e_front = (double *)_mm_malloc(sizeof(double) * (nx + 4) * (ny + 4) * (nghost + 2), 256);
  e_back = (double *)_mm_malloc(sizeof(double) * (nx + 4) * (ny + 4) * (nghost + 2), 256);
  ireq_left_right = (MPI_Request *)calloc(4, sizeof(MPI_Request));
  ireq_top_bottom = (MPI_Request *)calloc(4, sizeof(MPI_Request));
  ireq_front_back = (MPI_Request *)calloc(4, sizeof(MPI_Request));
  //convolution variables_cpu
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
  for (p = 0; p < NELASTIC; p++)
  {
 	for (q = 0; q < NELASTIC; q++)
	{
		hipMalloc (&fftRE[p][q], Approx * Approx * Approx * 8 * sizeof(double));
		hipMalloc (&tmpy_fftRE[p][q], Approx * Approx * Approx * 8 * sizeof(double));
	}
	hipMalloc (&Epsilon2d[p], DIM * DIM * sizeof(double));
	hipMalloc (&Sigma2d[p], DIM * DIM * sizeof(double));
  }
  hipMalloc (&C4D, DIM * DIM * DIM * DIM * sizeof(double));
  hipMalloc (&n11, NX * sizeof(double));
  hipMalloc (&n22, NY * sizeof(double));
  hipMalloc (&n33, NZ * sizeof(double));
  hipMalloc (&BN, NX * NY * NZ * sizeof(double));
  hipMalloc (&tmpy_RE1, NX * NY * NZ*sizeof(double));
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
  hipMalloc (&Elas, nac * nx * ny * nz* sizeof (double));
  hipMalloc (&Elas1, NX * NY * NZ* sizeof (double));
  hipMemset(Elas, 0.0, sizeof(double) * nac * nx * ny * nz);
  hipMalloc (&S_left, (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2) * sizeof (double));
  hipMalloc (&R_left, (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2) * sizeof (double));
  hipMalloc (&S_right, (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2) * sizeof (double));
  hipMalloc (&R_right, (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2) * sizeof (double));
  hipMalloc (&S_top, nx * (nghost + 2) * (nz + (nghost + 2) * 2) * sizeof (double));
  hipMalloc (&R_top, nx * (nghost + 2) * (nz + (nghost + 2) * 2) * sizeof (double));
  hipMalloc (&S_bottom, nx * (nghost + 2) * (nz + (nghost + 2) * 2) * sizeof (double));
  hipMalloc (&R_bottom, nx * (nghost + 2) * (nz + (nghost + 2) * 2) * sizeof (double));
  hipMalloc (&S_front, nx * ny * (nghost + 2) * sizeof (double));
  hipMalloc (&R_front, nx * ny * (nghost + 2) * sizeof (double));
  hipMalloc (&S_back, nx * ny * (nghost + 2) * sizeof (double));
  hipMalloc (&R_back, nx * ny * (nghost + 2) * sizeof (double));
  //convolution variables_dcu
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
}

void
elastic_init_transfer()
{
  MPI_Send_init(&s_left[0], 1, left_right, left, 9, MPI_COMM_WORLD, &ireq_left_right[0]);
  MPI_Recv_init(&r_right[0], 1, left_right, right, 9, MPI_COMM_WORLD, &ireq_left_right[1]);
  MPI_Send_init(&s_right[0], 1, left_right, right, 9, MPI_COMM_WORLD, &ireq_left_right[2]);
  MPI_Recv_init(&r_left[0], 1, left_right, left, 9, MPI_COMM_WORLD, &ireq_left_right[3]);
  MPI_Send_init(&s_top[0], 1, top_bottom, top, 9, MPI_COMM_WORLD, &ireq_top_bottom[0]);
  MPI_Recv_init(&r_bottom[0], 1, top_bottom, bottom, 9, MPI_COMM_WORLD, &ireq_top_bottom[1]);
  MPI_Send_init(&s_bottom[0], 1, top_bottom, bottom, 9, MPI_COMM_WORLD, &ireq_top_bottom[2]);
  MPI_Recv_init(&r_top[0], 1, top_bottom, top, 9, MPI_COMM_WORLD, &ireq_top_bottom[3]);
  MPI_Send_init(&s_front[0], 1, front_back, front, 9, MPI_COMM_WORLD, &ireq_front_back[0]);
  MPI_Recv_init(&r_back[0], 1, front_back, back, 9, MPI_COMM_WORLD, &ireq_front_back[1]);
  MPI_Send_init(&s_back[0], 1, front_back, back, 9, MPI_COMM_WORLD, &ireq_front_back[2]);
  MPI_Recv_init(&r_front[0], 1, front_back, front, 9, MPI_COMM_WORLD, &ireq_front_back[3]);
}

void
conv_init_transfer()
{
  MPI_Send_init(&conv_s_left[0], 1, conv_left_right, left, 9, MPI_COMM_WORLD, &conv_ireq_left_right[0]);
  MPI_Recv_init(&conv_r_right[0], 1, conv_left_right, right, 9, MPI_COMM_WORLD, &conv_ireq_left_right[1]);
  MPI_Send_init(&conv_s_right[0], 1, conv_left_right, right, 9, MPI_COMM_WORLD, &conv_ireq_left_right[2]);
  MPI_Recv_init(&conv_r_left[0], 1, conv_left_right, left, 9, MPI_COMM_WORLD, &conv_ireq_left_right[3]);
  MPI_Send_init(&conv_s_top[0], 1, conv_top_bottom, top, 9, MPI_COMM_WORLD, &conv_ireq_top_bottom[0]);
  MPI_Recv_init(&conv_r_bottom[0], 1, conv_top_bottom, bottom, 9, MPI_COMM_WORLD, &conv_ireq_top_bottom[1]);
  MPI_Send_init(&conv_s_bottom[0], 1, conv_top_bottom, bottom, 9, MPI_COMM_WORLD, &conv_ireq_top_bottom[2]);
  MPI_Recv_init(&conv_r_top[0], 1, conv_top_bottom, top, 9, MPI_COMM_WORLD, &conv_ireq_top_bottom[3]);
  MPI_Send_init(&conv_s_front[0], 1, conv_front_back, front, 9, MPI_COMM_WORLD, &conv_ireq_front_back[0]);
  MPI_Recv_init(&conv_r_back[0], 1, conv_front_back, back, 9, MPI_COMM_WORLD, &conv_ireq_front_back[1]);
  MPI_Send_init(&conv_s_back[0], 1, conv_front_back, back, 9, MPI_COMM_WORLD, &conv_ireq_front_back[2]);
  MPI_Recv_init(&conv_r_front[0], 1, conv_front_back, front, 9, MPI_COMM_WORLD, &conv_ireq_front_back[3]);
}

void elastic_finish()
{
  int p, q, l;
//  fft_finish();
  for (p = 0; p < NELASTIC; p++)
  {
    for (q = 0; q < NELASTIC; q++)
    {
      _mm_free(tmpy_fftre[p][q]);
      _mm_free(fftre[p][q]);

      hipFree(tmpy_fftRE[p][q]);
      hipFree(fftRE[p][q]);
    }
    hipFree(Epsilon2d[p]);
    hipFree(Sigma2d[p]);
  }
  hipFree(C4D);
  hipFree(n11);
  hipFree(n22);
  hipFree(n33);
  _mm_free(tmpy_re);
  hipFree(tmpy_RE1);
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
  /* free elastic variable*/
  _mm_free(s_left);
  _mm_free(s_right);
  _mm_free(s_top);
  _mm_free(s_bottom);
  _mm_free(s_front);
  _mm_free(s_back);
  _mm_free(r_left);
  _mm_free(r_right);
  _mm_free(r_top);
  _mm_free(r_bottom);
  _mm_free(r_front);
  _mm_free(r_back);
  _mm_free(e_left);
  _mm_free(e_right);
  _mm_free(e_top);
  _mm_free(e_bottom);
  _mm_free(e_front);
  _mm_free(e_back);
  for (l = 0; l < 4; l++)
  {
    MPI_Request_free(&ireq_left_right[l]);
    MPI_Request_free(&ireq_top_bottom[l]);
    MPI_Request_free(&ireq_front_back[l]);
  }
  free(ireq_left_right);
  free(ireq_top_bottom);
  free(ireq_front_back);

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
  hipFree(Elas);
  hipFree(Elas1);
  hipFree(S_left);
  hipFree(R_left);
  hipFree(S_right);
  hipFree(R_right);
  hipFree(S_top);
  hipFree(R_top);
  hipFree(S_bottom);
  hipFree(R_bottom);
  hipFree(S_front);
  hipFree(R_front);
  hipFree(S_back);
  hipFree(R_back);

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

void
elastic_index_3to6(const int i, const int j, int *m)
{
  if (i == 0 && j == 0)
  {
    *m = 0;
  }
  else if (i == 1 && j == 1)
  {
    *m = 1;
  }
  else if (i == 2 && j == 2)
  {
    *m = 2;
  }
  else if ((i == 1 && j == 2) || (i == 2 && j == 1))
  {
    *m = 3;
  }
  else if ((i == 0 && j == 2) || (i == 2 && j == 0))
  {
    *m = 4;
  }
  else if ((i == 0 && j == 1) || (i == 1 && j == 0))
  {
    *m = 5;
  }
}

void elastic_input()
{

  int i, j, k, l;
  int m, n;
  char filename[1024];
  FILE *fp;
  char *aline = NULL;
  size_t len = 0;
  double C2d[6 * 6];
  NX = nx - 2 * nghost;
  NY = ny - 2 * nghost;
  NZ = nz - 2 * nghost;
  sprintf(filename, "elastic_input.txt");
  fp = fopen(filename, "r");
  if (fp == NULL)
  {
    printf("open elastic_input.txt Fail!");
    exit(EXIT_FAILURE);
  }
  if (myrank == 0)
  {
    printf("Reading input.txt file:\n");
  }
  //getline(&aline, &len, fp);
  //printf("%s\n", aline);

  fscanf(fp, "%lf\n", &ElasticScale);
  /*
  C00 C01 C02 C03 C04 C05 
      C11 C12 C12 C14 C15
          C22 C23 C24 C25
              C33 C34 C35
                  C44 C45
                      C55
  */
  for (i = 0; i < 2 * DIM; i++)
  {
    for (j = 0; j < 2 * DIM; j++)
    {
      fscanf(fp, "%lf\n", &C2d[i * 2 * DIM + j]);
    }
  }

  // 00: 0, 11: 1, 22: 2, 12/21: 3, 02/20: 4, 01/10: 5
  for (i = 0; i < DIM; i++)
  {
    for (j = 0; j < DIM; j++)
    {
      for (k = 0; k < DIM; k++)
      {
        for (l = 0; l < DIM; l++)
        {
          elastic_index_3to6(i, j, &m);
          elastic_index_3to6(k, l, &n);
          C4d[i * DIM * DIM * DIM + j * DIM * DIM + k * DIM + l] = C2d[m * 2 * DIM + n];
        }
      }
    }
  }
  int p;
  for (p = 0; p < NELASTIC; p++)
  {
    // 00: 0, 11: 1, 22: 2, 12/21: 3, 02/20: 4, 01/10: 5
    for (i = 0; i < DIM; i++)
    {
      for (j = 0; j < DIM; j++)
      {
        fscanf(fp, "%lf", &(epsilon2d[p][i * DIM + j]));
      }
    }
    // sigma0[ij] = C[ijkl] * epsilon0[kl]
    for (i = 0; i < DIM; i++)
    {
      for (j = 0; j < DIM; j++)
      {
        sigma2d[p][i * DIM + j] = 0;
//       for (k = 0; k < DIM; k++)
//       {
//         for (l = 0; l < DIM; l++)
//         {
//           sigma2d[p][i * DIM + j] += C4d[i * DIM * DIM * DIM + j * DIM * DIM + k * DIM + l] * epsilon2d[p][k * DIM + l];
//         }
//       }
	//printf("sigma[%d][%d][%d] = %lf\n", p, i, j, sigma2d[p][i * DIM + j]);
      }
    }
  }
  fclose(fp);
  if (aline)
    free(aline);
}

// omega = 1/normal(omega_inver) * (-1)^(i+j) * M(omega_inver)
void
inverse_3x3(double *omega, double *omega_inver)
{
  double normal = 0.0;
  normal += omega_inver[0 * DIM + 0] * (omega_inver[1 * DIM + 1] * omega_inver[2 * DIM + 2] - omega_inver[1 * DIM + 2] * omega_inver[2 * DIM + 1]);
  normal -= omega_inver[0 * DIM + 1] * (omega_inver[1 * DIM + 0] * omega_inver[2 * DIM + 2] - omega_inver[1 * DIM + 2] * omega_inver[2 * DIM + 0]);
  normal += omega_inver[0 * DIM + 2] * (omega_inver[1 * DIM + 0] * omega_inver[2 * DIM + 1] - omega_inver[1 * DIM + 1] * omega_inver[2 * DIM + 0]);
  omega[0 * DIM + 0] = (+1) * (omega_inver[1 * DIM + 1] * omega_inver[2 * DIM + 2] - omega_inver[1 * DIM + 2] * omega_inver[2 * DIM + 1]) / normal;
  omega[0 * DIM + 1] = (-1) * (omega_inver[1 * DIM + 0] * omega_inver[2 * DIM + 2] - omega_inver[2 * DIM + 0] * omega_inver[1 * DIM + 2]) / normal;
  omega[0 * DIM + 2] = (+1) * (omega_inver[1 * DIM + 0] * omega_inver[2 * DIM + 1] - omega_inver[2 * DIM + 0] * omega_inver[1 * DIM + 1]) / normal;
  omega[1 * DIM + 0] = (-1) * (omega_inver[0 * DIM + 1] * omega_inver[2 * DIM + 2] - omega_inver[2 * DIM + 1] * omega_inver[0 * DIM + 2]) / normal;
  omega[1 * DIM + 1] = (+1) * (omega_inver[0 * DIM + 0] * omega_inver[2 * DIM + 2] - omega_inver[0 * DIM + 2] * omega_inver[2 * DIM + 0]) / normal;
  omega[1 * DIM + 2] = (-1) * (omega_inver[0 * DIM + 0] * omega_inver[2 * DIM + 1] - omega_inver[2 * DIM + 0] * omega_inver[0 * DIM + 1]) / normal;
  omega[2 * DIM + 0] = (+1) * (omega_inver[0 * DIM + 1] * omega_inver[1 * DIM + 2] - omega_inver[1 * DIM + 1] * omega_inver[0 * DIM + 2]) / normal;
  omega[2 * DIM + 1] = (-1) * (omega_inver[0 * DIM + 0] * omega_inver[1 * DIM + 2] - omega_inver[1 * DIM + 0] * omega_inver[0 * DIM + 2]) / normal;
  omega[2 * DIM + 2] = (+1) * (omega_inver[0 * DIM + 0] * omega_inver[1 * DIM + 1] - omega_inver[0 * DIM + 1] * omega_inver[1 * DIM + 0]) / normal;
}
#if 0
void
elastic_calculate_BN()
{
  int p, q;
  int x, y, z;
  int i, j, k, l;
  double n11[NX], n22[NY], n33[NZ];
  double normal;

  // calculate n
  int gnnx = procs[0] * NX; // 256
  int gnny = procs[1] * NY; // 256
  int gnnz = procs[2] * NZ; // 256
  int cntx = gnnx / 2;      // 128
  int cnty = gnny / 2;      // 128
  int cntz = gnnz / 2;      // 128
  for (x = 0; x < NX; x++)  // 0-64
  {
    int gx = cart_id[0] * NX + x; // 0-255
    if (gx < cntx)                // 0-127
    {
      n11[x] = 1.0 * gx; // 0-127
    }
    else
    {                              // 128-255
      n11[x] = -1.0 * (gnnx - gx); // (-128)-(-1)
    }
    //printf("n11[%d] = %+1.15lf\n", x, n11[x]);
  }

  for (y = 0; y < NY; y++) // 0-64
  {
    int gy = cart_id[1] * NY + y; // 0-255
    if (gy < cnty)                // 0-127
    {
      n22[y] = 1.0 * gy; // 0-127
    }
    else
    {                              // 128-255
      n22[y] = -1.0 * (gnny - gy); // (-128)-(-1)
    }
  }
  for (z = 0; z < NZ; z++) // 0-64
  {
    int gz = cart_id[2] * NZ + z; // 0-255
    if (gz < cntz)                // 0-127
    {
      n33[z] = 1.0 * gz; // 0-127
    }
    else
    {                              // 128-255
      n33[z] = -1.0 * (gnnz - gz); // (-128)-(-1)
    }
  }
  /* 
    Bn(p,q) = C[ijkl] * epsilon0(p)[ij] * epsilon0(q)[kl] 
     - n[i] * sigma0(p)[ij] * omega[jk] * sigma0(q)[kl] * n[l]
  */
  for (p = 0; p < NELASTIC; p++)
  {
    for (q = 0; q < NELASTIC; q++)
    {
      for (z = 0; z < NZ; z++)
      {
        for (y = 0; y < NY; y++)
        {
          for (x = 0; x < NX; x++)
          {
            double n123[DIM];
            double omega_inver[DIM * DIM];
            double omega[DIM * DIM];
            double tmp0, tmp1;

            int gx = cart_id[0] * NX + x;
            int gy = cart_id[1] * NY + y;
            int gz = cart_id[2] * NZ + z;
            if (gx == 0 && gy == 0 && gz == 0)
            {
              n123[0] = 0;
              n123[1] = 1;
              n123[2] = 0;
            }
            else
            {
              n123[0] = n11[x] / pow((pow(n11[x], 2) + pow(n22[y], 2) + pow(n33[z], 2)), 0.5);
              n123[1] = n22[y] / pow((pow(n11[x], 2) + pow(n22[y], 2) + pow(n33[z], 2)), 0.5);
              n123[2] = n33[z] / pow((pow(n11[x], 2) + pow(n22[y], 2) + pow(n33[z], 2)), 0.5);
            }

            // omega_inver[ik] = C[ijkl] * n[j] * n[l]
            for (i = 0; i < DIM; ++i)
            {
              for (k = 0; k < DIM; ++k)
              {
                omega_inver[i * DIM + k] = 0.0;
                for (j = 0; j < DIM; ++j)
                {
                  for (l = 0; l < DIM; ++l)
                  {
                    omega_inver[i * DIM + k] += C4d[i * DIM * DIM * DIM + j * DIM * DIM + k * DIM + l] * n123[j] * n123[l];
                  }
                }
              }
            }

            // omega = inverse(omega_inver)
            inverse_3x3(omega, omega_inver);
            tmp0 = 0;
            // tmp0 = C[ijkl] * epsilon[ij] * epsilon[kl]
            for (i = 0; i < DIM; i++)
            {
              for (j = 0; j < DIM; j++)
              {
                for (k = 0; k < DIM; k++)
                {
                  for (l = 0; l < DIM; l++)
                  {
                    tmp0 += C4d[i * DIM * DIM * DIM + j * DIM * DIM + k * DIM + l] * epsilon2d[p][i * DIM + j] * epsilon2d[q][k * DIM + l];
                  }
                }
              }
            }

            // tmp1 = n[i] * sigma0[ij] * omega[jk] * sigma0[kl] *n[l]
            tmp1 = 0;
            for (i = 0; i < DIM; i++)
            {
              for (j = 0; j < DIM; j++)
              {
                for (k = 0; k < DIM; k++)
                {
                  for (l = 0; l < DIM; l++)
                  {
                    tmp1 += n123[i] * sigma2d[p][i * DIM + j] * omega[j * DIM + k] * sigma2d[q][k * DIM + l] * n123[l];
                  }
                }
              }
            }
            // Bn = tmp0 - tmp1
            Bn[p][q][z * NY * NX + y * NX + x] = tmp0 - tmp1;
	    //Bn[z * NY * NX + y * NX + x] = 75 - tmp1;
          }
        }
      }
      hipMemcpy (BN[p][q], Bn[p][q], NZ* NY* NX* sizeof (double), hipMemcpyHostToDevice);
    }
  }
}
#endif
void elastic_init()
{
  //fft_setup();
  elastic_malloc();
  elastic_init_transfer();
  conv_init_transfer();
  //elastic_calculate_BN();
}

