#include "ScLETD.h"
#include "anisotropic_hip.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
//#include "fft3d.h"
#include <string.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif
// convert 3x3x3x3 Cijkl to 6x6
static void elastic_index_3to6(const int i, const int j, int *m)
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

// read Cijkl and epsilon tensor from file
void elastic_read_input()
{

  int i, j, k, l;
  int m, n;
  FILE *fp;
  char *aline = NULL;
  size_t len = 0;
  ssize_t read;

  fp = fopen("elastic_input.txt", "r");
  if (fp == NULL)
  {
    printf("open input.txt Fail!");
    exit(EXIT_FAILURE);
  }
  if (myrank == 0)
  {
    printf("Reading input.txt file:\n");
  }

  /*
  C00 C01 C02 C03 C04 C05 
      C11 C12 C12 C14 C15
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
  // 00: 0, 11: 1, 22: 2, 12/21: 3, 02/20: 4, 01/10: 5
for (n = 0; n < nac; n++) 
{
  for (i = 0; i < DIM; i++)
  {
    for (j = 0; j < DIM; j++)
    {
      fscanf(fp, "%lf", &(ac[n].epsilon2d[i * DIM + j]));
    }
  }

       // sigma0[ij] = C[ijkl] * epsilon0[kl]
        for (i = 0; i < DIM; i++)
        {
          for (j = 0; j < DIM; j++)
          {
            ac[n].sigma2d[i * DIM + j] = 0;
            for (k = 0; k < DIM; k++)
            {
              for (l = 0; l < DIM; l++)
              {
                ac[n].sigma2d[i * DIM + j] += C4d[i * DIM * DIM * DIM + j * DIM * DIM + k * DIM + l] * ac[n].epsilon2d[k * DIM + l];
              }
            }
          }
        }

}
  fclose(fp);
  if (aline)
  free(aline);

  int x, y, z;
  // initialize all elastic before calculate
for (n = 0; n < nac; n++) 
{
  // elastic
  for (x = 0; x < nx; x++)
  {
    for (y = 0; y < ny; y++)
    {
      for (z = 0; z < nz; z++)
      {
        ac[n].felas[z * ny * nx + y * nx + x] = 0.0;
      }
    }
  }
}
}
// omega = 1/normal(omega_inver) * (-1)^(i+j) * M(omega_inver)
void inverse_3x3(double *omega, double *omega_inver) 
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

// calculate BN 
// Bn = C[ijkl] * epsilon0[ij] * epsilon0[kl] 
// - n[i] * sigma0[ij] * omega[jk] * sigma0[kl] * n[l]
void elastic_calculate_BN()
{
  int x, y, z;
  int i, j, k, l;
  double n11[NX], n22[NY], n33[NZ];
  double normal;

  // calculate n
  int gnnx = procs[0] * NX; // 256
  int gnny = procs[1] * NY; // 256
  int gnnz = procs[2] * NZ; // 256
  int cntx = gnnx / 2; // 128
  int cnty = gnny / 2; // 128
  int cntz = gnnz / 2; // 128
  for (x = 0; x < NX; x++) // 0-64
  {
    int gx = cart_id[0] * NX + x; // 0-255
    if (gx < cntx) // 0-127
    {
      n11[x] = 1.0 * gx; // 0-127
    } else { // 128-255
      n11[x] = -1.0 * (gnnx - gx); // (-128)-(-1)
    }
  }

  for (y = 0; y < NY; y++) // 0-64
  {
    int gy = cart_id[1] * NY + y; // 0-255
    if (gy < cnty) // 0-127
    {
      n22[y] = 1.0 * gy; // 0-127
    } else
    { // 128-255
      n22[y] = -1.0 * (gnny - gy); // (-128)-(-1)
    }
  }
  for (z = 0; z < NZ; z++) // 0-64
  {
    int gz = cart_id[2] * NZ + z; // 0-255
    if (gz < cntz) // 0-127
    {
      n33[z] = 1.0 * gz; // 0-127
    } else 
    { // 128-255
      n33[z] = -1.0 * (gnnz - gz); // (-128)-(-1)
    }
  }

  // Bn = C[ijkl] * epsilon0[ij] * epsilon0[kl] 
  // - n[i] * sigma0[ij] * omega[jk] * sigma0[kl] * n[l]
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
        // calculate nijk
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
        // omega_inver[ij] = C[ijkl] * n[k] * n[l]
        for (i = 0; i < DIM; ++i)
        {
          for (j = 0; j < DIM; ++j)
          {
            omega_inver[i * DIM + j] = 0.0;
            for (k = 0; k < DIM; ++k)
            {
              for (l = 0; l < DIM; ++l)
              {
                omega_inver[i * DIM + j] += C4d[i * DIM * DIM * DIM + j * DIM * DIM + k * DIM + l] * n123[k] * n123[l];
              }
            }
          }
        }

        // omega = inverse(omega_inver)
        inverse_3x3(omega, omega_inver);

int n, m;
for (n = 0; n < nac; n++) 
{
  for (m = 0; m < nac; m++)
  {
        tmp0 = 0;
        // tmp0 = C[ijkl] * epsilon(n)[ij] * epsilon(m)[kl]
        for (i = 0; i < DIM; i++)
        {
          for (j = 0; j < DIM; j++)
          {
            for (k = 0; k < DIM; k++)
            {
              for (l = 0; l < DIM; l++)
              {
                tmp0 += C4d[i * DIM * DIM * DIM + j * DIM * DIM + k * DIM + l] * ac[n].epsilon2d[i * DIM + j] * ac[m].epsilon2d[k * DIM + l];
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
                tmp1 += n123[i] * ac[n].sigma2d[i * DIM + j] * omega[j * DIM + k] * ac[m].sigma2d[k * DIM + l] * n123[l];
              }
            }
          }
        }
        // Bn = tmp0 - tmp1
        ac[n].Bn[m * NZ * NY * NX + z * NY * NX + y * NX + x] = tmp0 - tmp1;
  }
}
      }
    }
  }
  hipMemcpy (BN, ac[0].Bn, sizeof (Dtype) * NX * NY * NZ, hipMemcpyHostToDevice);
}
#if 0
// Bn_star = dtheta(c) * IFFT(Bn * FFT(theta(c)))
void elastic_calculate_ElasDri()
{
  int x, y, z;
  int n, m;
  // prepare data for fft
for (n = 0; n < nac; n++) 
{
for (m = 0; m < nac; m++) 
{ 
  for (x = 0; x < NX; x++)
  {
    for (y = 0; y < NY; y++)
    {
      for (z = 0; z < NZ; z++)
      {
        ac[n].elas_re[z * NY * NX + y * NX + x] = ac[m].theta_re[z * NY * NX + y * NX + x]
                                              * ac[n].Bn[m * NZ * NY * NX + z * NY * NX + y * NX + x];
        ac[n].elas_im[z * NY * NX + y * NX + x] = ac[m].theta_im[z * NY * NX + y * NX + x]
                                              * ac[n].Bn[m * NZ * NY * NX + z * NY * NX + y * NX + x]; 
      }
    }
  }

  // Bn_starift = IFFT(Bn_star)
  fft_backward(ac[n].elas_re, ac[n].elas_im, ac[n].elas_re);
  // elas = 2 * c * Bn_stariftre
  for (x = 0; x < NX; x++)
  {
    for (y = 0; y < NY; y++)
    {
      for (z = 0; z < NZ; z++)
      {
        ac[n].elas[z * NY * NX + y * NX + x] 
          += ac[n].elas_re[z * NY * NX + y * NX + x]
          * 2 * ac[n].elas_field[z * NY * NX + y * NX + x];
      }
    }
  }
}
}
}
// copy data to buffer for elastic calculate
// NX = nx - 2*nghost
void elastic_copyin()
{
  int x, y, z;
  int n;
for (n = 0; n < nac; n++)
{
  for (x = 0; x < NX; x++)
  {
    for (y = 0; y < NY; y++)
    {
      for (z = 0; z < NZ; z++)
      {
        ac[n].elas_field[z * NY * NX + y * NX + x] 
         = ac[n].fieldE[(z + nghost) * ny * nx + (y + nghost) * nx + x + nghost];
        ac[n].elas[z * NY * NX + y * NX + x] = 0; 
      }
    }
  }

  for (x = 0; x < NX; x++)
  {
    for (y = 0; y < NY; y++)
    {
      for (z = 0; z < NZ; z++)
      {
        // theta(c) = eta * eta
        ac[n].theta_re[z * NY * NX + y * NX + x] = 
          ac[n].elas_field[z * NY * NX + y * NX + x]
          * ac[n].elas_field[z * NY * NX + y * NX + x];
      }
    }
  }
  // tmpy_fft = FFT(theta(c))
  fft_forward(ac[n].theta_re, ac[n].theta_re, ac[n].theta_im);
}

}

// copy calculated elastic out
void elastic_copyout()
{
  int x, y, z;
  int n = 0;

  for (x = 0; x < NX; x++)
  {
    for (y = 0; y < NY; y++)
    {
      for (z = 0; z < NZ; z++)
      {
        ac[n].felas[(z + nghost) * ny * nx + (y + nghost) * nx + x + nghost] =
           ac[n].elas[z * NY * NX + y * NX + x];
      }
    }
  }
}
#endif
// check min and max of elastic
void check_max(double *v)
{
  int i, j, k;
  double tmp, maxtmp, mintmp, mmax, mmin;
  int count = 0;
    maxtmp = -1.0e30;
    mintmp = 1.0e30;
    for (k = 0; k < nz; k++) {
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
          tmp = v[k * nx * ny + j * nx + i];
          if (tmp > maxtmp) {
            maxtmp = tmp;
          }
          if (tmp < mintmp) {
            mintmp = tmp;
          }
        }
      }
    }
    MPI_Reduce (&maxtmp, &mmax, 1, MPI_DOUBLE, MPI_MAX, prank, MPI_COMM_WORLD);
    MPI_Reduce (&mintmp, &mmin, 1, MPI_DOUBLE, MPI_MIN, prank, MPI_COMM_WORLD);
	
    if (myrank == prank) {
      printf ("max:\t\t%+1.15lf,\t\t%+1.15lf\n", mmax, mmin);
    }
}

// transfer elastic 
void elastic_transfer() 
{
  int n = 0;
  MPI_Startall (4, ac[n].ireq_front_back_felas);
  MPI_Waitall (4, ac[n].ireq_front_back_felas, status);

  top_bottom_pack (ac[n].felas, ac[n].fieldEs_top, ac[n].fieldEs_bottom, ac[n].fieldEr_front, ac[n].fieldEr_back);
  MPI_Startall (4, ac[n].ireq_top_bottom_felas);
  MPI_Waitall (4, ac[n].ireq_top_bottom_felas, status);

  left_right_pack (ac[n].felas, ac[n].fieldEs_left, ac[n].fieldEs_right, ac[n].fieldEr_top, ac[n].fieldEr_bottom, ac[n].fieldEr_front, ac[n].fieldEr_back);
  MPI_Startall (4, ac[n].ireq_left_right_felas);
  MPI_Waitall (4, ac[n].ireq_left_right_felas, status);
  unpack (ac[n].felas, ac[n].fieldEr_left, ac[n].fieldEr_right, ac[n].fieldEr_top, ac[n].fieldEr_bottom, ac[n].fieldEr_front, ac[n].fieldEr_back);
  enlarge (ac[n].felase_left, ac[n].felase_right, ac[n].felase_top, ac[n].felase_bottom, ac[n].felase_front, ac[n].felase_back,
  ac[n].fieldEr_left, ac[n].fieldEr_right, ac[n].fieldEr_top, ac[n].fieldEr_bottom, ac[n].fieldEr_front, ac[n].fieldEr_back);
//  check_max(ac[n].felas);
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

void
write_Bn(double *f)
{
  FILE *fp;
  int i, j, k;
  char fname[1024];
  sprintf(fname, "./Bn/Bn_KKS%d%d%d", cart_id[0], cart_id[1], cart_id[2]);
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
