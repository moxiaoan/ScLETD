#include "ScLETD.h"
#include "ScLETD_hip.h"
#include <assert.h>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "rocfft.h"
#include "hipfft.h"
#include <complex.h>
#include <rocfft-export.h>
#include <rocfft-version.h>
#define SCLETD_DEBUG

__global__ void
anisotropic_calc_dev_rc(double *f, double *fa, double *lambda, int *orie, int n)
{
  int i, j, k;
  //  int n_left, n_right, n_top, n_bottom, n_front, n_back;
  double a_left, a_right, a_top, a_bottom, a_front, a_back, a_middle;
  double a_left_top, a_left_bottom, a_left_front, a_left_back;
  double a_right_top, a_right_bottom, a_right_front, a_right_back;
  double a_top_front, a_top_back, a_bottom_front, a_bottom_back;
  double fxx, fyy, fzz, fxy, fxz, fyz;
  double a_top_temp, a_bottom_temp, a_front_temp, a_back_temp;
  double a_top_front_temp, a_top_back_temp;
  double a_bottom_front_temp, a_bottom_back_temp;

  int f_k = (ny + 2 * 2) * (nx + 2 * 2);
  int f_j = (nx + 2 * 2);
  int offset = 2 * f_k + 2 * f_j + 2;

  // Id of thread in the warp.
  int WARPSIZE_X = hipBlockDim_x;
  int WARPSIZE_Y = hipBlockDim_y;
  int WARPSIZE_Z = hipBlockDim_z;
  int localId_x = hipThreadIdx_x % WARPSIZE_X;
  int localId_y = hipThreadIdx_y % WARPSIZE_Y;
  int localId_z = hipThreadIdx_z % WARPSIZE_Z;

  // The first index of output element computed by this warp.
  int startOfWarp_x = hipBlockDim_x * hipBlockIdx_x + WARPSIZE_X * (hipThreadIdx_x / WARPSIZE_X);
  int startOfWarp_y = hipBlockDim_y * hipBlockIdx_y + WARPSIZE_Y * (hipThreadIdx_y / WARPSIZE_Y);
  int startOfWarp_z = hipBlockDim_z * hipBlockIdx_z + WARPSIZE_Z * (hipThreadIdx_z / WARPSIZE_Z);

  // The Id of the thread in the scope of the grid.
  i = localId_x + startOfWarp_x;
  j = localId_y + startOfWarp_y;
  k = localId_z + startOfWarp_z;
  // int globalId = k * mesh.nx * mesh.ny + j * mesh.nx + i;

  //////////////--left--middle--right///////////////////////////////////////////////////
  double rc[19];
  int globalId = k * f_k + j * f_j + i + offset;
  rc[0] = fa[k * f_k + j * f_j + i + offset];

  // if (localId_x ==(WARPSIZE_X-1) && globalId - WARPSIZE_X >=0 ) //left
  if (localId_x == (WARPSIZE_X - 1) && globalId - offset >= 0)
  {
    rc[1] = fa[globalId - WARPSIZE_X];                        // left
    rc[7] = fa[globalId - (nx + 4) - WARPSIZE_X];             // left_top
    rc[9] = fa[globalId + (nx + 4) - WARPSIZE_X];             // left_bottom
    rc[11] = fa[globalId - (ny + 4) * (nx + 4) - WARPSIZE_X]; // left_front
    rc[13] = fa[globalId + (ny + 4) * (nx + 4) - WARPSIZE_X]; // left_back
  }
  if (localId_x == 0 && WARPSIZE_X + globalId < (nx + 4) * (ny + 4) * (nz + 4))
  {
    rc[2] = fa[WARPSIZE_X + globalId];                        // right
    rc[8] = fa[globalId - (nx + 4) + WARPSIZE_X];             // right_top
    rc[10] = fa[globalId + (nx + 4) + WARPSIZE_X];            // right_bottom
    rc[12] = fa[globalId - (ny + 4) * (nx + 4) + WARPSIZE_X]; // right_front
    rc[14] = fa[globalId + (ny + 4) * (nx + 4) + WARPSIZE_X]; // right_back
  }
  if (localId_y == (WARPSIZE_Y - 1) && globalId - offset >= 0)
  {
    rc[3] = fa[globalId - WARPSIZE_Y * (nx + 4)];                        // top
    rc[15] = fa[globalId - (ny + 4) * (nx + 4) - WARPSIZE_Y * (nx + 4)]; // top_front
    rc[16] = fa[globalId + (ny + 4) * (nx + 4) - WARPSIZE_Y * (nx + 4)]; // top_back
  }
  if (localId_y == 0 && globalId + WARPSIZE_Y * (nx + 4) < (nx + 4) * (ny + 4) * (nz + 4))
  {
    rc[4] = fa[globalId + WARPSIZE_Y * (nx + 4)];                        // bottom
    rc[17] = fa[globalId - (ny + 4) * (nx + 4) + WARPSIZE_Y * (nx + 4)]; // bottom_back
    rc[18] = fa[globalId + (ny + 4) * (nx + 4) + WARPSIZE_Y * (nx + 4)]; // bottom_back
  }
  if (localId_z == (WARPSIZE_Z - 1) && globalId - offset >= 0)
  {
    rc[5] = fa[globalId - WARPSIZE_Z * (ny + 4) * (nx + 4)]; // front
  }
  if (localId_z == 0 && globalId + WARPSIZE_Z * (ny + 4) * (nx + 4) < (nx + 4) * (ny + 4) * (nz + 4))
  {
    rc[6] = fa[WARPSIZE_Z * (ny + 4) * (nx + 4) + globalId]; // back
  }
  double toShare = rc[0];
  a_middle = __shfl(toShare, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + localId_y * WARPSIZE_X + localId_x), 64);
  if (localId_x == (WARPSIZE_X - 1)) // left
  {
    toShare = rc[1];
  }
  a_left = __shfl(toShare, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + localId_y * WARPSIZE_X + localId_x - 1 + WARPSIZE_X) % WARPSIZE_X, WARPSIZE_X);
  //////
  if (localId_x == (WARPSIZE_X - 1))
  {
    toShare = rc[0];
  }
  if (localId_x == 0) // right
  {
    toShare = rc[2];
  }
  a_right = __shfl(toShare, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + localId_y * WARPSIZE_X + localId_x + 1 + WARPSIZE_X) % WARPSIZE_X, WARPSIZE_X);
  //////
  if (localId_x == 0) // right
  {
    toShare = rc[0];
  }
  if (localId_y == (WARPSIZE_Y - 1)) // top
  {
    toShare = rc[3];
  }
  a_top = __shfl(toShare, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y - 1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y * WARPSIZE_X);
  //////
  if (localId_y == (WARPSIZE_Y - 1)) // top
  {
    toShare = rc[0];
  }
  if (localId_y == 0) // bottom
  {
    toShare = rc[4];
  }
  a_bottom = __shfl(toShare, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y + 1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y * WARPSIZE_X);
  //////
  if (localId_y == 0) // bottom
  {
    toShare = rc[0];
  }
  if (localId_z == (WARPSIZE_Z - 1)) // front
  {
    toShare = rc[5];
  }
  a_front = __shfl(toShare, ((localId_z - 1) * WARPSIZE_Y * WARPSIZE_X + localId_y * WARPSIZE_X + localId_x), 64);
  //////
  if (localId_z == (WARPSIZE_Z - 1)) // front
  {
    toShare = rc[0];
  }
  if (localId_z == 0) // back
  {
    toShare = rc[6];
  }
  a_back = __shfl(toShare, ((localId_z + 1) * WARPSIZE_Y * WARPSIZE_X + localId_y * WARPSIZE_X + localId_x), 64);

  if (left < 0)
  {
    if (i == ix1)
    {
      a_left = a_right;
    }
  }
  if (right < 0)
  {
    if (i == ix4 - 1)
    {
      a_right = a_left;
    }
  }
  if (top < 0)
  {
    if (j == iy1)
    {
      a_top = a_bottom;
    }
  }
  if (bottom < 0)
  {
    if (j == iy4 - 1)
    {
      a_bottom = a_top;
    }
  }
  if (front < 0)
  {
    if (k == iz1)
    {
      a_front = a_back;
    }
  }
  if (back < 0)
  {
    if (k == iz4 - 1)
    {
      a_back = a_front;
    }
  }
  //////
  /*
     if (localId_z == 0) //back
     {
         toShare = rc[0]; //The following code is not used for toShare.
     }
 */
  a_top_temp = a_top;
  a_bottom_temp = a_bottom;
  a_front_temp = a_front;
  a_back_temp = a_back;
  if (localId_x == (WARPSIZE_X - 1))
  {
    a_top_temp = rc[7];    // left_top
    a_bottom_temp = rc[9]; // left_bottom
    a_front_temp = rc[11]; // left_front
    a_back_temp = rc[13];  // left_back
  }
  a_left_top = __shfl(a_top_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y)*WARPSIZE_X + localId_x - 1 + WARPSIZE_X) % (WARPSIZE_X), WARPSIZE_X);
  a_left_bottom = __shfl(a_bottom_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y)*WARPSIZE_X + localId_x - 1 + WARPSIZE_X) % (WARPSIZE_X), WARPSIZE_X);
  a_left_front = __shfl(a_front_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y)*WARPSIZE_X + localId_x - 1 + WARPSIZE_X) % (WARPSIZE_X), WARPSIZE_X);
  a_left_back = __shfl(a_back_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y)*WARPSIZE_X + localId_x - 1 + WARPSIZE_X) % (WARPSIZE_X), WARPSIZE_X);

  //////
  if (localId_x == (WARPSIZE_X - 1))
  {
    a_top_temp = a_top;       // left_top
    a_bottom_temp = a_bottom; // left_bottom
    a_front_temp = a_front;   // left_front
    a_back_temp = a_back;     // left_back
  }
  if (localId_x == 0)
  {
    a_top_temp = rc[8];     // right_top
    a_bottom_temp = rc[10]; // right_bottom
    a_front_temp = rc[12];  // right_front
    a_back_temp = rc[14];   // right_back
  }
  a_right_top = __shfl(a_top_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y)*WARPSIZE_X + localId_x + 1 + WARPSIZE_X) % (WARPSIZE_X), WARPSIZE_X);
  a_right_bottom = __shfl(a_bottom_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y)*WARPSIZE_X + localId_x + 1 + WARPSIZE_X) % (WARPSIZE_X), WARPSIZE_X);
  a_right_front = __shfl(a_front_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y)*WARPSIZE_X + localId_x + 1 + WARPSIZE_X) % (WARPSIZE_X), WARPSIZE_X);
  a_right_back = __shfl(a_back_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y)*WARPSIZE_X + localId_x + 1 + WARPSIZE_X) % (WARPSIZE_X), WARPSIZE_X);

  //////

  a_top_front_temp = a_front;
  a_top_back_temp = a_back;
  if (localId_y == (WARPSIZE_Y - 1))
  {
    a_top_front_temp = rc[15]; // top_front
    a_top_back_temp = rc[16];  // top_back
  }
  a_top_front = __shfl(a_top_front_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y - 1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y * WARPSIZE_X);
  a_top_back = __shfl(a_top_back_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y - 1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y * WARPSIZE_X);
  //////
  a_bottom_front_temp = a_front;
  a_bottom_back_temp = a_back;
  if (localId_y == 0)
  {
    a_bottom_front_temp = rc[17]; // bottom_front
    a_bottom_back_temp = rc[18];  // bottom_back
  }
  a_bottom_front = __shfl(a_bottom_front_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y + 1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y * WARPSIZE_X);
  a_bottom_back = __shfl(a_bottom_back_temp, ((localId_z)*WARPSIZE_Y * WARPSIZE_X + (localId_y + 1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y * WARPSIZE_X);

  double tmp = 0.0;
  fxx = (a_right - 2 * a_middle + a_left) / hx / hx;
  fyy = (a_bottom - 2 * a_middle + a_top) / hy / hy;
  fzz = (a_back - 2 * a_middle + a_front) / hz / hz;
  fxy = ((a_right_bottom - a_left_bottom) - (a_right_top - a_left_top)) * 0.25 / hx / hy;
  fxz = ((a_right_back - a_left_back) - (a_right_front - a_left_front)) * 0.25 / hx / hz;
  fyz = ((a_bottom_back - a_top_back) - (a_bottom_front - a_top_front)) * 0.25 / hy / hz;
  int o = orie[(k+nghost) * (nx+4) * (ny+4) + (j+nghost) * (nx+4) + (i+nghost)];
  tmp += lambda[0 * 3 + 0 + (o - 1) * 3 * 3 + n * ORI * 3 * 3] * fxx;
  tmp += lambda[0 * 3 + 1 + (o - 1) * 3 * 3 + n * ORI * 3 * 3] * fxy;
  tmp += lambda[0 * 3 + 2 + (o - 1) * 3 * 3 + n * ORI * 3 * 3] * fxz;
  tmp += lambda[1 * 3 + 0 + (o - 1) * 3 * 3 + n * ORI * 3 * 3] * fxy;
  tmp += lambda[1 * 3 + 1 + (o - 1) * 3 * 3 + n * ORI * 3 * 3] * fyy;
  tmp += lambda[1 * 3 + 2 + (o - 1) * 3 * 3 + n * ORI * 3 * 3] * fyz;
  tmp += lambda[2 * 3 + 0 + (o - 1) * 3 * 3 + n * ORI * 3 * 3] * fxz;
  tmp += lambda[2 * 3 + 1 + (o - 1) * 3 * 3 + n * ORI * 3 * 3] * fyz;
  tmp += lambda[2 * 3 + 2 + (o - 1) * 3 * 3 + n * ORI * 3 * 3] * fzz;
 //tmp += lambda[0 * 3 + 0] * fxx;
 //tmp += lambda[0 * 3 + 1] * fxy;
 //tmp += lambda[0 * 3 + 2] * fxz;
 //tmp += lambda[1 * 3 + 0] * fxy;
 //tmp += lambda[1 * 3 + 1] * fyy;
 //tmp += lambda[1 * 3 + 2] * fyz;
 //tmp += lambda[2 * 3 + 0] * fxz;
 //tmp += lambda[2 * 3 + 1] * fyz;
 //tmp += lambda[2 * 3 + 2] * fzz;
  f[k * nx * ny + j * nx + i] = tmp;
}

__global__ void
anisotropic_calc_dev(double *f, double *fa, double *lambda, int n)
{
  int i, j, k;
  int n_left, n_right, n_top, n_bottom, n_front, n_back;
  double a_left, a_right, a_top, a_bottom, a_front, a_back, a_middle;
  double a_left_top, a_left_bottom, a_left_front, a_left_back;
  double a_right_top, a_right_bottom, a_right_front, a_right_back;
  double a_top_front, a_top_back, a_bottom_front, a_bottom_back;
  double fxx, fyy, fzz, fxy, fxz, fyz;

  int f_k = (ny + 2 * 2) * (nx + 2 * 2);
  int f_j = (nx + 2 * 2);
  int offset = 2 * f_k + 2 * f_j + 2;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; //  iz4, iy4, ix4 (k,j,i)
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y; //  iz1, iy1, ix1
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  n_left = i - 1;
  if (left < 0)
  {
    if (i == ix1)
    {
      n_left = i + 1;
    }
  }

  n_right = i + 1;
  if (right < 0)
  {
    if (i == ix4 - 1)
    {
      n_right = i - 1;
    }
  }

  n_top = j - 1;
  if (top < 0)
  {
    if (j == iy1)
    {
      n_top = j + 1;
    }
  }

  n_bottom = j + 1;
  if (bottom < 0)
  {
    if (j == iy4 - 1)
    {
      n_bottom = j - 1;
    }
  }

  n_front = k - 1;
  if (front < 0)
  {
    if (k == iz1)
    {
      n_front = k + 1;
    }
  }

  n_back = k + 1;
  if (back < 0)
  {
    if (k == iz4 - 1)
    {
      n_back = k - 1;
    }
  }

  a_middle = fa[k * f_k + j * f_j + i + offset];

  a_front = fa[n_front * f_k + j * f_j + i + offset];
  a_back = fa[n_back * f_k + j * f_j + i + offset];
  a_top = fa[k * f_k + n_top * f_j + i + offset];
  a_bottom = fa[k * f_k + n_bottom * f_j + i + offset];
  a_right = fa[k * f_k + j * f_j + n_right + offset];
  a_left = fa[k * f_k + j * f_j + n_left + offset];

  a_left_top = fa[k * f_k + n_top * f_j + n_left + offset];
  a_left_bottom = fa[k * f_k + n_bottom * f_j + n_left + offset];
  a_left_front = fa[n_front * f_k + j * f_j + n_left + offset];
  a_left_back = fa[n_back * f_k + j * f_j + n_left + offset];

  a_right_top = fa[k * f_k + n_top * f_j + n_right + offset];
  a_right_bottom = fa[k * f_k + n_bottom * f_j + n_right + offset];
  a_right_front = fa[n_front * f_k + j * f_j + n_right + offset];
  a_right_back = fa[n_back * f_k + j * f_j + n_right + offset];

  a_top_front = fa[n_front * f_k + n_top * f_j + i + offset];
  a_top_back = fa[n_back * f_k + n_top * f_j + i + offset];

  a_bottom_front = fa[n_front * f_k + n_bottom * f_j + i + offset];
  a_bottom_back = fa[n_back * f_k + n_bottom * f_j + i + offset];

  double tmp = 0.0;
  fxx = (a_right - 2 * a_middle + a_left) / hx / hx;
  fyy = (a_bottom - 2 * a_middle + a_top) / hy / hy;
  fzz = (a_back - 2 * a_middle + a_front) / hz / hz;
  fxy = ((a_right_bottom - a_left_bottom) - (a_right_top - a_left_top)) / 2.0 / hx / 2.0 / hy;
  fxz = ((a_right_back - a_left_back) - (a_right_front - a_left_front)) / 2.0 / hx / 2.0 / hz;
  fyz = ((a_bottom_back - a_top_back) - (a_bottom_front - a_top_front)) / 2.0 / hy / 2.0 / hz;
  tmp += lambda[0 * 3 + 0] * fxx;
  tmp += lambda[0 * 3 + 1] * fxy;
  tmp += lambda[0 * 3 + 2] * fxz;
  tmp += lambda[1 * 3 + 0] * fxy;
  tmp += lambda[1 * 3 + 1] * fyy;
  tmp += lambda[1 * 3 + 2] * fyz;
  tmp += lambda[2 * 3 + 0] * fxz;
  tmp += lambda[2 * 3 + 1] * fyz;
  tmp += lambda[2 * 3 + 2] * fzz;
  f[k * nx * ny + j * nx + i] = tmp;
}

__global__ void
top_bottom_pack(double *field, double *fields_top, double *fields_bottom, double *fieldr_front, double *fieldr_back)
{
  int k_s, k_e;
  int f_j, f_k;
  int tb_j, tb_k;
  int t_ofst, b_ofst, i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  k_s = nghost + 2;
  k_e = nz + (nghost + 2);
  f_j = nx;
  f_k = nx * ny;
  tb_j = nx;
  tb_k = nx * (nghost + 2);
  t_ofst = nx * nghost;
  b_ofst = nx * (ny - nghost - (nghost + 2));

  if (k < k_s)
  {
    if (front >= 0)
    {
      fields_top[tb_k * k + tb_j * j + i] = fieldr_front[f_k * k + f_j * j + i + t_ofst];
      fields_bottom[tb_k * k + tb_j * j + i] = fieldr_front[f_k * k + f_j * j + i + b_ofst];
    }
  }
  else if (k >= k_e)
  {
    if (back >= 0)
    {
      fields_top[tb_k * k + tb_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * j + i + t_ofst];
      fields_bottom[tb_k * k + tb_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * j + i + b_ofst];
    }
  }
  else
  {
    fields_top[tb_k * k + tb_j * j + i] = field[f_k * (k - k_s) + f_j * j + i + t_ofst];
    fields_bottom[tb_k * k + tb_j * j + i] = field[f_k * (k - k_s) + f_j * j + i + b_ofst];
  }
}

__global__ void
left_right_pack(double *field, double *fields_left, double *fields_right, double *fieldr_top, double *fieldr_bottom, double *fieldr_front, double *fieldr_back)
{
  int j_s, j_e, k_s, k_e;
  int f_j, f_k;
  int lr_j, lr_k;
  int tb_j, tb_k;
  int l_ofst, r_ofst, i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  j_s = nghost + 2;
  j_e = ny + (nghost + 2);
  k_s = nghost + 2;
  k_e = nz + (nghost + 2);
  f_j = nx;
  f_k = nx * ny;
  lr_j = nghost + 2;
  lr_k = (nghost + 2) * (ny + (nghost + 2) * 2);
  tb_j = nx;
  tb_k = nx * (nghost + 2);
  l_ofst = nghost;
  r_ofst = nx - nghost - (nghost + 2);

  if (j < j_s)
  {
    if (top >= 0)
    {
      fields_left[lr_k * k + lr_j * j + i] = fieldr_top[tb_k * k + tb_j * j + i + l_ofst];
      fields_right[lr_k * k + lr_j * j + i] = fieldr_top[tb_k * k + tb_j * j + i + r_ofst];
    }
  }
  else if (j >= j_e)
  {
    if (bottom >= 0)
    {
      fields_left[lr_k * k + lr_j * j + i] = fieldr_bottom[tb_k * k + tb_j * (j - j_e) + i + l_ofst];
      fields_right[lr_k * k + lr_j * j + i] = fieldr_bottom[tb_k * k + tb_j * (j - j_e) + i + r_ofst];
    }
  }
  else
  {
    if (k < k_s)
    {
      if (front >= 0)
      {
        fields_left[lr_k * k + lr_j * j + i] = fieldr_front[f_k * k + f_j * (j - j_s) + i + l_ofst];
        fields_right[lr_k * k + lr_j * j + i] = fieldr_front[f_k * k + f_j * (j - j_s) + i + r_ofst];
      }
    }
    else if (k >= k_e)
    {
      if (back >= 0)
      {
        fields_left[lr_k * k + lr_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * (j - j_s) + i + l_ofst];
        fields_right[lr_k * k + lr_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * (j - j_s) + i + r_ofst];
      }
    }
    else
    {
      fields_left[lr_k * k + lr_j * j + i] = field[f_k * (k - k_s) + f_j * (j - j_s) + i + l_ofst];
      fields_right[lr_k * k + lr_j * j + i] = field[f_k * (k - k_s) + f_j * (j - j_s) + i + r_ofst];
    }
  }
}

__global__ void
left_right_unpack(double *field, double *fieldr_left, double *fieldr_right)
{
  int f_j, f_k;
  int lr_k, lr_j;
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f_j = nx;
  f_k = nx * ny;
  lr_j = nghost + 2;
  lr_k = (nghost + 2) * (ny + (nghost + 2) * 2);

  if (left >= 0)
  {
    field[f_k * k + f_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * (j + (nghost + 2)) + i + 2];
  }
  if (right >= 0)
  {
    field[f_k * k + f_j * j + i + (nx - nghost)] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * (j + (nghost + 2)) + i];
  }
}

__global__ void
top_bottom_unpack(double *field, double *fieldr_top, double *fieldr_bottom)
{

  int f_j, f_k;
  int tb_k, tb_j;
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f_j = nx;
  f_k = nx * ny;
  tb_j = nx;
  tb_k = nx * (nghost + 2);

  if (top >= 0)
  {
    field[f_k * k + f_j * j + i] = fieldr_top[tb_k * (k + (nghost + 2)) + tb_j * (j + 2) + i];
  }
  if (bottom >= 0)
  {
    field[f_k * k + f_j * (j + ny - nghost) + i] = fieldr_bottom[tb_k * (k + (nghost + 2)) + tb_j * j + i];
  }
}

__global__ void
front_back_unpack(double *field, double *fieldr_front, double *fieldr_back)
{

  int f_j, f_k;
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f_j = nx;
  f_k = nx * ny;

  if (front >= 0)
  {
    field[f_k * k + f_j * j + i] = fieldr_front[f_k * (k + 2) + f_j * j + i];
  }
  if (back >= 0)
  {
    field[f_k * (k + nz - nghost) + f_j * j + i] = fieldr_back[f_k * k + f_j * j + i];
  }
}

__global__ void
left_right_enlarge(double *fielde_left, double *fielde_right, double *fieldr_left, double *fieldr_right)
{
  int j_s, j_e, k_s, k_e;
  int elr_k, elr_j;
  int lr_k, lr_j;
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  j_s = nghost + 2;
  j_e = j_s + ny - (nghost + 2);

  if (top < 0)
  {
    j_s -= nghost;
  }
  if (bottom < 0)
  {
    j_e += nghost;
  }
  k_s = nghost + 2;
  k_e = k_s + nz - (nghost + 2);
  if (front < 0)
  {
    k_s -= nghost;
  }
  if (back < 0)
  {
    k_e += nghost;
  }

  elr_j = nghost + 2;
  elr_k = (nghost + 2) * (ny + 4);
  lr_j = nghost + 2;
  lr_k = (nghost + 2) * (ny + (nghost + 2) * 2);

  if (left >= 0)
  {
    if (front >= 0 && k < k_s)
    {
      if (j >= 2 && j < ny + 2)
      {
        fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * k + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2)
      {
        fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * k + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny)
      {
        fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * k + lr_j * (j + (nghost + 2)) + i];
      }
    }
    if (k >= k_s && k < k_e)
    {
      if (j >= 2 && j < ny + 2)
      {
        fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + nghost) + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2)
      {
        fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + nghost) + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny)
      {
        fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + nghost) + lr_j * (j + (nghost + 2)) + i];
      }
    }
    if (back >= 0 && k >= k_e)
    {
      if (j >= 2 && j < ny + 2)
      {
        fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2)
      {
        fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny)
      {
        fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * (j + (nghost + 2)) + i];
      }
    }
  }

  if (right >= 0)
  {
    if (front >= 0 && k < k_s)
    {
      if (j >= 2 && j < ny + 2)
      {
        fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * k + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2)
      {
        fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * k + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny)
      {
        fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * k + lr_j * (j + (nghost + 2)) + i];
      }
    }
    if (k >= k_s && k < k_e)
    {
      if (j >= 2 && j < ny + 2)
      {
        fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + nghost) + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2)
      {
        fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + nghost) + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny)
      {
        fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + nghost) + lr_j * (j + (nghost + 2)) + i];
      }
    }
    if (back >= 0 && k >= k_e)
    {
      if (j >= 2 && j < ny + 2)
      {
        fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2)
      {
        fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny)
      {
        fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * (j + (nghost + 2)) + i];
      }
    }
  }
}

__global__ void
top_bottom_enlarge(double *fielde_top, double *fielde_bottom, double *fieldr_left, double *fieldr_right, double *fieldr_top, double *fieldr_bottom)
{
  int j_s, j_e, k_s, k_e;
  int etb_k, etb_j;
  int lr_k, lr_j, tb_k, tb_j;
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  j_s = nghost + 2;
  j_e = j_s + ny - (nghost + 2);
  if (top < 0)
  {
    j_s -= nghost;
  }
  if (bottom < 0)
  {
    j_e += nghost;
  }
  k_s = nghost + 2;
  k_e = k_s + nz - (nghost + 2);
  if (front < 0)
  {
    k_s -= nghost;
  }
  if (back < 0)
  {
    k_e += nghost;
  }

  etb_j = nx + 4;
  etb_k = (nx + 4) * (nghost + 2);
  lr_j = nghost + 2;
  lr_k = (nghost + 2) * (ny + (nghost + 2) * 2);
  tb_j = nx;
  tb_k = nx * (nghost + 2);

  if (top >= 0)
  {
    if (front >= 0 && k < k_s)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_top[etb_k * k + etb_j * j + i] = fieldr_top[tb_k * k + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_top[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * k + lr_j * j + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_top[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * k + lr_j * j + (i - nx)];
      }
    }
    if (k >= k_s && k < k_e)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_top[etb_k * k + etb_j * j + i] = fieldr_top[tb_k * (k + nghost) + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_top[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * (k + nghost) + lr_j * j + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_top[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * (k + nghost) + lr_j * j + (i - nx)];
      }
    }
    if (back >= 0 && k >= k_e)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_top[etb_k * k + etb_j * j + i] = fieldr_top[tb_k * (k + (nghost + 2)) + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_top[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * j + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_top[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * j + (i - nx)];
      }
    }
  }

  if (bottom >= 0)
  {
    if (front >= 0 && k < k_s)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_bottom[tb_k * k + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * k + lr_j * (j + (ny + (nghost + 2))) + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * k + lr_j * (j + (ny + (nghost + 2))) + (i - nx)];
      }
    }
    if (k >= k_s && k < k_e)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_bottom[tb_k * (k + nghost) + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * (k + nghost) + lr_j * (j + (ny + (nghost + 2))) + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * (k + nghost) + lr_j * (j + (ny + (nghost + 2))) + (i - nx)];
      }
    }
    if (back >= 0 && k >= k_e)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_bottom[tb_k * (k + (nghost + 2)) + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * (j + (ny + (nghost + 2))) + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * (j + (ny + (nghost + 2))) + (i - nx)];
      }
    }
  }
}

__global__ void
front_back_enlarge(double *fielde_front, double *fielde_back, double *fieldr_left, double *fieldr_right, double *fieldr_top, double *fieldr_bottom, double *fieldr_front, double *fieldr_back)
{
  int j_s, j_e, k_s, k_e;
  int efb_k, efb_j;
  int lr_k, lr_j, tb_k, tb_j, fb_k, fb_j;
  int b_ofst;
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  j_s = nghost + 2;
  j_e = j_s + ny - (nghost + 2);
  if (top < 0)
  {
    j_s -= nghost;
  }
  if (bottom < 0)
  {
    j_e += nghost;
  }
  k_s = nghost + 2;
  k_e = k_s + nz - (nghost + 2);
  if (front < 0)
  {
    k_s -= nghost;
  }
  if (back < 0)
  {
    k_e += nghost;
  }

  efb_j = (nx + 4) * (nghost + 2);
  efb_k = nx + 4;
  lr_j = nghost + 2;
  lr_k = (nghost + 2) * (ny + (nghost + 2) * 2);
  tb_j = nx;
  tb_k = nx * (nghost + 2);
  fb_j = nx;
  fb_k = nx * ny;
  b_ofst = nz + (nghost + 2);

  if (front >= 0)
  {
    if (top >= 0 && j < j_s)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_front[efb_k * k + efb_j * j + i] = fieldr_top[tb_k * k + tb_j * j + i - 2];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_front[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * k + lr_j * j + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_front[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * k + lr_j * j + i - nx];
      }
    }
    if (j >= j_s && j < j_e)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_front[efb_k * k + efb_j * j + i] = fieldr_front[fb_k * k + fb_j * (j - nghost) + i - 2];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_front[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * k + lr_j * (j + nghost) + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_front[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * k + lr_j * (j + nghost) + i - nx];
      }
    }
    if (bottom >= 0 && j >= j_e)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_front[efb_k * k + efb_j * j + i] = fieldr_bottom[tb_k * k + tb_j * (j - j_e) + i - 2];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_front[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * k + lr_j * (j + (nghost + 2)) + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_front[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * k + lr_j * (j + (nghost + 2)) + i - nx];
      }
    }
  }
  if (back >= 0)
  {
    if (top >= 0 && j < j_s)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_back[efb_k * k + efb_j * j + i] = fieldr_top[tb_k * (k + b_ofst) + tb_j * j + i - 2];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_back[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * (k + b_ofst) + lr_j * j + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_back[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * (k + b_ofst) + lr_j * j + i - nx];
      }
    }
    if (j >= j_s && j < j_e)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_back[efb_k * k + efb_j * j + i] = fieldr_back[fb_k * k + fb_j * (j - nghost) + i - 2];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_back[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * (k + b_ofst) + lr_j * (j + nghost) + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_back[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * (k + b_ofst) + lr_j * (j + nghost) + i - nx];
      }
    }
    if (bottom >= 0 && j >= j_e)
    {
      if (i >= 2 && i < nx + 2)
      {
        fielde_back[efb_k * k + efb_j * j + i] = fieldr_bottom[tb_k * (k + b_ofst) + tb_j * (j - j_e) + i - 2];
      }
      if (left >= 0 && i < nghost + 2)
      {
        fielde_back[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * (k + b_ofst) + lr_j * (j + (nghost + 2)) + i];
      }
      if (right >= 0 && i >= nx)
      {
        fielde_back[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * (k + b_ofst) + lr_j * (j + (nghost + 2)) + i - nx];
      }
    }
  }
}

__global__ void
left_right_mu(double *Eu_left, double *Eu_right, double *Ee_left, double *Ee_right)
{
  int i, j, k;
  int mu_i, mu_j, mu_k;
  int e_i, e_j, e_k;
  int l_e, l_mu;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y + 2;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z + 2;

  mu_k = ny;
  e_j = nghost + 2;
  e_k = (nghost + 2) * (ny + 4);

  if (left >= 0)
  {

    if (k == 2 && front < 0)
    {
      l_e = k * e_k + j * e_j + 1;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_left[l_mu] = Ee_left[l_e];
    }
    else if (k == nz + 1 && back < 0)
    {
      l_e = k * e_k + j * e_j + 1;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_left[l_mu] = Ee_left[l_e];
    }
    else
    {
      l_e = k * e_k + j * e_j + 1;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_left[l_mu] = Ee_left[l_e];
    }
  }

  if (right >= 0)
  {

    if (k == 2 && front < 0)
    {
      l_e = k * e_k + j * e_j + 2;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_right[l_mu] = Ee_right[l_e];
    }
    else if (k == nz + 1 && back < 0)
    {
      l_e = k * e_k + j * e_j + 2;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_right[l_mu] = Ee_right[l_e];
    }
    else
    {
      l_e = k * e_k + j * e_j + 2;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_right[l_mu] = Ee_right[l_e];
    }
  }
}

__global__ void
top_bottom_mu(double *Eu_top, double *Eu_bottom, double *Ee_top, double *Ee_bottom)
{
  int i, j, k;
  int mu_i, mu_j, mu_k;
  int e_i, e_j, e_k;
  int l_e, l_mu;

  i = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  j = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x + 2;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z + 2;

  mu_k = nx;
  e_j = nx + 4;
  e_k = (nx + 4) * (nghost + 2);

  if (top >= 0)
  {

    if (k == 2 && front < 0)
    {
      l_e = k * e_k + e_j + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_top[l_mu] = Ee_top[l_e];
    }
    else if (k == nz + 1 && back < 0)
    {
      l_e = k * e_k + e_j + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_top[l_mu] = Ee_top[l_e];
    }
    else
    {
      l_e = k * e_k + e_j + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_top[l_mu] = Ee_top[l_e];
    }
  }

  if (bottom >= 0)
  {

    if (k == 2 && front < 0)
    {
      l_e = k * e_k + e_j * nghost + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_bottom[l_mu] = Ee_bottom[l_e];
    }
    else if (k == nz + 1 && back < 0)
    {
      l_e = k * e_k + e_j * nghost + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_bottom[l_mu] = Ee_bottom[l_e];
    }
    else
    {
      l_e = k * e_k + e_j * nghost + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_bottom[l_mu] = Ee_bottom[l_e];
    }
  }
}

__global__ void
front_back_mu(Dtype *Eu_front, Dtype *Eu_back, Dtype *Ee_front, Dtype *Ee_back)
{
  int i, j, k;
  int mu_i, mu_j, mu_k;
  int e_i, e_j, e_k;
  int l_e, l_mu;

  i = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  j = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x + 2;
  k = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y + 2;

  mu_k = nx;
  e_j = nx + 4;
  e_k = (nx + 4) * (nghost + 2);

  if (front >= 0)
  {

    l_e = k * e_k + e_j + j;
    l_mu = (k - 2) * mu_k + j - 2;
    Eu_front[l_mu] = Ee_front[l_e];
  }

  if (back >= 0)
  {

    l_e = k * e_k + e_j * nghost + j;
    l_mu = (k - 2) * mu_k + j - 2;
    Eu_back[l_mu] = Ee_back[l_e];
  }
}

// tsb
__global__ void
unpack_lr(double *fieldr, double *fieldr_left, double *fieldr_right)
{
  int f_j, f_k;
  int lr_k, lr_j;
  int i, j, k;

  f_j = nx + 2 * 2;
  f_k = (nx + 2 * 2) * (ny + 2 * 2);
  lr_j = nghost + 2;
  lr_k = (nghost + 2) * (ny + (nghost + 2) * 2);

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; //  nz+2*2 , ny+2*2 , nghost+2 (k,j,i)
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  if (left >= 0)
  {
    fieldr[f_k * k + f_j * j + i] = fieldr_left[lr_k * (k + 2) + lr_j * (j + 2) + i];
  }
  if (right >= 0)
  {
    fieldr[f_k * k + f_j * j + i + nx] = fieldr_right[lr_k * (k + 2) + lr_j * (j + 2) + i];
  }
}

__global__ void
unpack_tb(double *fieldr, double *fieldr_top, double *fieldr_bottom)
{
  int f_j, f_k;
  int tb_k, tb_j;
  int i, j, k;
  f_j = nx + 2 * 2;
  f_k = (nx + 2 * 2) * (ny + 2 * 2);
  tb_j = nx;
  tb_k = nx * (nghost + 2);
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; //  nz+2*2 ,  nghost+2, nx (k,j,i)
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  if (top >= 0)
  {
    fieldr[f_k * k + f_j * j + i + 2] = fieldr_top[tb_k * (k + 2) + tb_j * j + i];
  }
  if (bottom >= 0)
  {
    fieldr[f_k * k + f_j * (j + ny) + i + 2] = fieldr_bottom[tb_k * (k + 2) + tb_j * j + i];
  }
}

__global__ void
unpack_fb(double *fieldr, double *fieldr_front, double *fieldr_back)
{
  int f_j, f_k;
  int fb_j, fb_k;
  int i, j, k;

  f_j = nx + 2 * 2;
  f_k = (nx + 2 * 2) * (ny + 2 * 2);
  fb_j = nx;
  fb_k = nx * ny;
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; //  nghost+2, ny, nx (k,j,i)
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  if (front >= 0)
  {
    fieldr[f_k * k + f_j * (j + 2) + (i + 2)] = fieldr_front[fb_k * k + fb_j * j + i];
  }

  if (back >= 0)
  {
    fieldr[f_k * (k + nz) + f_j * (j + 2) + (i + 2)] = fieldr_back[fb_k * k + fb_j * j + i];
  }
}

__global__ void
unpack_all(double *fieldr, double *field)
{
  int f_j, f_k;
  int i, j, k;
  f_j = nx + 2 * 2;
  f_k = (nx + 2 * 2) * (ny + 2 * 2);
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; //  nz, ny, nx (k,j,i)
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  fieldr[f_k * (k + 2) + f_j * (j + 2) + (i + 2)] = field[nx * ny * k + nx * j + i];
}
__global__ void
Ee_dev(double *fieldE, double *fielde)
{
  int i, j, k;
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  fieldE[k * nx * ny + j * nx + i] = fielde[nx * ny * k + nx * j + i];
}

void transfer(void)
{
  int n;

  int threads_x, threads_y, threads_z;

  for (n = 0; n < nac; n++)
  {

#ifdef SCLETD_DEBUG
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    hipMemcpy(ac[n].fieldEs_front, &fieldE[n * offset + nx * ny * nghost], sizeof(Dtype) * fb_size, hipMemcpyDeviceToHost);
    hipMemcpy(ac[n].fieldEs_back, &fieldE[n * offset + nx * ny * (nz - nghost - (nghost + 2))], sizeof(Dtype) * fb_size, hipMemcpyDeviceToHost);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    MPI_Startall(4, ac[n].ireq_front_back_fieldE);
    MPI_Waitall(4, ac[n].ireq_front_back_fieldE, status);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_MPI_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    hipMemcpy(fieldEr_front + n * fb_size, ac[n].fieldEr_front, sizeof(Dtype) * fb_size, hipMemcpyHostToDevice);
    hipMemcpy(fieldEr_back + n * fb_size, ac[n].fieldEr_back, sizeof(Dtype) * fb_size, hipMemcpyHostToDevice);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    dim3 blocks_tb_pack(nx / THREADS_PER_BLOCK_X, (nghost + 2) / THREADS_PER_BLOCK_Y, (nz + (nghost + 2) * 2) / THREADS_PER_BLOCK_Z);
    dim3 threads_tb_pack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL(top_bottom_pack, blocks_tb_pack, threads_tb_pack, 0, 0,
                       fieldE + n * offset, fieldEs_top, fieldEs_bottom, fieldEr_front + n * fb_size, fieldEr_back + n * fb_size);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_pack_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    hipMemcpy(ac[n].fieldEs_top, fieldEs_top, sizeof(Dtype) * tb_size, hipMemcpyDeviceToHost);
    hipMemcpy(ac[n].fieldEs_bottom, fieldEs_bottom, sizeof(Dtype) * tb_size, hipMemcpyDeviceToHost);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    MPI_Startall(4, ac[n].ireq_top_bottom_fieldE);
    MPI_Waitall(4, ac[n].ireq_top_bottom_fieldE, status);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_MPI_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    hipMemcpy(fieldEr_top + n * tb_size, ac[n].fieldEr_top, sizeof(Dtype) * tb_size, hipMemcpyHostToDevice);
    hipMemcpy(fieldEr_bottom + n * tb_size, ac[n].fieldEr_bottom, sizeof(Dtype) * tb_size, hipMemcpyHostToDevice);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    dim3 blocks_lr_pack((nghost + 2) / THREADS_PER_BLOCK_X, (ny + (nghost + 2) * 2) / THREADS_PER_BLOCK_Y, (nz + (nghost + 2) * 2) / THREADS_PER_BLOCK_Z);
    dim3 threads_lr_pack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL(left_right_pack, blocks_lr_pack, threads_lr_pack, 0, 0,
                       fieldE + n * offset, fieldEs_left, fieldEs_right, fieldEr_top + n * tb_size,
                       fieldEr_bottom + n * tb_size, fieldEr_front + n * fb_size, fieldEr_back + n * fb_size);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_pack_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    hipMemcpy(ac[n].fieldEs_left, fieldEs_left, sizeof(Dtype) * lr_size, hipMemcpyDeviceToHost);
    hipMemcpy(ac[n].fieldEs_right, fieldEs_right, sizeof(Dtype) * lr_size, hipMemcpyDeviceToHost);
#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    MPI_Startall(4, ac[n].ireq_left_right_fieldE);
    MPI_Waitall(4, ac[n].ireq_left_right_fieldE, status);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_MPI_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    hipMemcpy(fieldEr_left + n * lr_size, ac[n].fieldEr_left, sizeof(Dtype) * lr_size, hipMemcpyHostToDevice);
    hipMemcpy(fieldEr_right + n * lr_size, ac[n].fieldEr_right, sizeof(Dtype) * lr_size, hipMemcpyHostToDevice);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    dim3 blocks_lr_enlarge((nghost + 2) / THREADS_PER_BLOCK_X, (ny + 4) / THREADS_PER_BLOCK_Y, (nz + 4) / THREADS_PER_BLOCK_Z);
    dim3 threads_lr_enlarge(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL(left_right_enlarge, blocks_lr_enlarge, threads_lr_enlarge, 0, 0,
                       fieldEe_left, fieldEe_right, fieldEr_left + n * lr_size, fieldEr_right + n * lr_size);

    dim3 blocks_tb_enlarge((nx + 4) / THREADS_PER_BLOCK_X, (nghost + 2) / THREADS_PER_BLOCK_Y, (nz + 4) / THREADS_PER_BLOCK_Z);
    dim3 threads_tb_enlarge(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL(top_bottom_enlarge, blocks_tb_enlarge, threads_tb_enlarge, 0, 0,
                       fieldEe_top, fieldEe_bottom, fieldEr_left + n * lr_size, fieldEr_right + n * lr_size,
                       fieldEr_top + n * tb_size, fieldEr_bottom + n * tb_size);

    dim3 blocks_fb_enlarge((nx + 4) / THREADS_PER_BLOCK_X, (ny + 4) / THREADS_PER_BLOCK_Y, (nghost + 2) / THREADS_PER_BLOCK_Z);
    dim3 threads_fb_enlarge(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL(front_back_enlarge, blocks_fb_enlarge, threads_fb_enlarge, 0, 0,
                       fieldEe_front, fieldEe_back, fieldEr_left + n * lr_size, fieldEr_right + n * lr_size,
                       fieldEr_top + n * tb_size, fieldEr_bottom + n * tb_size, fieldEr_front + n * fb_size,
                       fieldEr_back + n * fb_size);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_enlarge_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    threads_x = 1;
    dim3 blocks_lr_mu(1 / threads_x, ny / THREADS_PER_BLOCK_Y, nz / THREADS_PER_BLOCK_Z);
    dim3 threads_lr_mu(threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL(left_right_mu, blocks_lr_mu, threads_lr_mu, 0, 0,
                       fieldEu_left + n * u_lr, fieldEu_right + n * u_lr, fieldEe_left, fieldEe_right);

    threads_y = 1;
    dim3 blocks_tb_mu(nx / THREADS_PER_BLOCK_X, 1 / threads_y, nz / THREADS_PER_BLOCK_Z);
    dim3 threads_tb_mu(THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL(top_bottom_mu, blocks_tb_mu, threads_tb_mu, 0, 0,
                       fieldEu_top + n * u_tb, fieldEu_bottom + n * u_tb, fieldEe_top, fieldEe_bottom);

    threads_z = 1;
    dim3 blocks_fb_mu(nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, 1 / threads_z);
    dim3 threads_fb_mu(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);
    hipLaunchKernelGGL(front_back_mu, blocks_fb_mu, threads_fb_mu, 0, 0,
                       fieldEu_front + n * u_fb, fieldEu_back + n * u_fb, fieldEe_front, fieldEe_back);

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_mu_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif

    threads_x = nghost;
    dim3 blocks_lr_unpack(nghost / threads_x, ny / THREADS_PER_BLOCK_Y, nz / THREADS_PER_BLOCK_Z);
    dim3 threads_lr_unpack(threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL(left_right_unpack, blocks_lr_unpack, threads_lr_unpack, 0, 0,
                       fieldE + n * offset, fieldEr_left + n * lr_size, fieldEr_right + n * lr_size);
    threads_y = nghost;
    dim3 blocks_tb_unpack(nx / THREADS_PER_BLOCK_X, nghost / threads_y, nz / THREADS_PER_BLOCK_Z);
    dim3 threads_tb_unpack(THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL(top_bottom_unpack, blocks_tb_unpack, threads_tb_unpack, 0, 0,
                       fieldE + n * offset, fieldEr_top + n * tb_size, fieldEr_bottom + n * tb_size);

    threads_z = nghost;
    dim3 blocks_fb_unpack(nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, nghost / threads_z);
    dim3 threads_fb_unpack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);
    hipLaunchKernelGGL(front_back_unpack, blocks_fb_unpack, threads_fb_unpack, 0, 0,
                       fieldE + n * offset, fieldEr_front + n * fb_size, fieldEr_back + n * fb_size);

// tsb
/*     threads_x = nghost;
     dim3 blocks3 ((nghost+2) / threads_x, (ny+2*2) / THREADS_PER_BLOCK_Y, (nz+2*2) / THREADS_PER_BLOCK_Z);
     dim3 threads3 (threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
     hipLaunchKernelGGL (unpack_lr, blocks3, threads3, 0, 0,\
                        fieldEr, fieldEr_left + n * lr_size, fieldEr_right + n * lr_size);

     threads_y = nghost;
     dim3 blocks4 ((nx) / THREADS_PER_BLOCK_X, (nghost+2) / threads_y, (nz+2*2) / THREADS_PER_BLOCK_Z);
     dim3 threads4 (THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);
     hipLaunchKernelGGL (unpack_tb, blocks4, threads4, 0, 0,\
                    fieldEr, fieldEr_top + n * tb_size, fieldEr_bottom + n * tb_size);

     threads_z = nghost;
     dim3 blocks5 ((nx) / THREADS_PER_BLOCK_X, (ny) / THREADS_PER_BLOCK_Y, (nghost+2) / threads_z);
     dim3 threads5 (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);
     hipLaunchKernelGGL (unpack_fb, blocks5, threads5, 0, 0,\
                    fieldEr, fieldEr_front + n * fb_size, fieldEr_back + n * fb_size);

     dim3 blocks6 ((nx) / THREADS_PER_BLOCK_X, (ny) / THREADS_PER_BLOCK_Y, (nz) / THREADS_PER_BLOCK_Z);
     dim3 threads6 (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
     hipLaunchKernelGGL (unpack_all, blocks6, threads6, 0, 0,\
                    fieldEr, fieldE + n * offset);
*/
#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    trans_unpack_time += timer;
#endif
  }
}

__global__ void
ac_calc_F1_dev(Dtype *fieldE, Dtype *f1, Dtype *fieldEu_left, Dtype *fieldEu_right,
               Dtype *fieldEu_top, Dtype *fieldEu_bottom, Dtype *fieldEu_front, Dtype *fieldEu_back,
               Dtype LE, Dtype KE, int n)
{
  int m;
  int i, j, k;
  int n_left, n_right, n_top, n_bottom, n_front, n_back;
  double f_left, f_right, f_top, f_bottom, f_front, f_back;
  double tmp1, tmp2;
  double u0, u1, u2, u3, un;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  u0 = fieldE[k * nx * ny + j * nx + i];
  u1 = fieldE[k * nx * ny + j * nx + i + offset * 1];
  u2 = fieldE[k * nx * ny + j * nx + i + offset * 2];
  u3 = fieldE[k * nx * ny + j * nx + i + offset * 3];

  if (n == 0)
    un = u0;
  if (n == 1)
    un = u1;
  if (n == 2)
    un = u2;
  if (n == 3)
    un = u3;
  f1[k * nx * ny + j * nx + i] = LE * KE * un;

  double f;
  double sumh22;
  sumh22 = u0 * u0 + u1 * u1 + u2 * u2 + u3 * u3;
  f = f_A * un - f_B * un * un + f_C * sumh22 * un;
  f1[k * nx * ny + j * nx + i] -= LE * f;
  if (left >= 0)
  {
    if (i == ix1)
    {
      f1[k * nx * ny + j * nx + i] += LE * epn2 * fieldEu_left[k * ny + j] / hx / hx;
    }
  }

  if (right >= 0)
  {
    if (i == ix4 - 1)
    {
      f1[k * nx * ny + j * nx + i] += LE * epn2 * fieldEu_right[k * ny + j] / hx / hx;
    }
  }

  if (top >= 0)
  {
    if (j == iy1)
    {
      f1[k * nx * ny + j * nx + i] += LE * epn2 * fieldEu_top[k * nx + i] / hy / hy;
    }
  }
  if (bottom >= 0)
  {
    if (j == iy4 - 1)
    {
      f1[k * nx * ny + j * nx + i] += LE * epn2 * fieldEu_bottom[k * nx + i] / hy / hy;
    }
  }

  if (front >= 0)
  {
    if (k == iz1)
    {
      f1[k * nx * ny + j * nx + i] += LE * epn2 * fieldEu_front[j * nx + i] / hz / hz;
    }
  }

  if (back >= 0)
  {
    if (k == iz4 - 1)
    {
      f1[k * nx * ny + j * nx + i] += LE * epn2 * fieldEu_back[j * nx + i] / hz / hz;
    }
  }
}

void rotation_matrix()
{
  int n, i, j, k;
  double m, p, q;

  if (simulate_checkpoint == 1)
  {
   //m = rand() / (double)(RAND_MAX);
   //p = rand() / (double)(RAND_MAX);
   //q = rand() / (double)(RAND_MAX);

    for (k = 0; k < ORI; k++)
    {
      m = rand() / (double)(RAND_MAX);
      p = rand() / (double)(RAND_MAX);
      q = rand() / (double)(RAND_MAX);
      theta[k * DIM + 0] = m * 180;
      theta[k * DIM + 1] = p * 180;
      theta[k * DIM + 2] = q * 180;
    }
    MPI_Bcast(&theta[0], ORI * DIM, MPI_DOUBLE, 0, XYZ_COMM);
    if (myrank == 0)
    {
      char filename[1024];
      FILE *fp;
      sprintf(filename, "%stheta.dat", data_dir);
      fp = fopen(filename, "w");
      for (k = 0; k < ORI; k++)
      {
        for (i = 0; i < 3; i++)
        {
          fprintf(fp, "%lf\n", theta[k * DIM + i]);
        }
      }
      fclose(fp);
    }
  }
#if 1
  for (n = 0; n < nac; n++)
  {
    for (i = 0; i < ORI; i++)
    {
      lambdar[0 * DIM + 0] = ac[n].lambda[0][0] * cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) + ac[n].lambda[1][0] * sin(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) - ac[n].lambda[2][0] * sin(theta[i * DIM + 1] * PI / 180);
      lambdar[0 * DIM + 1] = ac[n].lambda[0][1] * cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) + ac[n].lambda[1][1] * sin(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) - ac[n].lambda[2][1] * sin(theta[i * DIM + 1] * PI / 180);
      lambdar[0 * DIM + 2] = ac[n].lambda[0][2] * cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) + ac[n].lambda[1][2] * sin(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) - ac[n].lambda[2][2] * sin(theta[i * DIM + 1] * PI / 180);
      lambdar[1 * DIM + 0] = ac[n].lambda[0][0] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) - sin(theta[i * DIM + 2] * PI / 180 * cos(theta[i * DIM + 0] * PI / 180))) + ac[n].lambda[1][0] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) + cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180)) + ac[n].lambda[2][0] * cos(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180);
      lambdar[1 * DIM + 1] = ac[n].lambda[0][1] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) - sin(theta[i * DIM + 2] * PI / 180 * cos(theta[i * DIM + 0] * PI / 180))) + ac[n].lambda[1][1] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) + cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180)) + ac[n].lambda[2][1] * cos(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180);
      lambdar[1 * DIM + 2] = ac[n].lambda[0][2] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) - sin(theta[i * DIM + 2] * PI / 180 * cos(theta[i * DIM + 0] * PI / 180))) + ac[n].lambda[1][2] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) + cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180)) + ac[n].lambda[2][2] * cos(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180);
      lambdar[2 * DIM + 0] = ac[n].lambda[0][0] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) + sin(theta[i * DIM + 2] * PI / 180 * sin(theta[i * DIM + 0] * PI / 180))) + ac[n].lambda[1][0] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) - cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180)) + ac[n].lambda[2][0] * cos(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180);
      lambdar[2 * DIM + 1] = ac[n].lambda[0][1] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) + sin(theta[i * DIM + 2] * PI / 180 * sin(theta[i * DIM + 0] * PI / 180))) + ac[n].lambda[1][1] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) - cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180)) + ac[n].lambda[2][1] * cos(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180);
      lambdar[2 * DIM + 2] = ac[n].lambda[0][2] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) + sin(theta[i * DIM + 2] * PI / 180 * sin(theta[i * DIM + 0] * PI / 180))) + ac[n].lambda[1][2] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) - cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180)) + ac[n].lambda[2][2] * cos(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180);
      lambdar1[0 * DIM + 0 + i * DIM * DIM + n * ORI * DIM * DIM] = lambdar[0 * DIM + 0] * cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) + lambdar[0 * DIM + 1] * sin(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) - lambdar[0 * DIM + 2] * sin(theta[i * DIM + 1] * PI / 180);
      lambdar1[0 * DIM + 1 + i * DIM * DIM + n * ORI * DIM * DIM] = lambdar[0 * DIM + 0] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) - sin(theta[i * DIM + 2] * PI / 180 * cos(theta[i * DIM + 0] * PI / 180))) + lambdar[0 * DIM + 1] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) + cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180)) + lambdar[0 * DIM + 2] * cos(theta[1] * PI / 180) * sin(theta[0] * PI / 180);
      lambdar1[0 * DIM + 2 + i * DIM * DIM + n * ORI * DIM * DIM] = lambdar[0 * DIM + 0] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) + sin(theta[i * DIM + 2] * PI / 180 * sin(theta[i * DIM + 0] * PI / 180))) + lambdar[0 * DIM + 1] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) - cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180)) + lambdar[0 * DIM + 2] * cos(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180);
      lambdar1[1 * DIM + 0 + i * DIM * DIM + n * ORI * DIM * DIM] = lambdar[1 * DIM + 0] * cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) + lambdar[1 * DIM + 1] * sin(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) - lambdar[1 * DIM + 2] * sin(theta[i * DIM + 1] * PI / 180);
      lambdar1[1 * DIM + 1 + i * DIM * DIM + n * ORI * DIM * DIM] = lambdar[1 * DIM + 0] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) - sin(theta[i * DIM + 2] * PI / 180 * cos(theta[i * DIM + 0] * PI / 180))) + lambdar[1 * DIM + 1] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) + cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180)) + lambdar[1 * DIM + 2] * cos(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180);
      lambdar1[1 * DIM + 2 + i * DIM * DIM + n * ORI * DIM * DIM] = lambdar[1 * DIM + 0] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) + sin(theta[i * DIM + 2] * PI / 180 * sin(theta[i * DIM + 0] * PI / 180))) + lambdar[1 * DIM + 1] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) - cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180)) + lambdar[1 * DIM + 2] * cos(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180);
      lambdar1[2 * DIM + 0 + i * DIM * DIM + n * ORI * DIM * DIM] = lambdar[2 * DIM + 0] * cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) + lambdar[2 * DIM + 1] * sin(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 1] * PI / 180) - lambdar[2 * DIM + 2] * sin(theta[i * DIM + 1] * PI / 180);
      lambdar1[2 * DIM + 1 + i * DIM * DIM + n * ORI * DIM * DIM] = lambdar[2 * DIM + 0] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) - sin(theta[i * DIM + 2] * PI / 180 * cos(theta[i * DIM + 0] * PI / 180))) + lambdar[2 * DIM + 1] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180) + cos(theta[i * DIM + 2] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180)) + lambdar[2 * DIM + 2] * cos(theta[i * DIM + 1] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180);
      lambdar1[2 * DIM + 2 + i * DIM * DIM + n * ORI * DIM * DIM] = lambdar[2 * DIM + 0] * (cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) + sin(theta[i * DIM + 2] * PI / 180 * sin(theta[i * DIM + 0] * PI / 180))) + lambdar[2 * DIM + 1] * (sin(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180) - cos(theta[i * DIM + 2] * PI / 180) * sin(theta[i * DIM + 0] * PI / 180)) + lambdar[2 * DIM + 2] * cos(theta[i * DIM + 1] * PI / 180) * cos(theta[i * DIM + 0] * PI / 180);
    }
  }
#endif
}

/*
void rotation_matrix()
{
  int n, i, j;

  double m, p, q;
  if (simulate_checkpoint == 1)
  {
    m = rand() / (double)(RAND_MAX);
    p = rand() / (double)(RAND_MAX);
    q = rand() / (double)(RAND_MAX);

    theta[0] = m * 180;
    theta[1] = p * 180;
    theta[2] = q * 180;
    // theta[0] = 45.0;
    // theta[1] = 45.0;
    // theta[2] = 45.0;
    MPI_Bcast(&theta[0], 3, MPI_DOUBLE, 0, R_COMM);
  }
  //      printf("id = %d, %d, %d, %d, theta0 = %lf, theta1 = %lf, theta2 = %lf\n",myrank,cart_id[0],cart_id[1],cart_id[2],theta[0],theta[1],theta[2]);
  for (n = 0; n < nac; n++)
  {

    ac[n].lambdar[0][0] = ac[n].lambda[0][0] * cos(theta[2] * PI / 180) * cos(theta[1] * PI / 180) + ac[n].lambda[1][0] * sin(theta[2] * PI / 180) * cos(theta[1] * PI / 180) - ac[n].lambda[2][0] * sin(theta[1] * PI / 180);
    ac[n].lambdar[0][1] = ac[n].lambda[0][1] * cos(theta[2] * PI / 180) * cos(theta[1] * PI / 180) + ac[n].lambda[1][1] * sin(theta[2] * PI / 180) * cos(theta[1] * PI / 180) - ac[n].lambda[2][1] * sin(theta[1] * PI / 180);
    ac[n].lambdar[0][2] = ac[n].lambda[0][2] * cos(theta[2] * PI / 180) * cos(theta[1] * PI / 180) + ac[n].lambda[1][2] * sin(theta[2] * PI / 180) * cos(theta[1] * PI / 180) - ac[n].lambda[2][2] * sin(theta[1] * PI / 180);
    ac[n].lambdar[1][0] = ac[n].lambda[0][0] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) - sin(theta[2] * PI / 180 * cos(theta[0] * PI / 180))) + ac[n].lambda[1][0] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) + cos(theta[2] * PI / 180) * cos(theta[0] * PI / 180)) + ac[n].lambda[2][0] * cos(theta[1] * PI / 180) * sin(theta[0] * PI / 180);
    ac[n].lambdar[1][1] = ac[n].lambda[0][1] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) - sin(theta[2] * PI / 180 * cos(theta[0] * PI / 180))) + ac[n].lambda[1][1] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) + cos(theta[2] * PI / 180) * cos(theta[0] * PI / 180)) + ac[n].lambda[2][1] * cos(theta[1] * PI / 180) * sin(theta[0] * PI / 180);
    ac[n].lambdar[1][2] = ac[n].lambda[0][2] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) - sin(theta[2] * PI / 180 * cos(theta[0] * PI / 180))) + ac[n].lambda[1][2] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) + cos(theta[2] * PI / 180) * cos(theta[0] * PI / 180)) + ac[n].lambda[2][2] * cos(theta[1] * PI / 180) * sin(theta[0] * PI / 180);
    ac[n].lambdar[2][0] = ac[n].lambda[0][0] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) + sin(theta[2] * PI / 180 * sin(theta[0] * PI / 180))) + ac[n].lambda[1][0] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) - cos(theta[2] * PI / 180) * sin(theta[0] * PI / 180)) + ac[n].lambda[2][0] * cos(theta[1] * PI / 180) * cos(theta[0] * PI / 180);
    ac[n].lambdar[2][1] = ac[n].lambda[0][1] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) + sin(theta[2] * PI / 180 * sin(theta[0] * PI / 180))) + ac[n].lambda[1][1] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) - cos(theta[2] * PI / 180) * sin(theta[0] * PI / 180)) + ac[n].lambda[2][1] * cos(theta[1] * PI / 180) * cos(theta[0] * PI / 180);
    ac[n].lambdar[2][2] = ac[n].lambda[0][2] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) + sin(theta[2] * PI / 180 * sin(theta[0] * PI / 180))) + ac[n].lambda[1][2] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) - cos(theta[2] * PI / 180) * sin(theta[0] * PI / 180)) + ac[n].lambda[2][2] * cos(theta[1] * PI / 180) * cos(theta[0] * PI / 180);
    ac[n].lambdar1[0][0] = ac[n].lambdar[0][0] * cos(theta[2] * PI / 180) * cos(theta[1] * PI / 180) + ac[n].lambdar[0][1] * sin(theta[2] * PI / 180) * cos(theta[1] * PI / 180) - ac[n].lambdar[0][2] * sin(theta[1] * PI / 180);
    ac[n].lambdar1[0][1] = ac[n].lambdar[0][0] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) - sin(theta[2] * PI / 180 * cos(theta[0] * PI / 180))) + ac[n].lambdar[0][1] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) + cos(theta[2] * PI / 180) * cos(theta[0] * PI / 180)) + ac[n].lambdar[0][2] * cos(theta[1] * PI / 180) * sin(theta[0] * PI / 180);
    ac[n].lambdar1[0][2] = ac[n].lambdar[0][0] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) + sin(theta[2] * PI / 180 * sin(theta[0] * PI / 180))) + ac[n].lambdar[0][1] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) - cos(theta[2] * PI / 180) * sin(theta[0] * PI / 180)) + ac[n].lambdar[0][2] * cos(theta[1] * PI / 180) * cos(theta[0] * PI / 180);
    ac[n].lambdar1[1][0] = ac[n].lambdar[1][0] * cos(theta[2] * PI / 180) * cos(theta[1] * PI / 180) + ac[n].lambdar[1][1] * sin(theta[2] * PI / 180) * cos(theta[1] * PI / 180) - ac[n].lambdar[1][2] * sin(theta[1] * PI / 180);
    ac[n].lambdar1[1][1] = ac[n].lambdar[1][0] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) - sin(theta[2] * PI / 180 * cos(theta[0] * PI / 180))) + ac[n].lambdar[1][1] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) + cos(theta[2] * PI / 180) * cos(theta[0] * PI / 180)) + ac[n].lambdar[1][2] * cos(theta[1] * PI / 180) * sin(theta[0] * PI / 180);
    ac[n].lambdar1[1][2] = ac[n].lambdar[1][0] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) + sin(theta[2] * PI / 180 * sin(theta[0] * PI / 180))) + ac[n].lambdar[1][1] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) - cos(theta[2] * PI / 180) * sin(theta[0] * PI / 180)) + ac[n].lambdar[1][2] * cos(theta[1] * PI / 180) * cos(theta[0] * PI / 180);
    ac[n].lambdar1[2][0] = ac[n].lambdar[2][0] * cos(theta[2] * PI / 180) * cos(theta[1] * PI / 180) + ac[n].lambdar[2][1] * sin(theta[2] * PI / 180) * cos(theta[1] * PI / 180) - ac[n].lambdar[2][2] * sin(theta[1] * PI / 180);
    ac[n].lambdar1[2][1] = ac[n].lambdar[2][0] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) - sin(theta[2] * PI / 180 * cos(theta[0] * PI / 180))) + ac[n].lambdar[2][1] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * sin(theta[0] * PI / 180) + cos(theta[2] * PI / 180) * cos(theta[0] * PI / 180)) + ac[n].lambdar[2][2] * cos(theta[1] * PI / 180) * sin(theta[0] * PI / 180);
    ac[n].lambdar1[2][2] = ac[n].lambdar[2][0] * (cos(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) + sin(theta[2] * PI / 180 * sin(theta[0] * PI / 180))) + ac[n].lambdar[2][1] * (sin(theta[2] * PI / 180) * sin(theta[1] * PI / 180) * cos(theta[0] * PI / 180) - cos(theta[2] * PI / 180) * sin(theta[0] * PI / 180)) + ac[n].lambdar[2][2] * cos(theta[1] * PI / 180) * cos(theta[0] * PI / 180);
  }
}
*/
// divergence((D-lambda)gradient(eta))
void anisotropic_input(void)
{
  int i, j, n, m;
  double D[3][3];
  char filename[1024];
  FILE *fp;
  D[0][0] = kkx;
  D[0][1] = 0.0;
  D[0][2] = 0.0;
  D[1][0] = 0.0;
  D[1][1] = kky;
  D[1][2] = 0.0;
  D[2][0] = 0.0;
  D[2][1] = 0.0;
  D[2][2] = kkz;

  sprintf(filename, "anisotropic_input.txt");
  fp = fopen(filename, "r");
  for (n = 0; n < nac; n++)
  {
      for (m = 0; m < 1; m++)
      {
          for (i = 0; i < 3; i++)
          {
            for (j = 0; j < 3; j++)
            {
              fscanf(fp, "%lf", &(lambdar1[i * DIM + j + m * DIM * DIM + n * DIM * DIM]));
            }
          }
      }
  }
  fclose(fp);
#if 0
  if (rotation == 1)
  {
    rotation_matrix();
    if (simulate_checkpoint == 1 && myrank == 0)
    {
      char filename[1024];
      FILE *fp;
      for (n = 0; n < nac; n++)
      {
        sprintf(filename, "%sorient_%d.dat", data_dir, n);
        fp = fopen(filename, "w");
        for (m = 0; m < ORI; m++)
        {
          for (i = 0; i < 3; i++)
          {
            for (j = 0; j < 3; j++)
            {
              fprintf(fp, "%lf\n", lambdar1[i * DIM + j + m * DIM * DIM + n * ORI * DIM * DIM]);
            }
          }
        }
        fclose(fp);
      }
    }
  }
#endif

    for (n = 0; n < nac; n++)
    {
      for (m = 0; m < 1; m++)
      {
        for (i = 0; i < 3; i++)
        {
          for (j = 0; j < 3; j++)
          {
            ac[n].lambda_check[i][j] = lambdar1[i * DIM + j + n * DIM * DIM];
            lambdar1[i * DIM + j + m * DIM * DIM + n * 1 * DIM * DIM] = D[i][j] - lambdar1[i * DIM + j + m * DIM * DIM + n * 1 * DIM * DIM];
          }
        }
      }
    }
#if 0
  else
  {
    for (n = 0; n < nac; n++)
    {
      for (i = 0; i < 3; i++)
      {
        for (j = 0; j < 3; j++)
        {
          //ac[n].lambda_check[i][j] = ac[n].lambda[i][j];
        }
      }
    }
    for (n = 0; n < nac; n++)
    {
      for (i = 0; i < 3; i++)
      {
        for (j = 0; j < 3; j++)
        {
          ac[n].lambda[i][j] = D[i][j] - ac[n].lambda[i][j];
        }
      }
    }
  }
#endif
  hipMemcpy(lambda, lambdar1, nac * 1 * DIM * DIM * sizeof(Dtype), hipMemcpyHostToDevice);
}

__global__ void
ac_add_F1_F2_dev(Dtype *f1, Dtype *f2, Dtype *Elas, Dtype *fieldE, Dtype epn2, Dtype LE, int ANISOTROPIC, int n)
{
  int i, j, k, l;
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  // fieldE1[k * nx * ny + j * nx + i] = f1[k * nx * ny + j * nx + i];
  if (ANISOTROPIC == 1)
  {
    f1[k * nx * ny + j * nx + i] -= epn2 * LE * f2[k * nx * ny + j * nx + i];
  }
  f1[k * nx * ny + j * nx + i] -= LE * ElasticScale * Elas[k * ny * nx + j * nx + i];
}

__global__ void
ac_updateU_new(int n, Dtype *field, Dtype *field1, Dtype *DDX, Dtype *DDY, Dtype *DDZ, Dtype LE, Dtype KE, Dtype epn2, Dtype *phiE)
{
  double tmp, Hijk;
  int i, j, k, l, s;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  l = k * ny * nx + j * nx + i;
  s = (k + 2) * (ny + 2 * 2) * (nx + 2 * 2) + (j + 2) * (nx + 2 * 2) + i + 2;
  tmp = kkz * DDZ[k] + kky * DDY[j] + kkx * DDX[i];
  Hijk = -LE * (tmp * epn2 - KE);
  if (fabs(Hijk) < 1.0e-8)
  {
    Hijk = 0.0;
  }
  if (fabs(Hijk) > 1.0e-8)
  {
    tmp = exp(-dt * Hijk);
    phiE[s] = (1.0 - tmp) / Hijk;
  }
  else
  {
    phiE[s] = dt;
  }
  tmp = 1.0 - phiE[s] * Hijk;
  field[l] = tmp * field[l] + phiE[s] * field1[l];
}

__global__ void
elastic_copyin(Dtype *f, Dtype *fe)
{
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f[k * NX * NY + j * NX + i] = fe[(k + nghost) * ny * nx + (j + nghost) * nx + i + nghost];
}

__global__ void
elastic_multiply_BN(Dtype *Bnre, Dtype *Bnim, Dtype *BN, Dtype *ftre, Dtype *ftim, int q)
{
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  if (q == 0)
  {
    Bnre[k * NY * NX + j * NX + i] = BN[k * NY * NX + j * NX + i] * ftre[k * NY * NX + j * NX + i];
    Bnim[k * NY * NX + j * NX + i] = BN[k * NY * NX + j * NX + i] * ftim[k * NY * NX + j * NX + i];
  }
  else
  {
    Bnre[k * NY * NX + j * NX + i] += BN[k * NY * NX + j * NX + i] * ftre[k * NY * NX + j * NX + i];
    Bnim[k * NY * NX + j * NX + i] += BN[k * NY * NX + j * NX + i] * ftim[k * NY * NX + j * NX + i];
  }
}
/*
__global__ void
initial_elas(double *elas)
{
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  elas[k * nx * ny + j * nx + i] = 0.0;
}
*/
__global__ void
elastic_copyout(Dtype *f, Dtype *fe)
{
  int i, j, k;
  double np0 = (double)(NX * procs[0]);
  double np1 = (double)(NY * procs[1]);
  double np2 = (double)(NZ * procs[2]);
  double c = 1.0 / np0 / np1 / np2;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f[(k + nghost) * ny * nx + (j + nghost) * nx + i + nghost] = fe[k * NX * NY + j * NX + i];
}
/*
void elastic_forward_calculate(void)
{
  dim3 blocks ((ix4 - ix1) / THREADS_PER_BLOCK_X, (iy4 - iy1) / THREADS_PER_BLOCK_Y, (iz4 - iz1) / THREADS_PER_BLOCK_Z);
  dim3 threads (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  dim3 blocks_ela (NX / THREADS_PER_BLOCK_X, NY / THREADS_PER_BLOCK_Y, NZ / THREADS_PER_BLOCK_Z);
  dim3 threads_ela (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

  for(int p = 0; p < NELASTIC; p++)
  {
    hipLaunchKernelGGL (elastic_copyin, blocks_ela, threads_ela, 0, 0, tmpy_RE1, fieldE + p * offset);
    hipMemcpy (tmpy_re, tmpy_RE1, NZ * NY * NX * sizeof (double), hipMemcpyDeviceToHost);
    fft_forward(tmpy_re, tmpy_fftre[p], tmpy_fftim[p]);
    hipMemcpy (tmpy_fftRE[p], tmpy_fftre[p], NZ * NY * NX * sizeof (double), hipMemcpyHostToDevice);
    hipMemcpy (tmpy_fftIM[p], tmpy_fftim[p], NZ * NY * NX * sizeof (double), hipMemcpyHostToDevice);
  }
}

void elastic_backward_calculate(int p)
{
  dim3 blocks ((ix4 - ix1) / THREADS_PER_BLOCK_X, (iy4 - iy1) / THREADS_PER_BLOCK_Y, (iz4 - iz1) / THREADS_PER_BLOCK_Z);
  dim3 threads (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  dim3 blocks_ela (NX / THREADS_PER_BLOCK_X, NY / THREADS_PER_BLOCK_Y, NZ / THREADS_PER_BLOCK_Z);
  dim3 threads_ela (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

    for(int q = 0; q < NELASTIC; q++)
    {
//     hipLaunchKernelGGL(elastic_calculate_tmp0, blocks_ela, threads_ela, 0, 0,
//                        BN, Epsilon2d + p * DIM * DIM, C4D, Epsilon2d + q * DIM * DIM);
//     hipLaunchKernelGGL(elastic_multiply_BN_0, blocks_ela, threads_ela, 0, 0,
//                        BN, n11, n22, n33, C4D, Sigma2d + p * DIM * DIM, Sigma2d + q * DIM * DIM, p, q);
      hipLaunchKernelGGL (elastic_multiply_BN, blocks_ela, threads_ela, 0, 0, BN_fftRE[p], BN_fftIM[p], BN[p][q], tmpy_fftRE[q], tmpy_fftIM[q], q);
    }
    hipMemcpy (tmpy_fftre[p], BN_fftRE[p], NX * NY * NZ * sizeof (double), hipMemcpyDeviceToHost);
    hipMemcpy (tmpy_fftim[p], BN_fftIM[p], NX * NY * NZ * sizeof (double), hipMemcpyDeviceToHost);
    //write_Bn_im(tmpy_fftre[p], tmpy_fftim[p], p);
    fft_backward(tmpy_fftre[p], tmpy_fftim[p], tmpy_re);
    //fft_backward(Bn[p][0], tmpy_fftre[p], tmpy_re);
    //write_Bn_im(tmpy_re, tmpy_re, p);
    hipMemcpy (tmpy_RE1, tmpy_re, NX * NY * NZ * sizeof (double), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(initial_elas, blocks, threads, 0, 0, Elas);
    hipLaunchKernelGGL (elastic_copyout, blocks_ela, threads_ela, 0, 0, Elas, tmpy_RE1);
    //hipMemcpy (elas, Elas, sizeof (Dtype) * offset, hipMemcpyDeviceToHost);
}
*/
void elastic_transfer(int n)
{
  int threads_x, threads_y, threads_z;
  //  for (int n = 0; n < nac; n++)
  //  {
  hipMemcpy(s_front, &Elas[nx * ny * nghost + n * offset], sizeof(Dtype) * fb_size, hipMemcpyDeviceToHost);
  hipMemcpy(s_back, &Elas[nx * ny * (nz - nghost - (nghost + 2)) + n * offset], sizeof(Dtype) * fb_size, hipMemcpyDeviceToHost);

  MPI_Startall(4, ireq_front_back);
  MPI_Waitall(4, ireq_front_back, status);

  hipMemcpy(R_front, r_front, sizeof(Dtype) * fb_size, hipMemcpyHostToDevice);
  hipMemcpy(R_back, r_back, sizeof(Dtype) * fb_size, hipMemcpyHostToDevice);
  dim3 blocks_tb_pack(nx / THREADS_PER_BLOCK_X, (nghost + 2) / THREADS_PER_BLOCK_Y, (nz + (nghost + 2) * 2) / THREADS_PER_BLOCK_Z);
  dim3 threads_tb_pack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(top_bottom_pack, blocks_tb_pack, threads_tb_pack, 0, 0,
                     Elas + n * offset, S_top, S_bottom, R_front, R_back);

  hipMemcpy(s_top, S_top, sizeof(Dtype) * tb_size, hipMemcpyDeviceToHost);
  hipMemcpy(s_bottom, S_bottom, sizeof(Dtype) * tb_size, hipMemcpyDeviceToHost);
  MPI_Startall(4, ireq_top_bottom);
  MPI_Waitall(4, ireq_top_bottom, status);

  hipMemcpy(R_top, r_top, sizeof(Dtype) * tb_size, hipMemcpyHostToDevice);
  hipMemcpy(R_bottom, r_bottom, sizeof(Dtype) * tb_size, hipMemcpyHostToDevice);

  dim3 blocks_lr_pack((nghost + 2) / THREADS_PER_BLOCK_X, (ny + (nghost + 2) * 2) / THREADS_PER_BLOCK_Y, (nz + (nghost + 2) * 2) / THREADS_PER_BLOCK_Z);
  dim3 threads_lr_pack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(left_right_pack, blocks_lr_pack, threads_lr_pack, 0, 0,
                     Elas + n * offset, S_left, S_right, R_top, R_bottom, R_front, R_back);

  hipMemcpy(s_left, S_left, sizeof(Dtype) * lr_size, hipMemcpyDeviceToHost);
  hipMemcpy(s_right, S_right, sizeof(Dtype) * lr_size, hipMemcpyDeviceToHost);

  MPI_Startall(4, ireq_left_right);
  MPI_Waitall(4, ireq_left_right, status);

  hipMemcpy(R_left, r_left, sizeof(Dtype) * lr_size, hipMemcpyHostToDevice);
  hipMemcpy(R_right, r_right, sizeof(Dtype) * lr_size, hipMemcpyHostToDevice);

  threads_x = nghost;
  dim3 blocks_lr_unpack(nghost / threads_x, ny / THREADS_PER_BLOCK_Y, nz / THREADS_PER_BLOCK_Z);
  dim3 threads_lr_unpack(threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(left_right_unpack, blocks_lr_unpack, threads_lr_unpack, 0, 0,
                     Elas + n * offset, R_left, R_right);
  threads_y = nghost;
  dim3 blocks_tb_unpack(nx / THREADS_PER_BLOCK_X, nghost / threads_y, nz / THREADS_PER_BLOCK_Z);
  dim3 threads_tb_unpack(THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(top_bottom_unpack, blocks_tb_unpack, threads_tb_unpack, 0, 0,
                     Elas + n * offset, R_top, R_bottom);
  threads_z = nghost;
  dim3 blocks_fb_unpack(nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, nghost / threads_z);
  dim3 threads_fb_unpack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);
  hipLaunchKernelGGL(front_back_unpack, blocks_fb_unpack, threads_fb_unpack, 0, 0,
                     Elas + n * offset, R_front, R_back);
  //  }
}


extern void hip_init()
{
  deviceId = myrank % 4;
  // hipGetDevice(&deviceId);
  hipGetDeviceProperties(&props, deviceId);
}
__global__ void
Dtype2B_E(Dtype *E, Stype *F2B)
{
  int i, j, k, l;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  l = k * nx * ny + j * nx + i;
  F2B[l] = (Stype)((E[l] + 0.1) * 30000);
}

__global__ void
Dtype2B_E_2(Dtype *E, Stype *F2B)
{
  int i, j, k, l;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  l = k * ((nx - nghost * 2) / 2) * ((ny - nghost * 2) / 2) + j * ((nx - nghost * 2) / 2) + i;
  F2B[l] = (Stype)((E[l] + 0.1) * 30000);
}

__global__ void
skip_copyin_new(Dtype *f, Dtype *fe)
{
  int i, j, k;
  int dim_x = nx / 2;
  int dim_y = ny / 2;
  int stride = 2;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f[k * dim_y * dim_x + j * dim_x + i] = (1.0 / 8.0) *
                                         (fe[(k * stride) * ny * nx + (j * stride) * nx + i * stride] +
                                          fe[(k * stride) * ny * nx + (j * stride) * nx + i * stride + 1] +
                                          fe[(k * stride) * ny * nx + (j * stride + 1) * nx + i * stride] +
                                          fe[(k * stride) * ny * nx + (j * stride + 1) * nx + i * stride + 1] +
                                          fe[(k * stride + 1) * ny * nx + (j * stride) * nx + i * stride] +
                                          fe[(k * stride + 1) * ny * nx + (j * stride) * nx + i * stride + 1] +
                                          fe[(k * stride + 1) * ny * nx + (j * stride + 1) * nx + i * stride] +
                                          fe[(k * stride + 1) * ny * nx + (j * stride + 1) * nx + i * stride + 1]);
}

__global__ void
skip_copyin_new_2(Dtype *f, Dtype *fe)
{
  int i, j, k;
  int dim_x = (nx - nghost * 2) / 2;
  int dim_y = (ny - nghost * 2) / 2;
  int stride = 2;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f[k * dim_y * dim_x + j * dim_x + i] = (1.0 / 8.0) *
                                         (fe[(k * stride) * (ny - nghost * 2) * (nx - nghost * 2) + (j * stride) * (nx - nghost * 2) + i * stride] +
                                          fe[(k * stride) * (ny - nghost * 2) * (nx - nghost * 2) + (j * stride) * (nx - nghost * 2) + i * stride + 1] +
                                          fe[(k * stride) * (ny - nghost * 2) * (nx - nghost * 2) + (j * stride + 1) * (nx - nghost * 2) + i * stride] +
                                          fe[(k * stride) * (ny - nghost * 2) * (nx - nghost * 2) + (j * stride + 1) * (nx - nghost * 2) + i * stride + 1] +
                                          fe[(k * stride + 1) * (ny - nghost * 2) * (nx - nghost * 2) + (j * stride) * (nx - nghost * 2) + i * stride] +
                                          fe[(k * stride + 1) * (ny - nghost * 2) * (nx - nghost * 2) + (j * stride) * (nx - nghost * 2) + i * stride + 1] +
                                          fe[(k * stride + 1) * (ny - nghost * 2) * (nx - nghost * 2) + (j * stride + 1) * (nx - nghost * 2) + i * stride] +
                                          fe[(k * stride + 1) * (ny - nghost * 2) * (nx - nghost * 2) + (j * stride + 1) * (nx - nghost * 2) + i * stride + 1]);
}

__global__ void
calc2contr(Dtype *f, Dtype *fe)
{
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f[k * (ny - 2 * nghost) * (nx - 2 * nghost) + j * (nx - 2 * nghost) + i] = fe[(k + nghost) * ny * nx + (j + nghost) * nx + (i + nghost)];
}

// 固定将f下采样一倍至fe
__global__ void
skip_copyin_new_3(Dtype *f, Dtype *fe, int input_dim_x, int input_dim_y)
{
  int i, j, k;
  int dim_x = input_dim_x / 2;
  int dim_y = input_dim_y / 2;
  int stride = 2;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f[k * dim_y * dim_x + j * dim_x + i] = (1.0 / 8.0) *
                                         (fe[(k * stride) * input_dim_y * input_dim_x + (j * stride) * input_dim_x + i * stride] +
                                          fe[(k * stride) * input_dim_y * input_dim_x + (j * stride) * input_dim_x + i * stride + 1] +
                                          fe[(k * stride) * input_dim_y * input_dim_x + (j * stride + 1) * input_dim_x + i * stride] +
                                          fe[(k * stride) * input_dim_y * input_dim_x + (j * stride + 1) * input_dim_x + i * stride + 1] +
                                          fe[(k * stride + 1) * input_dim_y * input_dim_x + (j * stride) * input_dim_x + i * stride] +
                                          fe[(k * stride + 1) * input_dim_y * input_dim_x + (j * stride) * input_dim_x + i * stride + 1] +
                                          fe[(k * stride + 1) * input_dim_y * input_dim_x + (j * stride + 1) * input_dim_x + i * stride] +
                                          fe[(k * stride + 1) * input_dim_y * input_dim_x + (j * stride + 1) * input_dim_x + i * stride + 1]);
}

dim3 create_blocks(int size[3])
{
  dim3 blocks(size[0] / THREADS_PER_BLOCK_X, size[1] / THREADS_PER_BLOCK_Y, size[2] / THREADS_PER_BLOCK_Z);
  return blocks;
}

dim3 create_threads(int size[3])
{
  dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  return threads;
}

__global__ void
elastic_calculate_sigma2d(Dtype *Sigma2d1, Dtype *C4D1, Dtype *Epsilon2d1)
{
  int i, j, k, l, m;
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  if (i < DIM && j < DIM && k == 0)
  {
    for (m = 0; m < DIM; m++)
    {
      for (l = 0; l < DIM; l++)
      {
        Sigma2d1[i * DIM + j] += C4D1[i * DIM * DIM * DIM + j * DIM * DIM + m * DIM + l] * Epsilon2d1[m * DIM + l];
      }
    }
  }
}

__global__ void
elastic_calculate_n(Dtype *n11, Dtype *n22, Dtype *n33)
{
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  int gx, gy, gz;
  int gnnx = procs[0] * NX;
  int cntx = gnnx / 2;
  int gnny = procs[1] * NY;
  int cnty = gnny / 2;
  int gnnz = procs[2] * NZ;
  int cntz = gnnz / 2;
  if (i < NX && j == 0 && k == 0)
  {
    gx = cart_id[0] * NX + i;
    if (gx < cntx)
    {
      n11[i] = 1.0 * gx;
    }
    else
    {
      n11[i] = -1.0 * (gnnx - gx);
    }
  }
  if (i == 0 && j < NY && k == 0)
  {
    gy = cart_id[1] * NY + j;
    if (gy < cnty)
    {
      n22[j] = 1.0 * gy;
    }
    else
    {
      n22[j] = -1.0 * (gnny - gy);
    }
  }
  if (i == 0 && j == 0 && k < NZ)
  {
    gz = cart_id[2] * NZ + k;
    if (gz < cntz)
    {
      n33[k] = 1.0 * gz;
    }
    else
    {
      n33[k] = -1.0 * (gnnz - gz);
    }
  }
}

__global__ void
elastic_calculate_tmp0(Dtype *Bn, Dtype *epsilon2d, Dtype *C4D, Dtype *epsilon2d2)
{
  int i, j, k, l, m, n, s;
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  double tmp0 = 0.0;
  for (n = 0; n < DIM; n++)
  {
    for (s = 0; s < DIM; s++)
    {
      for (m = 0; m < DIM; m++)
      {
        for (l = 0; l < DIM; l++)
        {
           tmp0 += C4D[n * DIM * DIM * DIM + s * DIM * DIM + m * DIM + l] * epsilon2d[n * DIM + s] * epsilon2d2[m * DIM + l];
        }
      }
    }
  }
  Bn[k * NY * NX + j * NX + i] = tmp0;
}

__global__ void
elastic_multiply_BN_0 (Dtype *Bn, Dtype *n11, Dtype *n22, Dtype *n33, Dtype *C4D, Dtype *Sigma2d1, Dtype *Sigma2d2, int p, int q)
{
        int i, j, k;
        i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
        int gx = cart_id[0] * NX + i;
        int gy = cart_id[1] * NY + j;
        int gz = cart_id[2] * NZ + k;
        double n123[3], omega_inver[9] = {0.0}, omega[9] = {0.0};
        //double tmpin, tmpinv;
        double tmp1 = 0.0;
        if (gx == 0 && gy == 0 && gz == 0)
        {
                n123[0] = 0;
                n123[1] = 1;
                n123[2] = 0;
        }
        else
        {
                //double norm = 1.0 / sqrt(n11[i]*n11[i]+n22[j]*n22[j]+n33[k]*n33[k]);
                double norm = 1.0 / pow((pow(n11[i], 2) + pow(n22[j], 2) + pow(n33[k], 2)), 0.5);
                n123[0] = n11[i] * norm;
                n123[1] = n22[j] * norm;
                n123[2] = n33[k] * norm;
        }
        for (int n = 0; n < DIM; ++n)
        {
                for (int s = 0; s < DIM; ++s)
                {
                        for (int m = 0; m < DIM; ++m)
                        {
                                for (int l = 0; l < DIM; ++l)
                                {
                                        omega_inver[n * DIM + s] += C4D[n * DIM * DIM * DIM + m * DIM * DIM + s * DIM + l] * n123[m] * n123[l];
                                }
                        }
                }
        }
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
        for (int n = 0; n < DIM; ++n)
        {
                for (int s = 0; s < DIM; ++s)
                {
                        for (int m=0; m < DIM; ++m)
                        {
                                for (int l = 0; l < DIM; ++l)
                                {
                                        tmp1 += n123[n] * Sigma2d1[n * DIM + s] * omega[s * DIM + m] * Sigma2d2[m * DIM + l] * n123[l];
                                }
                        }
                }
        }
        Bn[k * NY * NX + j * NX + i] = Bn[k * NY * NX + j * NX + i] - tmp1;
}

__global__ void
FFT_IFFT_Bn_exp(double *Gx_re, double *Gx_im, double *Gy_re, double *Gy_im, double *Gz_re, double *Gz_im)
{
  int i, j, k;
  int dim = (nx - 2 * nghost) / 1;
  int num = Approx;
  double x = dim + Approx - 1;
  //double x = 287;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

  if (i < dim + Approx - 1){
      Gx_re[i * num + j] = cos(2.0 * PI * i * j / x);
      Gx_im[i * num + j] = sin(2.0 * PI * i * j / x);
      Gy_re[i * num + j] = cos(2.0 * PI * i * j / x);
      Gy_im[i * num + j] = sin(2.0 * PI * i * j / x);
      Gz_re[i * num + j] = cos(2.0 * PI * i * j / x);
      Gz_im[i * num + j] = sin(2.0 * PI * i * j / x);
  }
}

__global__ void
conv_exp_matrix(double *Gx_re, double *Gx_im, double *Gy_re, double *Gy_im, double *Gz_re, double *Gz_im)
{
        int i, j, k;
        int dim = (nx - 2 * nghost) / 1;
        int num = dim + Approx - 1;

        double dc0 = (double)dim * cart_id[0];
        double dc1 = (double)dim * cart_id[1];
        double dc2 = (double)dim * cart_id[2];
        double dp0 = (double)dim * procs[0];
        double dp1 = (double)dim * procs[1];
        double dp2 = (double)dim * procs[2];
        //double x = (double)(319*319*319);
        double x = (double)(num);

        i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

        //i: [255-30~255][0~255] j: [255-30~255][0~255] k: [255-30~255][0~255]  
        if (i < dim + Approx - 1 && j < dim + Approx - 1) {
                Gx_re[j * num + i] = cos(2.0 * PI * i * j / x);
                Gx_im[j * num + i] = sin(2.0 * PI * i * j / x);
                Gy_re[j * num + i] = cos(2.0 * PI * i * j / x);
                Gy_im[j * num + i] = sin(2.0 * PI * i * j / x);
                Gz_re[j * num + i] = cos(2.0 * PI * i * j / x);
                Gz_im[j * num + i] = sin(2.0 * PI * i * j / x);
        }
}

__global__ void
initial_elas(double *elas)
{
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  elas[k * nx * ny + j * nx + i] = 0.0;
}

__global__ void
down_sample_eta(Dtype *f, Dtype *fe)
{
  int i, j, k;
  int dim = (nx - 2 * nghost) / 1;
  int stride = 1;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f[k * dim * dim + j * dim + i] = fe[(k * stride + nghost) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost];
}

__global__ void
conv_top_bottom_pack(double *field, double *fields_top, double *fields_bottom, double *fieldr_front, double *fieldr_back)
{
  int k_s, k_e;
  int f_j, f_k;
  int tb_j, tb_k;
  int t_ofst, b_ofst, i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  k_s = Approx;
  k_e = (nz - 2 * nghost) + Approx;
  f_j = nx - 2 * nghost;
  f_k = (nx - 2 * nghost) * (ny - 2 * nghost);
  tb_j = nx - 2 * nghost;
  tb_k = (nx - 2 * nghost) * Approx;
  t_ofst = nx - 2 * nghost;
  b_ofst = (nx - 2 * nghost) * ((ny - 2 * nghost) - Approx);

  if (k < k_s)
  {
    if (front >= 0)
    {
      fields_top[tb_k * k + tb_j * j + i] = fieldr_front[f_k * k + f_j * j + i];
      fields_bottom[tb_k * k + tb_j * j + i] = fieldr_front[f_k * k + f_j * j + i + b_ofst];
    }
  }
  else if (k >= k_e)
  {
    if (back >= 0)
    {
      fields_top[tb_k * k + tb_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * j + i];
      fields_bottom[tb_k * k + tb_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * j + i + b_ofst];
    }
  }
  else
  {
    fields_top[tb_k * k + tb_j * j + i] = field[f_k * (k - k_s) + f_j * j + i];
    fields_bottom[tb_k * k + tb_j * j + i] = field[f_k * (k - k_s) + f_j * j + i + b_ofst];
  }
}

__global__ void
conv_left_right_pack(double *field, double *fields_left, double *fields_right, double *fieldr_top, double *fieldr_bottom, double *fieldr_front, double *fieldr_back)
{
  int j_s, j_e, k_s, k_e;
  int f_j, f_k;
  int lr_j, lr_k;
  int tb_j, tb_k;
  int l_ofst, r_ofst, i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  j_s = Approx;
  j_e = (ny - 2 * nghost) + Approx;
  k_s = Approx;
  k_e = (nz - 2 * nghost) + Approx;
  f_j = nx - 2 * nghost;
  f_k = (nx - 2 * nghost) * (ny - 2 * nghost);
  lr_j = Approx;
  lr_k = Approx * ((ny - 2 * nghost) + Approx * 2);
  tb_j = nx - 2 * nghost;
  tb_k = (nx - 2 * nghost) * Approx;
  l_ofst = nghost;
  r_ofst = (nx - 2 * nghost) - Approx;

  if (j < j_s)
  {
    if (top >= 0)
    {
      fields_left[lr_k * k + lr_j * j + i] = fieldr_top[tb_k * k + tb_j * j + i];
      fields_right[lr_k * k + lr_j * j + i] = fieldr_top[tb_k * k + tb_j * j + i + r_ofst];
    }
  }
  else if (j >= j_e)
  {
    if (bottom >= 0)
    {
      fields_left[lr_k * k + lr_j * j + i] = fieldr_bottom[tb_k * k + tb_j * (j - j_e) + i];
      fields_right[lr_k * k + lr_j * j + i] = fieldr_bottom[tb_k * k + tb_j * (j - j_e) + i + r_ofst];
    }
  }
  else
  {
    if (k < k_s)
    {
      if (front >= 0)
      {
        fields_left[lr_k * k + lr_j * j + i] = fieldr_front[f_k * k + f_j * (j - j_s) + i];
        fields_right[lr_k * k + lr_j * j + i] = fieldr_front[f_k * k + f_j * (j - j_s) + i + r_ofst];
      }
    }
    else if (k >= k_e)
    {
      if (back >= 0)
      {
        fields_left[lr_k * k + lr_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * (j - j_s) + i];
        fields_right[lr_k * k + lr_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * (j - j_s) + i + r_ofst];
      }
    }
    else
    {
      fields_left[lr_k * k + lr_j * j + i] = field[f_k * (k - k_s) + f_j * (j - j_s) + i];
      fields_right[lr_k * k + lr_j * j + i] = field[f_k * (k - k_s) + f_j * (j - j_s) + i + r_ofst];
    }
  }
}

void conv_transfer(int n, double *f)
{
  int threads_x, threads_y, threads_z;
  hipMemcpy(conv_s_front, &f[0], sizeof(Dtype) * conv_fb_size, hipMemcpyDeviceToHost);
  hipMemcpy(conv_s_back, &f[NX * NY * (NZ - Approx)], sizeof(Dtype) * conv_fb_size, hipMemcpyDeviceToHost);

  MPI_Startall(4, conv_ireq_front_back);
  MPI_Waitall(4, conv_ireq_front_back, status);

  hipMemcpy(conv_R_front, conv_r_front, sizeof(Dtype) * conv_fb_size, hipMemcpyHostToDevice);
  hipMemcpy(conv_R_back, conv_r_back, sizeof(Dtype) * conv_fb_size, hipMemcpyHostToDevice);

  dim3 blocks_conv_tb_pack(NX / THREADS_PER_BLOCK_X, Approx / THREADS_PER_BLOCK_Y, (NZ + Approx * 2) / THREADS_PER_BLOCK_Z);
  dim3 threads_conv_tb_pack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(conv_top_bottom_pack, blocks_conv_tb_pack, threads_conv_tb_pack, 0, 0,
                     f, conv_S_top, conv_S_bottom, conv_R_front, conv_R_back);

  hipMemcpy(conv_s_top, conv_S_top, sizeof(Dtype) * conv_tb_size, hipMemcpyDeviceToHost);
  hipMemcpy(conv_s_bottom, conv_S_bottom, sizeof(Dtype) * conv_tb_size, hipMemcpyDeviceToHost);

  MPI_Startall(4, conv_ireq_top_bottom);
  MPI_Waitall(4, conv_ireq_top_bottom, status);

  hipMemcpy(conv_R_top, conv_r_top, sizeof(Dtype) * conv_tb_size, hipMemcpyHostToDevice);
  hipMemcpy(conv_R_bottom, conv_r_bottom, sizeof(Dtype) * conv_tb_size, hipMemcpyHostToDevice);

  dim3 blocks_conv_lr_pack(Approx / THREADS_PER_BLOCK_X, ((ny - 2 * nghost) + Approx * 2) / THREADS_PER_BLOCK_Y, ((nz - 2 * nghost) + Approx * 2) / THREADS_PER_BLOCK_Z);
  dim3 threads_conv_lr_pack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(conv_left_right_pack, blocks_conv_lr_pack, threads_conv_lr_pack, 0, 0,
                     f, conv_S_left, conv_S_right, conv_R_top, conv_R_bottom, conv_R_front, conv_R_back);
  hipMemcpy(conv_s_left, conv_S_left, sizeof(Dtype) * conv_lr_size, hipMemcpyDeviceToHost);
  hipMemcpy(conv_s_right, conv_S_right, sizeof(Dtype) * conv_lr_size, hipMemcpyDeviceToHost);

  MPI_Startall(4, conv_ireq_left_right);
  MPI_Waitall(4, conv_ireq_left_right, status);

  hipMemcpy(conv_R_left, conv_r_left, sizeof(Dtype) * conv_lr_size, hipMemcpyHostToDevice);
  hipMemcpy(conv_R_right, conv_r_right, sizeof(Dtype) * conv_lr_size, hipMemcpyHostToDevice);
}

__global__ void
conv_left_right_unpack(double *fieldr, double *fieldr_left, double *fieldr_right, int flag)
{
  int f_j, f_k;
  int lr_k, lr_j;
  int i, j, k;

  f_j = (nx - 2 * nghost) + Approx - 1;
  f_k = ((nx - 2 * nghost) + Approx - 1) * ((ny - 2 * nghost) + Approx - 1);
  lr_j = Approx;
  lr_k = Approx * ((ny - 2 * nghost) + Approx * 2);

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  if (flag == 0) {
    if (i > 0 && j > 0 && k > 0) {
      if (left >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i - 1] = fieldr_left[lr_k * k + lr_j * j + i];
      }
    }
  }
  if (flag == 1) {
    if (i > 0 && j > 0 && k > 0) {
      if (left >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i - 1] = fieldr_left[lr_k * (k + Approx) + lr_j * j + i];
      }
    }
  }
  if (flag == 2) {
    if (i > 0 && j > 0 && k > 0) {
      if (left >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i - 1] = fieldr_left[lr_k * k + lr_j * (j + Approx) + i];
      }
    }
  }
  if (flag == 3) {
    if (i > 0 && j > 0 && k > 0) {
      if (left >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i - 1] = fieldr_left[lr_k * (k + Approx) + lr_j * (j + Approx) + i];
      }
    }
  }
  if (flag == 4) {
    if (j > 0 && k > 0) {
      if (right >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i + NX - 1] = fieldr_right[lr_k * k + lr_j * j + i];
      }
    }
  }
  if (flag == 5) {
    if (j > 0 && k > 0) {
      if (right >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i + NX - 1] = fieldr_right[lr_k * (k + Approx) + lr_j * j + i];
      }
    }
  }
  if (flag == 6) {
    if (j > 0 && k > 0) {
      if (right >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i + NX - 1] = fieldr_right[lr_k * k + lr_j * (j + Approx) + i];
      }
    }
  }
  if (flag == 7) {
    if (j > 0 && k > 0) {
      if (right >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i + NX - 1] = fieldr_right[lr_k * (k + Approx) + lr_j * (j + Approx) + i];
      }
    }
  }
}
__global__ void
conv_top_bottom_unpack(double *fieldr, double *fieldr_top, double *fieldr_bottom, int flag)
{
  int f_j, f_k;
  int tb_k, tb_j;
  int i, j, k;
  f_j = (nx - 2 * nghost) + Approx - 1;
  f_k = ((nx - 2 * nghost) + Approx - 1) * ((nx - 2 * nghost) + Approx - 1);
  tb_j = nx - 2 * nghost;
  tb_k = (nx - 2 * nghost) * Approx;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  if (flag == 0) {
    if (j > 0 && k > 0) {
      if (top >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i + Approx - 1] = fieldr_top[tb_k * k + tb_j * j + i];
        //fieldr[f_k * (k - 1) + f_j * (j - 1 + NY) + i] = fieldr_top[tb_k * k + tb_j * j + i];
      }
    }
  }
  if (flag == 1) {
    if (j > 0 && k > 0) {
      if (top >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i + Approx - 1] = fieldr_top[tb_k * (k + Approx) + tb_j * j + i];
      }
    }
  }
  if (flag == 2) {
    if (k > 0) {
      if (bottom >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j + NY - 1) + i + Approx - 1] = fieldr_bottom[tb_k * k + tb_j * j + i];
      }
    }
  }
  if (flag == 3) {
    if (k > 0) {
      if (bottom >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j + NY - 1) + i + Approx - 1] = fieldr_bottom[tb_k * (k + Approx) + tb_j * j + i];
      }
    }
  }
  if (flag == 4) {
    if (i > 0 && j > 0 && k > 0) {
      if (top >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i - 1] = fieldr_top[tb_k * k + tb_j * j + i];
      }
    }
  }
  if (flag == 5) {
    if (i > 0 && j > 0 && k > 0) {
      if (top >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i - 1] = fieldr_top[tb_k * (k + Approx) + tb_j * j + i];
      }
    }
  }
  if (flag == 6) {
    if (i > 0 && k > 0) {
      if (bottom >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j + NY - 1) + i - 1] = fieldr_bottom[tb_k * k + tb_j * j + i];
      }
    }
  }
  if (flag == 7) {
    if (i > 0 && k > 0) {
      if (bottom >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j + NY - 1) + i - 1] = fieldr_bottom[tb_k * (k + Approx) + tb_j * j + i];
      }
    }
  }
}
__global__ void
conv_front_back_unpack(double *fieldr, double *fieldr_front, double *fieldr_back, int flag)
{
  int f_j, f_k;
  int fb_j, fb_k;
  int i, j, k;

  f_j = (nx - 2 * nghost) + Approx - 1;
  f_k = ((nx - 2 * nghost) + Approx - 1) * ((nx - 2 * nghost) + Approx - 1);
  fb_j = nx - 2 * nghost;
  fb_k = (nx - 2 * nghost) * (ny - 2 * nghost);
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  if (flag == 0) {
    if (k > 0) {
      if (front >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j + Approx - 1) + i + Approx - 1] = fieldr_front[fb_k * k + fb_j * j + i];
        //fieldr[f_k * (k - 1 + NZ) + f_j * j + i] = fieldr_front[fb_k * k + fb_j * j + i];
      }
    }
  }
  if (flag == 1) {
    if (back >= 0)
    {
      fieldr[f_k * (k + NZ - 1) + f_j * (j + Approx - 1) + i + Approx - 1] = fieldr_back[fb_k * k + fb_j * j + i];
    }
  }
  if (flag == 2) {
    if (j > 0 && k > 0) {
      if (front >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + i + Approx - 1] = fieldr_front[fb_k * k + fb_j * j + i];
      }
    }
  }
  if (flag == 3) {
    if (j > 0) {
      if (back >= 0)
      {
        fieldr[f_k * (k + NZ - 1) + f_j * (j - 1) + i + Approx - 1] = fieldr_back[fb_k * k + fb_j * j + i];
      }
    }
  }
  if (flag == 4) {
    if (i > 0 && k > 0) {
      if (front >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j + Approx - 1) + (i - 1)] = fieldr_front[fb_k * k + fb_j * j + i];
      }
    }
  }
  if (flag == 5) {
    if (i > 0) {
      if (back >= 0)
      {
        fieldr[f_k * (k + NZ - 1) + f_j * (j + Approx - 1) + i - 1] = fieldr_back[fb_k * k + fb_j * j + i];
      }
    }
  }
  if (flag == 6) {
    if (i > 0 && j > 0 && k > 0) {
      if (front >= 0)
      {
        fieldr[f_k * (k - 1) + f_j * (j - 1) + (i - 1)] = fieldr_front[fb_k * k + fb_j * j + i];
      }
    }
  }
  if (flag == 7) {
    if (i > 0 && j > 0) {
      if (back >= 0)
      {
        fieldr[f_k * (k + NZ - 1) + f_j * (j - 1) + (i - 1)] = fieldr_back[fb_k * k + fb_j * j + i];
      }
    }
  }
}
__global__ void
conv_inner_unpack(double *fieldr, double *field, int flag)
{
  int f_j, f_k;
  int i, j, k;
  f_j = (nx - 2 * nghost) + Approx - 1;
  f_k = ((nx - 2 * nghost) + Approx - 1) * ((ny - 2 * nghost) + Approx - 1);
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  if (flag == 0) {
    fieldr[f_k * (k + Approx - 1) + f_j * (j + Approx - 1) + (i + Approx - 1)] = field[(nx - 2 * nghost) * (ny - 2 * nghost) * k + (nx - 2 * nghost) * j + i];
    //fieldr[f_k * k + f_j * j + i] = field[(nx - 2 * nghost) * (ny - 2 * nghost) * k + (nx - 2 * nghost) * j + i];
  }
  if (flag == 1) {
    if (k > 0) {
      fieldr[f_k * (k - 1) + f_j * (j + Approx - 1) + (i + Approx - 1)] = field[(nx - 2 * nghost) * (ny - 2 * nghost) * k + (nx - 2 * nghost) * j + i];
    }
  }
  if (flag == 2) {
    if (j > 0) {
      fieldr[f_k * (k + Approx - 1) + f_j * (j - 1) + (i + Approx - 1)] = field[(nx - 2 * nghost) * (ny - 2 * nghost) * k + (nx - 2 * nghost) * j + i];
    }
  }
  if (flag == 3) {
    if (j > 0 && k > 0) {
      fieldr[f_k * (k - 1) + f_j * (j - 1) + (i + Approx - 1)] = field[(nx - 2 * nghost) * (ny - 2 * nghost) * k + (nx - 2 * nghost) * j + i];
    }
  }
  if (flag == 4) {
    if (i > 0) {
      fieldr[f_k * (k + Approx - 1) + f_j * (j + Approx - 1) + (i - 1)] = field[(nx - 2 * nghost) * (ny - 2 * nghost) * k + (nx - 2 * nghost) * j + i];
    }
  }
  if (flag == 5) {
    if (i > 0 && k > 0) {
      fieldr[f_k * (k - 1) + f_j * (j + Approx - 1) + (i - 1)] = field[(nx - 2 * nghost) * (ny - 2 * nghost) * k + (nx - 2 * nghost) * j + i];
    }
  }
  if (flag == 6) {
    if (i > 0 && j > 0) {
      fieldr[f_k * (k + Approx - 1) + f_j * (j - 1) + (i - 1)] = field[(nx - 2 * nghost) * (ny - 2 * nghost) * k + (nx - 2 * nghost) * j + i];
    }
  }
  if (flag == 7) {
    if (i > 0 && j > 0 && k > 0) {
      fieldr[f_k * (k - 1) + f_j * (j - 1) + (i - 1)] = field[(nx - 2 * nghost) * (ny - 2 * nghost) * k + (nx - 2 * nghost) * j + i];
    }
  }
}

void conv_unpack(int flag)
{
  int threads_x, threads_y, threads_z;
  threads_x = Approx;
  dim3 blocks_conv_lr_unpack(Approx / threads_x, ((ny - 2 * nghost) + Approx) / THREADS_PER_BLOCK_Y, ((nz - 2 * nghost) + Approx) / THREADS_PER_BLOCK_Z);
  dim3 threads_conv_lr_unpack(threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(conv_left_right_unpack, blocks_conv_lr_unpack, threads_conv_lr_unpack, 0, 0,
                     conv_e, conv_R_left, conv_R_right, flag);
  threads_y = Approx;
  dim3 blocks_conv_tb_unpack((nx - 2 * nghost) / THREADS_PER_BLOCK_X, Approx / threads_y, ((nz - 2 * nghost) + Approx) / THREADS_PER_BLOCK_Z);
  dim3 threads_conv_tb_unpack(THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(conv_top_bottom_unpack, blocks_conv_tb_unpack, threads_conv_tb_unpack, 0, 0,
                     conv_e, conv_R_top, conv_R_bottom, flag);
  threads_z = Approx;
  dim3 blocks_conv_fb_unpack((nx - 2 * nghost) / THREADS_PER_BLOCK_X, (ny - 2 * nghost) / THREADS_PER_BLOCK_Y, Approx / threads_z);
  dim3 threads_conv_fb_unpack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);
  hipLaunchKernelGGL(conv_front_back_unpack, blocks_conv_fb_unpack, threads_conv_fb_unpack, 0, 0,
                     conv_e, conv_R_front, conv_R_back, flag);
  dim3 blocks_conv_inner_unpack((nx - 2 * nghost) / THREADS_PER_BLOCK_X, (ny - 2 * nghost) / THREADS_PER_BLOCK_Y, (nz - 2 * nghost) / THREADS_PER_BLOCK_Z);
  dim3 threads_conv_inner_unpack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(conv_inner_unpack, blocks_conv_inner_unpack, threads_conv_inner_unpack, 0, 0,
                     conv_e, tmpy_RE1, flag);
}

__global__ void
initial_fftre_im(double *re_out)
{
  int i, j, k;
  int num = Approx * 2;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  re_out[k * num * num + j * num + i] = 0.0;
}

__global__ void
IGx_IGy_IGz_matrix(double *Gx_re, double *Gx_im, double *Gy_re, double *Gy_im, double *Gz_re, double *Gz_im, int flag)
{
  int i, j, k;
  int dim = (nx - 2 * nghost) / 1;
  int num = Approx;

  double dc0 = (double)dim * cart_id[0];
  double dc1 = (double)dim * cart_id[1];
  double dc2 = (double)dim * cart_id[2];
  double dp0 = (double)dim * procs[0];
  double dp1 = (double)dim * procs[1];
  double dp2 = (double)dim * procs[2];

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

  if (flag == 0)
  {
    Gx_re[j * dim + i] = cos(2.0 * PI * (dc0 + i) * j / dp0);
    Gx_im[j * dim + i] = sin(2.0 * PI * (dc0 + i) * j / dp0);
    Gy_re[j * dim + i] = cos(2.0 * PI * (dc1 + i) * j / dp1);
    Gy_im[j * dim + i] = sin(2.0 * PI * (dc1 + i) * j / dp1);
    Gz_re[j * dim + i] = cos(2.0 * PI * (dc2 + i) * j / dp2);
    Gz_im[j * dim + i] = sin(2.0 * PI * (dc2 + i) * j / dp2);
  }
  else if (flag == 4)
  {
    Gx_re[j * dim + i] = cos(2.0 * PI * (dc0 + i) * (dp0 + j - num) / dp0);
    Gx_im[j * dim + i] = sin(2.0 * PI * (dc0 + i) * (dp0 + j - num) / dp0);
    Gy_re[j * dim + i] = cos(2.0 * PI * (dc1 + i) * j / dp1);
    Gy_im[j * dim + i] = sin(2.0 * PI * (dc1 + i) * j / dp1);
    Gz_re[j * dim + i] = cos(2.0 * PI * (dc2 + i) * j / dp2);
    Gz_im[j * dim + i] = sin(2.0 * PI * (dc2 + i) * j / dp2);
  }
  else if (flag == 1)
  {
    Gx_re[j * dim + i] = cos(2.0 * PI * (dc0 + i) * j / dp0);
    Gx_im[j * dim + i] = sin(2.0 * PI * (dc0 + i) * j / dp0);
    Gy_re[j * dim + i] = cos(2.0 * PI * (dc1 + i) * j / dp1);
    Gy_im[j * dim + i] = sin(2.0 * PI * (dc1 + i) * j / dp1);
    Gz_re[j * dim + i] = cos(2.0 * PI * (dc2 + i) * (dp2 + j - num) / dp2);
    Gz_im[j * dim + i] = sin(2.0 * PI * (dc2 + i) * (dp2 + j - num) / dp2);
  }
  else if (flag == 5)
  {
    Gx_re[j * dim + i] = cos(2.0 * PI * (dc0 + i) * (dp0 + j - num) / dp0);
    Gx_im[j * dim + i] = sin(2.0 * PI * (dc0 + i) * (dp0 + j - num) / dp0);
    Gy_re[j * dim + i] = cos(2.0 * PI * (dc1 + i) * j / dp1);
    Gy_im[j * dim + i] = sin(2.0 * PI * (dc1 + i) * j / dp1);
    Gz_re[j * dim + i] = cos(2.0 * PI * (dc2 + i) * (dp2 + j - num) / dp2);
    Gz_im[j * dim + i] = sin(2.0 * PI * (dc2 + i) * (dp2 + j - num) / dp2);
  }
  else if (flag == 2)
  {
    Gx_re[j * dim + i] = cos(2.0 * PI * (dc0 + i) * j / dp0);
    Gx_im[j * dim + i] = sin(2.0 * PI * (dc0 + i) * j / dp0);
    Gy_re[j * dim + i] = cos(2.0 * PI * (dc1 + i) * (dp1 + j - num) / dp1);
    Gy_im[j * dim + i] = sin(2.0 * PI * (dc1 + i) * (dp1 + j - num) / dp1);
    Gz_re[j * dim + i] = cos(2.0 * PI * (dc2 + i) * j / dp2);
    Gz_im[j * dim + i] = sin(2.0 * PI * (dc2 + i) * j / dp2);
  }
  else if (flag == 6)
  {
    Gx_re[j * dim + i] = cos(2.0 * PI * (dc0 + i) * (dp0 + j - num) / dp0);
    Gx_im[j * dim + i] = sin(2.0 * PI * (dc0 + i) * (dp0 + j - num) / dp0);
    Gy_re[j * dim + i] = cos(2.0 * PI * (dc1 + i) * (dp1 + j - num) / dp1);
    Gy_im[j * dim + i] = sin(2.0 * PI * (dc1 + i) * (dp1 + j - num) / dp1);
    Gz_re[j * dim + i] = cos(2.0 * PI * (dc2 + i) * j / dp2);
    Gz_im[j * dim + i] = sin(2.0 * PI * (dc2 + i) * j / dp2);
  }
  else if (flag == 3)
  {
    Gx_re[j * dim + i] = cos(2.0 * PI * (dc0 + i) * j / dp0);
    Gx_im[j * dim + i] = sin(2.0 * PI * (dc0 + i) * j / dp0);
    Gy_re[j * dim + i] = cos(2.0 * PI * (dc1 + i) * (dp1 + j - num) / dp1);
    Gy_im[j * dim + i] = sin(2.0 * PI * (dc1 + i) * (dp1 + j - num) / dp1);
    Gz_re[j * dim + i] = cos(2.0 * PI * (dc2 + i) * (dp2 + j - num) / dp2);
    Gz_im[j * dim + i] = sin(2.0 * PI * (dc2 + i) * (dp2 + j - num) / dp2);
  }
  else if (flag == 7)
  {
    Gx_re[j * dim + i] = cos(2.0 * PI * (dc0 + i) * (dp0 + j - num) / dp0);
    Gx_im[j * dim + i] = sin(2.0 * PI * (dc0 + i) * (dp0 + j - num) / dp0);
    Gy_re[j * dim + i] = cos(2.0 * PI * (dc1 + i) * (dp1 + j - num) / dp1);
    Gy_im[j * dim + i] = sin(2.0 * PI * (dc1 + i) * (dp1 + j - num) / dp1);
    Gz_re[j * dim + i] = cos(2.0 * PI * (dc2 + i) * (dp2 + j - num) / dp2);
    Gz_im[j * dim + i] = sin(2.0 * PI * (dc2 + i) * (dp2 + j - num) / dp2);
  }
}

__global__ void
fft_ifftBN_multiply_fft_eta(Dtype *Out_re, Dtype *Out_im, Dtype *Bn_fftre, Dtype *Bn_fftim, Dtype *fft_re, Dtype *fft_im, int flag, int q)
{
        int i, j, k;
        double BN_re, BN_im, fft_eta_re, fft_eta_im, out_re, out_im;
        i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

        if (i < NX + Approx - 1 && j < NY + Approx - 1 && k < NZ + Approx - 1) {
                fft_eta_re = fft_re[k * (NX + Approx - 1) * (NY + Approx - 1) + j * (NX + Approx - 1) + i];
                fft_eta_im = fft_im[k * (NX + Approx - 1) * (NY + Approx - 1) + j * (NX + Approx - 1) + i];
                BN_re = Bn_fftre[k * (NX + Approx - 1) * (NY + Approx - 1) + j * (NX + Approx - 1) + i];
                BN_im = Bn_fftim[k * (NX + Approx - 1) * (NY + Approx - 1) + j * (NX + Approx - 1) + i];

                out_re = BN_re * fft_eta_re - BN_im * fft_eta_im;
                out_im = BN_re * fft_eta_im + BN_im * fft_eta_re;
                if (flag == 0 && q == 0) {
                        Out_re[k * (NX + Approx - 1) * (NY + Approx - 1) + j * (NX + Approx - 1) + i] = out_re;
                        Out_im[k * (NX + Approx - 1) * (NY + Approx - 1) + j * (NX + Approx - 1) + i] = out_im;
                }
                else {
                        Out_re[k * (NX + Approx - 1) * (NY + Approx - 1) + j * (NX + Approx - 1) + i] += out_re;
                        Out_im[k * (NX + Approx - 1) * (NY + Approx - 1) + j * (NX + Approx - 1) + i] += out_im;
                }
        }
}

__global__ void
ifft_multiply_scale(Dtype *ifftre)
{
  int i, j, k;
  double np0 = (double)(NX * procs[0]);
  double np1 = (double)(NY * procs[1]);
  double np2 = (double)(NZ * procs[2]);
  double c = 1.0 / np0 / np1 / np2;
  int num = Approx;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  ifftre[k * Approx * Approx + j * Approx + i] = c * ifftre[k * Approx * Approx + j * Approx + i];
}

__global__ void
elas_cal(Dtype *elas_small, Dtype *elas_big, int flag, double *elas)
{
        int i, j, k;
        double a = nx - 2 * nghost + Approx - 1;
        double c = 1.0 / a / a / a;
        int num = Approx;

        i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

	elas_small[(k + nghost) * nx * ny + (j + nghost) * nx + (i + nghost)] = c * elas_big[(k + num - 1) * (NX + num - 1) * (NY + num - 1) + (j + num - 1) * (NX + num - 1) + i + num - 1];
	elas[k * NX * NY + j * NX + i] = c * elas_big[(k + num - 1) * (NX + num - 1) * (NY + num - 1) + (j + num - 1) * (NX + num - 1) + i + num - 1];
}

void elastic_calculate()
{
        dim3 blocks((ix4 - ix1) / THREADS_PER_BLOCK_X, (iy4 - iy1) / THREADS_PER_BLOCK_Y, (iz4 - iz1) / THREADS_PER_BLOCK_Z);
        dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
        dim3 blocks_ela2(NX / THREADS_PER_BLOCK_X, NY / THREADS_PER_BLOCK_Y, NZ / THREADS_PER_BLOCK_Z);
        dim3 threads_ela2(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
        dim3 blocks_ela(NX / THREADS_PER_BLOCK_X, Approx / THREADS_PER_BLOCK_Y, Approx / THREADS_PER_BLOCK_Z);
        dim3 threads_ela(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
        dim3 blocks_ela3(2 * Approx / THREADS_PER_BLOCK_X, 2 * Approx / THREADS_PER_BLOCK_Y, 2 * Approx / THREADS_PER_BLOCK_Z);
        dim3 threads_ela3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
        dim3 blocks_scale(Approx / THREADS_PER_BLOCK_X, Approx / THREADS_PER_BLOCK_Y, Approx / THREADS_PER_BLOCK_Z);
        dim3 threads_scale(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
        dim3 blocks_exp_conv2((NX + Approx) / THREADS_PER_BLOCK_X, (NX + Approx) / THREADS_PER_BLOCK_Y, (NX + Approx) / THREADS_PER_BLOCK_Z);
        dim3 threads_exp_conv2(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

#ifdef SCLETD_DEBUG
  hipEventRecord(st, NULL);
  hipEventSynchronize(st);
#endif
        for (int q = 0; q < NELASTIC; q++)
        {
                hipLaunchKernelGGL(initial_elas, blocks, threads, 0, 0, Elas + q * offset);
                hipLaunchKernelGGL(down_sample_eta, blocks_ela2, threads_ela2, 0, 0, tmpy_RE1, fieldE + q * offset);
                //hipMemcpy(tmpy_re, tmpy_RE1, sizeof(Dtype) * NX * NY * NZ, hipMemcpyDeviceToHost);
		//write_Bn_im (tmpy_re, tmpy_re, q);
                conv_transfer(q, tmpy_RE1);
                for (flag = 0; flag < 8; flag++)
                {
                        conv_unpack(flag);
                        // x -- dim
                        //(eta)^T*(Gx_re) <--> ((287*287)*287)*(287*287)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_x_n, conv_big_x_m, conv_big_x_k, &alpha, conv_e, conv_big_x_k, conv_big_x_re, conv_big_x_k, &beta, conv_temp9, conv_big_x_n);
                        //(eta)^T*(Gx_im) <--> ((319*319)*319)*(319*nx)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_x_n, conv_big_x_m, conv_big_x_k, &alpha, conv_e, conv_big_x_k, conv_big_x_im, conv_big_x_k, &beta, conv_temp10, conv_big_x_n);

                        // y -- dim
                        //(conv_temp1)^T*(Gy_re) <--> ((nx*319)*319)*(319*ny)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_y_n, conv_big_y_m, conv_big_y_k, &alpha, conv_temp9, conv_big_y_k, conv_big_y_re, conv_big_y_k, &beta, conv_temp11, conv_big_y_n);
                        //(conv_temp2)^T*(Gy_im) <--> ((nx*319)*319)*(319*ny)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_y_n, conv_big_y_m, conv_big_y_k, &alpha1, conv_temp10, conv_big_y_k, conv_big_y_im, conv_big_y_k, &beta, conv_temp12, conv_big_y_n);
                        // caculate real part
                        rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_big_y_n, conv_big_y_m, &alpha, conv_temp11, conv_big_y_n, &beta2, conv_temp12, conv_big_y_n, conv_e, conv_big_y_n);
                        //(temp1)^T*(Gy_im) <--> ((nx*319)*319)*(319*ny)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_y_n, conv_big_y_m, conv_big_y_k, &alpha, conv_temp9, conv_big_y_k, conv_big_y_im, conv_big_y_k, &beta, conv_temp11, conv_big_y_n);
                        //(temp2)^T*(Gy_re) <--> ((nx*319)*319)*(319*ny)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_y_n, conv_big_y_m, conv_big_y_k, &alpha, conv_temp10, conv_big_y_k, conv_big_y_re, conv_big_y_k, &beta, conv_temp12, conv_big_y_n);
                        // caculate imag part
                        rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_big_y_n, conv_big_y_m, &alpha, conv_temp11, conv_big_y_n, &beta2, conv_temp12, conv_big_y_n, conv_temp9, conv_big_y_n);
                        // z -- dim
                        //(temp5)^T*(Gz_re) <--> ((nx*ny)*319)*(319*nz)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_z_n, conv_big_z_m, conv_big_z_k, &alpha, conv_e, conv_big_z_k, conv_big_z_re, conv_big_z_k, &beta, conv_temp10, conv_big_z_n);
                        //(temp6)^T*(Gz_im) <--> ((nx*ny)*319)*(319*nz)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_z_n, conv_big_z_m, conv_big_z_k, &alpha1, conv_temp9, conv_big_z_k, conv_big_z_im, conv_big_z_k, &beta, conv_temp11, conv_big_z_n);
                        // caculate real part
                        rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_big_z_n, conv_big_z_m, &alpha, conv_temp10, conv_big_z_n, &beta2, conv_temp11, conv_big_z_n, eta_RE + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), conv_big_z_n);
                        //(temp5)^T*(Gz_im) <--> ((nx*ny)*319)*(319*nz)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_z_n, conv_big_z_m, conv_big_z_k, &alpha, conv_e, conv_big_z_k, conv_big_z_im, conv_big_z_k, &beta, conv_temp10, conv_big_z_n);
                        //(temp6)^T*(Gz_re) <--> ((nx*ny)*319)*(319*nz)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_z_n, conv_big_z_m, conv_big_z_k, &alpha, conv_temp9, conv_big_z_k, conv_big_z_re, conv_big_z_k, &beta, conv_temp11, conv_big_z_n);
                        // caculate imag part
                        rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_big_z_n, conv_big_z_m, &alpha, conv_temp10, conv_big_z_n, &beta2, conv_temp11, conv_big_z_n, eta_IM + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), conv_big_z_n);
                }
                for (int p = 0; p < NELASTIC; p++)
                {
                        hipLaunchKernelGGL(initial_fftre_im, blocks_ela3, threads_ela3, 0, 0, tmpy_fftRE[p][q]);
                        if (iter == 0)
                        {
                                hipLaunchKernelGGL(elastic_calculate_tmp0, blocks_ela2, threads_ela2, 0, 0,
                                                   BN, Epsilon2d[p], C4D, Epsilon2d[q]);
                                hipLaunchKernelGGL(elastic_multiply_BN_0, blocks_ela2, threads_ela2, 0, 0,
                                                   BN, n11, n22, n33, C4D, Sigma2d[p], Sigma2d[q], p, q);
                                for (flag = 0; flag < 8; flag++)
                                {
                                        hipLaunchKernelGGL(IGx_IGy_IGz_matrix, blocks_ela, threads_ela, 0, 0, Gx_re, Gx_im, Gy_re, Gy_im, Gz_re, Gz_im, flag);
                                        // x -- dim
                                        //(BN)^T*(Gx_re)^T <--> ((ny*nz)*nx)*(nx*32)
                                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, Gx_n, Gx_k, Gx_m, &alpha, BN, Gx_m, Gx_re, Gx_m, &beta, temp1, Gx_n);
                                        //(BN)^T*(Gx_im)^T <--> ((ny*nz)*nx)*(nx*32)
                                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, Gx_n, Gx_k, Gx_m, &alpha1, BN, Gx_m, Gx_im, Gx_m, &beta, temp2, Gx_n);
                                        // y -- dim
                                        //(temp1)^T*(Gy_re)^T <--> ((32*nz)*ny)*(ny*32)
                                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, Gy_n, Gy_k, Gy_m, &alpha, temp1, Gy_m, Gy_re, Gy_m, &beta, temp3, Gy_n);
                                        //(temp2)^T*(Gy_im)^T <--> ((32*nz)*ny)*(ny*32)
                                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, Gy_n, Gy_k, Gy_m, &alpha, temp2, Gy_m, Gy_im, Gy_m, &beta, temp4, Gy_n);
                                        // caculate real part
                                        rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, Gy_n, Gy_k, &alpha, temp3, Gy_n, &beta2, temp4, Gy_n, temp5, Gy_n);

                                        //(temp1)^T*(Gy_im)^T <--> ((32*nz)*ny)*(ny*32)
                                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, Gy_n, Gy_k, Gy_m, &alpha1, temp1, Gy_m, Gy_im, Gy_m, &beta, temp3, Gy_n);
                                        //(temp2)^T*(Gy_re)^T <--> ((32*nz)*ny)*(ny*32)
                                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, Gy_n, Gy_k, Gy_m, &alpha, temp2, Gy_m, Gy_re, Gy_m, &beta, temp4, Gy_n);
                                        // caculate imag part
                                        rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, Gy_n, Gy_k, &alpha, temp3, Gy_n, &beta2, temp4, Gy_n, temp6, Gy_n);

                                        // z -- dim
                                        //(temp7)^T*(Gz_re)^T <--> ((32*32)*nz)*(nz*32)
                                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, Gz_n, Gz_k, Gz_m, &alpha, temp5, Gz_m, Gz_re, Gz_m, &beta, temp7, Gz_n);
                                        //(temp8)^T*(Gz_im)^T <--> ((32*32)*nz)*(nz*32)
                                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, Gz_n, Gz_k, Gz_m, &alpha, temp6, Gz_m, Gz_im, Gz_m, &beta, temp8, Gz_n);
                                        // caculate real part
                                        rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, Gz_n, Gz_k, &alpha, temp7, Gz_n, &beta2, temp8, Gz_n, tmpy_fftRE[p][q] + flag * ela_size, Gz_n);
                                        hipLaunchKernelGGL(ifft_multiply_scale, blocks_scale, threads_scale, 0, 0, tmpy_fftRE[p][q] + flag * ela_size);
                                }
                                hipMemcpy(tmpy_fftre[p][q], tmpy_fftRE[p][q], sizeof(Dtype) * 8 * ela_size, hipMemcpyDeviceToHost);
                                //kernel matrix
                                MPI_Allreduce(&tmpy_fftre[p][q][0], &fftre[p][q][0], 8 * ela_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                                hipMemcpy(fftRE[p][q], fftre[p][q], sizeof(Dtype) * 8 * ela_size, hipMemcpyHostToDevice);
				//write_Bn(fftre[p][q], p, q);
                        }
                        for (flag = 0; flag < 8; flag++)
                        {
                                // x -- dim
                                //(ifftBN)^T*(conv_x_re) <--> ((32*32)*32)*(32*287)
                                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_x_n, conv_x_m, conv_x_k, &alpha, fftRE[p][q] + flag * ela_size, conv_x_k, conv_x_re, conv_x_k, &beta, conv_temp5, conv_x_n);
                                //(ifftBN)^T*(conv_x_im) <--> ((32*32)*32)*(32*287)
                                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_x_n, conv_x_m, conv_x_k, &alpha, fftRE[p][q] + flag * ela_size, conv_x_k, conv_x_im, conv_x_k, &beta, conv_temp6, conv_x_n);

                                // y -- dim
                                //(conv_temp5)^T*(conv_y_re) <--> ((32*287)*32)*(32*287)
                                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_y_n, conv_y_m, conv_y_k, &alpha, conv_temp5, conv_y_k, conv_y_re, conv_y_k, &beta, conv_temp1, conv_y_n);
                                //(conv_temp6)^T*(conv_y_im) <--> ((32*287)*32)*(32*287)
                                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_y_n, conv_y_m, conv_y_k, &alpha1, conv_temp6, conv_y_k, conv_y_im, conv_y_k, &beta, conv_temp2, conv_y_n);
                                // caculate real part
                                rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_y_n, conv_y_m, &alpha, conv_temp1, conv_y_n, &beta2, conv_temp2, conv_y_n, conv_temp3, conv_y_n);
                                //(conv_temp5)^T*(conv_y_im) <--> ((32*287)*32)*(32*287)
                                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_y_n, conv_y_m, conv_y_k, &alpha, conv_temp5, conv_y_k, conv_y_im, conv_y_k, &beta, conv_temp1, conv_y_n);
                                //(conv_temp6)^T*(conv_y_re) <--> ((32*287)*32)*(32*287)
                                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_y_n, conv_y_m, conv_y_k, &alpha, conv_temp6, conv_y_k, conv_y_re, conv_y_k, &beta, conv_temp2, conv_y_n);
                                // caculate imag part
                                rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_y_n, conv_y_m, &alpha, conv_temp1, conv_y_n, &beta2, conv_temp2, conv_y_n, conv_temp4, conv_y_n);
                                // z -- dim
                                //(conv_temp3)^T*(conv_z_re) <--> ((287*287)*32)*(32*287)
                                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_z_n, conv_z_m, conv_z_k, &alpha, conv_temp3, conv_z_k, conv_z_re, conv_z_k, &beta, conv_temp9, conv_z_n);
                                //(conv_temp4)^T*(conv_z_im) <--> ((287*287)*32)*(32*287)
                                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_z_n, conv_z_m, conv_z_k, &alpha1, conv_temp4, conv_z_k, conv_z_im, conv_z_k, &beta, conv_temp10, conv_z_n);
                                // caculate real part
                                rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_z_n, conv_z_m, &alpha, conv_temp9, conv_z_n, &beta2, conv_temp10, conv_z_n, BN_RE + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), conv_z_n);
                                //(conv_temp3)^T*(conv_z_im) <--> ((287*287)*32)*(32*287)
                                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_z_n, conv_z_m, conv_z_k, &alpha, conv_temp3, conv_z_k, conv_z_im, conv_z_k, &beta, conv_temp9, conv_z_n);
                                //(conv_temp4)^T*(conv_z_re) <--> ((287*287)*32)*(32*287)
                                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_z_n, conv_z_m, conv_z_k, &alpha, conv_temp4, conv_z_k, conv_z_re, conv_z_k, &beta, conv_temp10, conv_z_n);
                                // caculate imag part
                                rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_z_n, conv_z_m, &alpha, conv_temp9, conv_z_n, &beta2, conv_temp10, conv_z_n, BN_IM + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), conv_z_n);
                                hipLaunchKernelGGL(fft_ifftBN_multiply_fft_eta, blocks_exp_conv2, threads_exp_conv2, 0, 0, bn_RE + p * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), bn_IM + p * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), BN_RE + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), BN_IM + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), eta_RE + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), eta_IM + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), flag, q);
                        }
                }
        }
        for (int p = 0; p < NELASTIC; p++)
        {
                // x -- dim
                //(conv_e_big_re)^T*(Gx_re) <--> ((319*319)*319)*(319*nx)
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_x_n, conv_big_x_m, conv_big_x_k, &alpha, bn_RE + p * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), conv_big_x_k, conv_big_x_re, conv_big_x_k, &beta, conv_temp9, conv_big_x_n);
                //(conv_e_big_im)^T*(Gx_im) <--> ((319*319)*319)*(319*nx)
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_x_n, conv_big_x_m, conv_big_x_k, &alpha, bn_IM + p * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), conv_big_x_k, conv_big_x_im, conv_big_x_k, &beta, conv_temp12, conv_big_x_n);
                // caculate real part
                rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_big_x_n, conv_big_x_m, &alpha, conv_temp9, conv_big_x_n, &beta2, conv_temp12, conv_big_x_n, conv_e, conv_big_x_n);
                //(conv_e_big_im)^T*(Gx_re) <--> ((319*319)*319)*(319*nx)
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_x_n, conv_big_x_m, conv_big_x_k, &alpha, bn_IM + p * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), conv_big_x_k, conv_big_x_re, conv_big_x_k, &beta, conv_temp9, conv_big_x_n);
                //(conv_e_big_re)^T*(Gx_im) <--> ((319*319)*319)*(319*nx)
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_x_n, conv_big_x_m, conv_big_x_k, &alpha1, bn_RE + p * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), conv_big_x_k, conv_big_x_im, conv_big_x_k, &beta, conv_temp12, conv_big_x_n);
                // caculate imag part
                rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_big_x_n, conv_big_x_m, &alpha, conv_temp9, conv_big_x_n, &beta2, conv_temp12, conv_big_x_n, conv_temp10, conv_big_x_n);

                // y -- dim
                //(conv_big_temp3)^T*(Gy_re) <--> ((nx*319)*319)*(319*ny)
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_y_n, conv_big_y_m, conv_big_y_k, &alpha, conv_e, conv_big_y_k, conv_big_y_re, conv_big_y_k, &beta, conv_temp9, conv_big_y_n);
                //(conv_big_temp4)^T*(Gy_im) <--> ((nx*319)*319)*(319*ny)
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_y_n, conv_big_y_m, conv_big_y_k, &alpha, conv_temp10, conv_big_y_k, conv_big_y_im, conv_big_y_k, &beta, conv_temp12, conv_big_y_n);
                // caculate real part
                rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_big_y_n, conv_big_y_m, &alpha, conv_temp9, conv_big_y_n, &beta2, conv_temp12, conv_big_y_n, conv_temp11, conv_big_y_n);
                //(conv_big_temp3)^T*(Gy_im) <--> ((nx*319)*319)*(319*ny)
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_y_n, conv_big_y_m, conv_big_y_k, &alpha1, conv_e, conv_big_y_k, conv_big_y_im, conv_big_y_k, &beta, conv_temp9, conv_big_y_n);
                //(conv_big_temp4)^T*(Gy_re) <--> ((nx*319)*319)*(319*ny)
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_y_n, conv_big_y_m, conv_big_y_k, &alpha, conv_temp10, conv_big_y_k, conv_big_y_re, conv_big_y_k, &beta, conv_temp12, conv_big_y_n);
                // caculate imag part
                rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_big_y_n, conv_big_y_m, &alpha, conv_temp9, conv_big_y_n, &beta2, conv_temp12, conv_big_y_n, conv_temp10, conv_big_y_n);
                // z -- dim
                //(conv_big_temp7)^T*(Gz_re) <--> ((nx*ny)*319)*(319*nz)
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_z_n, conv_big_z_m, conv_big_z_k, &alpha, conv_temp11, conv_big_z_k, conv_big_z_re, conv_big_z_k, &beta, conv_temp9, conv_big_z_n);
                //(conv_big_temp8)^T*(Gz_im) <--> ((nx*ny)*319)*(319*nz)
                rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_z_n, conv_big_z_m, conv_big_z_k, &alpha, conv_temp10, conv_big_z_k, conv_big_z_im, conv_big_z_k, &beta, conv_temp12, conv_big_z_n);
                // caculate real part
                rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_big_z_n, conv_big_z_m, &alpha, conv_temp9, conv_big_z_n, &beta2, conv_temp12, conv_big_z_n, conv_temp10, conv_big_z_n);
                hipLaunchKernelGGL(elas_cal, blocks_ela2, threads_ela2, 0, 0, Elas + p * offset, conv_temp10, flag, Elas1);
                hipMemcpy(elas, Elas1, sizeof(Dtype) * NX*NY*NZ, hipMemcpyDeviceToHost);
		write_Bn_im (elas, elas, p);
        }
#ifdef SCLETD_DEBUG
  hipEventRecord(ed, NULL);
  hipEventSynchronize(ed);
  hipEventElapsedTime(&timer, st, ed);
  conv_calculate_time += timer;
#endif
}

// f1
void ac_calc_F1(float *field2_all, int out_iter, int detect_iter)
{

  int threads_x, threads_y, threads_z;
  dim3 blocks((ix4 - ix1) / THREADS_PER_BLOCK_X, (iy4 - iy1) / THREADS_PER_BLOCK_Y, (iz4 - iz1) / THREADS_PER_BLOCK_Z);
  dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

  dim3 blocks2B(nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, nz / THREADS_PER_BLOCK_Z);
  dim3 threads2B(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

  threads_x = nghost;
  dim3 blocks3((nghost + 2) / threads_x, (ny + 2 * 2) / THREADS_PER_BLOCK_Y, (nz + 2 * 2) / THREADS_PER_BLOCK_Z);
  dim3 threads3(threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

  threads_y = nghost;
  dim3 blocks4(nx / THREADS_PER_BLOCK_X, (nghost + 2) / threads_y, (nz + 2 * 2) / THREADS_PER_BLOCK_Z);
  dim3 threads4(THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);

  threads_z = nghost;
  dim3 blocks5(nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, (nghost + 2) / threads_z);
  dim3 threads5(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);

  dim3 blocksEe(nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, nac * nz / THREADS_PER_BLOCK_Z);
  dim3 threadsEe(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

  dim3 blocks_ela (NX / THREADS_PER_BLOCK_X, NY / THREADS_PER_BLOCK_Y, NZ / THREADS_PER_BLOCK_Z);
  dim3 threads_ela (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  dim3 blocks_exp_conv((NX + Approx) / THREADS_PER_BLOCK_X, Approx / THREADS_PER_BLOCK_Y, Approx / THREADS_PER_BLOCK_Z);
  dim3 threads_exp_conv(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  dim3 blocks_exp_conv1((NX + Approx) / THREADS_PER_BLOCK_X, (NX + Approx) / THREADS_PER_BLOCK_Y, Approx / THREADS_PER_BLOCK_Z);
  dim3 threads_exp_conv1(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

  int n;
  if (iter == 0) {
        //CPU --> DCU elastic
        hipMemcpy (C4D, C4d, DIM * DIM * DIM * DIM * sizeof (double), hipMemcpyHostToDevice);
        for (n = 0; n < nac; n++)
        {
                hipMemcpy (Epsilon2d[n], epsilon2d[n], DIM * DIM * sizeof (double), hipMemcpyHostToDevice);
                hipMemcpy (Sigma2d[n], sigma2d[n], DIM * DIM * sizeof (double), hipMemcpyHostToDevice);
                hipLaunchKernelGGL(elastic_calculate_sigma2d, blocks_ela, threads_ela, 0, 0,
                                   Sigma2d[n], C4D, Epsilon2d[n]);
                hipLaunchKernelGGL(elastic_calculate_n, blocks_ela, threads_ela, 0, 0,
                                   n11, n22, n33);
        }
        hipLaunchKernelGGL(FFT_IFFT_Bn_exp, blocks_exp_conv, threads_exp_conv, 0, 0, conv_x_re, conv_x_im, conv_y_re, conv_y_im, conv_z_re, conv_z_im);
        hipLaunchKernelGGL(conv_exp_matrix, blocks_exp_conv1, threads_exp_conv1, 0, 0, conv_big_x_re, conv_big_x_im, conv_big_y_re, conv_big_y_im, conv_big_z_re, conv_big_z_im);
  }
  tt = 0;
  while (tt < 1)
  {
    dgemm_time = 0.0f;
    F1_time = 0.0f;
    F2_time = 0.0f;
    fieldE1_time = 0.0f;
    trans_time = 0.0f;
    trans_Memcpy_time = 0.0f;
    trans_MPI_time = 0.0f;
    trans_pack_time = 0.0f;
    trans_enlarge_time = 0.0f;
    trans_mu_time = 0.0f;
    trans_unpack_time = 0.0f;
    updateU_new_time = 0.0f;
    conv_transfer_time = 0.0f;
    conv_calculate_time = 0.0f;
    elastic_calculate_time = 0.0f;
#if 1
    if (iter > 0)
    {
      hipLaunchKernelGGL(Ee_dev, blocksEe, threadsEe, 0, 0,
                         fieldE, fielde);
      for (int m = 0; m < nac; m++)
      {
        hipMemcpy(ac[m].fieldE, fieldE + m * offset, nx * ny * nz * sizeof(double), hipMemcpyDeviceToHost);
//	write_Bn_im (ac[m].fieldE, ac[m].fieldE, m);
      }
      check_soln_new(tt);
    }
#endif
    if (simulate_checkpoint == 1 && iter < 41)
    {
      for (int t = 0; t < nac; t++)
      {
        myball(iter, rad, ac[t].fieldE);
      }
      for (n = 0; n < nac; n++)
      {
        hipMemcpy(fieldE + n * offset, ac[n].fieldE, nx * ny * nz * sizeof(double), hipMemcpyHostToDevice);
      }
    }
    if (iter == 0)
    {
      check_soln_new(tt);
    }
    tt += 1;
    //        tt   +=   dt;
    runtime = MPI_Wtime();
#ifdef SCLETD_DEBUG
    hipEventRecord(st2, NULL);
    hipEventSynchronize(st2);
#endif

    transfer();

#ifdef SCLETD_DEBUG
    hipEventRecord(ed2, NULL);
    hipEventSynchronize(ed2);
    hipEventElapsedTime(&timer2, st2, ed2);
    trans_time = timer2;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif
    elastic_calculate();
#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    elastic_calculate_time += timer;
#endif
    for (n = 0; n < nac; n++)
    {
#ifdef SCLETD_DEBUG
      hipEventRecord(st, NULL);
      hipEventSynchronize(st);
#endif
      elastic_transfer(n);
#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize(ed);
    hipEventElapsedTime(&timer, st, ed);
    conv_transfer_time += timer;
    hipEventRecord(st, NULL);
    hipEventSynchronize(st);
#endif


      rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, x_n, x_k, x_m, &alpha, fieldE + n * offset, x_k, mpxi, x_m, &beta, f1, x_n);
      rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, y_n, y_k, y_m, &alpha, f1, y_k, mpyi, y_m, &beta, f2, y_n);
      rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, z_n, z_k, z_m, &alpha, f2, z_k, mpzi, z_m, &beta, fielde + n * offset, z_n);

#ifdef SCLETD_DEBUG
      hipEventRecord(ed, NULL);
      hipEventSynchronize(ed);
      hipEventElapsedTime(&timer, st, ed);
      dgemm_time += timer;
      hipEventRecord(st, NULL);
      hipEventSynchronize(st);
#endif

     hipLaunchKernelGGL(ac_calc_F1_dev, blocks, threads, 0, 0,
                        fieldE, f1, fieldEu_left + n * u_lr, fieldEu_right + n * u_lr,
                        fieldEu_top + n * u_tb, fieldEu_bottom + n * u_tb, fieldEu_front + n * u_fb,
                        fieldEu_back + n * u_fb, ac[n].LE, ac[n].KE, n);

#ifdef SCLETD_DEBUG
      hipEventRecord(ed, NULL);
      hipEventSynchronize(ed);
      hipEventElapsedTime(&timer, st, ed);
      F1_time += timer;
      hipEventRecord(st, NULL);
      hipEventSynchronize(st);
#endif

      if (ANISOTROPIC == 1)
      {
        hipLaunchKernelGGL(unpack_lr, blocks3, threads3, 0, 0,
                           fieldEr, fieldEr_left + n * lr_size, fieldEr_right + n * lr_size);

        hipLaunchKernelGGL(unpack_tb, blocks4, threads4, 0, 0,
                           fieldEr, fieldEr_top + n * tb_size, fieldEr_bottom + n * tb_size);

        hipLaunchKernelGGL(unpack_fb, blocks5, threads5, 0, 0,
                           fieldEr, fieldEr_front + n * fb_size, fieldEr_back + n * fb_size);

        hipLaunchKernelGGL(unpack_all, blocks, threads, 0, 0, fieldEr, fieldE + n * offset);
        hipLaunchKernelGGL(anisotropic_calc_dev, blocks, threads, 0, 0,
                           f2, fieldEr, lambda + n * DIM * DIM, n);
      }

#ifdef SCLETD_DEBUG
      hipEventRecord(ed, NULL);
      hipEventSynchronize(ed);
      hipEventElapsedTime(&timer, st, ed);
      F2_time += timer;
      hipEventRecord(st, NULL);
      hipEventSynchronize(st);
#endif
      //elastic_transfer(n);

      hipLaunchKernelGGL(ac_add_F1_F2_dev, blocks, threads, 0, 0,
                         f1, f2, Elas + n * offset, fieldE + n * offset, epn2, ac[n].LE, ANISOTROPIC, n);

#ifdef SCLETD_DEBUG
      hipEventRecord(ed, NULL);
      hipEventSynchronize(ed);
      hipEventElapsedTime(&timer, st, ed);
      fieldE1_time += timer;
      hipEventRecord(st, NULL);
      hipEventSynchronize(st);
#endif

      rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, x_n, x_k, x_m, &alpha, f1, x_k, mpxi, x_m, &beta, f2, x_n);
      rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, y_n, y_k, y_m, &alpha, f2, y_k, mpyi, y_m, &beta, f1, y_n);
      rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, z_n, z_k, z_m, &alpha, f1, z_k, mpzi, z_m, &beta, f2, z_n);

#ifdef SCLETD_DEBUG
      hipEventRecord(ed, NULL);
      hipEventSynchronize(ed);
      hipEventElapsedTime(&timer, st, ed);
      dgemm_time += timer;
      hipEventRecord(st, NULL);
      hipEventSynchronize(st);
#endif

      hipLaunchKernelGGL(ac_updateU_new, blocks, threads, 0, 0,
                         n, fielde + n * offset, f2, ddx, ddy, ddz, ac[n].LE, ac[n].KE, epn2, fieldEr);

#ifdef SCLETD_DEBUG
      hipEventRecord(ed, NULL);
      hipEventSynchronize(ed);
      hipEventElapsedTime(&timer, st, ed);
      updateU_new_time += timer;
      hipEventRecord(st, NULL);
      hipEventSynchronize(st);
#endif

      rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, x_n, x_k, x_m, &alpha, fielde + n * offset, x_k, mpx, x_m, &beta, f1, x_n);
      rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, y_n, y_k, y_m, &alpha, f1, y_k, mpy, y_m, &beta, f2, y_n);
      rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose, z_n, z_k, z_m, &alpha, f2, z_k, mpz, z_m, &beta, fielde + n * offset, z_n);

#ifdef SCLETD_DEBUG
      hipEventRecord(ed, NULL);
      hipEventSynchronize(ed);
      hipEventElapsedTime(&timer, st, ed);
      dgemm_time += timer;
#endif
    }

    ot1 += 1;
    if (ot1 >= nout)
    {
      ot1 = 0;
      irun += 1;
     for (n = 0; n < nac; n++)
     {
       hipLaunchKernelGGL(Dtype2B_E, blocks2B, threads2B, 0, 0,
                          fielde + n * offset, field2B + n * offset);
     }
       write_field2B (irun);
    }

    ot2 += 1;
    if (ot2 >= nchk)
    {
      ot2 = 0;
      write_chk();
      chk = (chk + 1) % 2;
    }
    //MPI_Barrier(MPI_COMM_WORLD);

    if (myrank == prank)
    {
#ifdef SCLETD_DEBUG
      printf("*******************************\n");
      printf("dgemm_time\t%lf\n", dgemm_time);
      printf("conv_transfer_time\t%lf\n", conv_transfer_time);
      printf("conv_calculate_time\t%lf\n", conv_calculate_time);
      printf("elastic_calculate_time\t%lf\n", elastic_calculate_time);
      printf("chemical_time\t%lf\n", F1_time);
      printf("anisotropic_time\t%lf\n", F2_time);
      printf("nonlinear_add_time\t%lf\n", fieldE1_time);
      printf("trans_time\t%lf\n", trans_time);
      printf("trans_Memcpy_time\t%lf\n", trans_Memcpy_time);
      printf("trans_MPI_time\t%lf\n", trans_MPI_time);
      printf("trans_pack_time\t%lf\n", trans_pack_time);
      printf("trans_enlarge_time\t%lf\n", trans_enlarge_time);
      printf("trans_mu_time\t%lf\n", trans_mu_time);
      printf("trans_unpack_time\t%lf\n", trans_unpack_time);
      printf("updateU_new_time\t%lf\n", updateU_new_time);
#endif
      printf("runtime\t%lf\n", MPI_Wtime() - runtime);
      printf("--------------------------------------------------!\n");
    }
    iter++;
  }
}

// size是指将数组压缩至char之后的大小
void float2bin(float *input, unsigned char *buff, int size)
{
  unsigned char temp_int = 1;
  unsigned char temp_c = 0;
  for (int temp_i = 0; temp_i < size; temp_i++)
  {
    temp_int = 1;
    temp_c = 0;
    for (int k = 0; k < 8; k++)
    {
      if (input[temp_i * 8 + k] > 0.1)
      {
        temp_c = temp_c | temp_int;
        temp_int = temp_int << 1;
        // temp_c = temp_c << 1;
        // printf("k:%d %d yes\n", k, temp_c);
      }
      else
      {
        temp_int = temp_int << 1;
        // printf("k:%d %d\n", k, temp_c);
      }
    }

    buff[temp_i] = temp_c;
  }
}

void compress_save_output(char *output_path, float *input, unsigned char *output, int size)
{
  float2bin(input, output, size);

  FILE *fp_out = fopen(output_path, "w"); //打开输出文件
  if (fp_out == NULL)
  {
    printf("Fail to save\n");
    exit(0);
  }

  for (int i = 0; i < size; i++)
  // for (int i = 0; i < 64*64*64*32; i++)
  {
    // printf("%f\n", (output_buffer + ((buffer_i + 1) % 2) * buffer_block_size)[i]);
    // fprintf(fp_out, "%f\n", (output_buffer + ((buffer_i + 1) % 2) * buffer_block_size)[i]);
    fprintf(fp_out, "%d\n", (output)[i]); // res_4_conv2
    // if ((output)[i] != 0)
    //   printf("%d\n", (output)[i]);
  }
  fclose(fp_out);
}
