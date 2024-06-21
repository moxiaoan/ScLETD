#include "ScLETD.h"
#include "anisotropic_hip.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

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
eta_eta(Dtype *f, Dtype *fe)
{
  int i, j, k;
  int dim = (nx - 2 * nghost) / 1;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f[k * dim * dim + j * dim + i] = fe[k * dim * dim + j * dim + i] * fe[k * dim * dim + j * dim + i];
}
#if 1
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

void conv_transfer(double *f)
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
                     conv_e, tmpy_RE2, flag);
}
#endif
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
fft_ifftBN_multiply_fft_eta(Dtype *Out_re, Dtype *Out_im, Dtype *Bn_fftre, Dtype *Bn_fftim, Dtype *fft_re, Dtype *fft_im, int flag)
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
                if (flag == 0) {
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
elas_cal(Dtype *elas_small, Dtype *elas_big, int flag, double *tmpy_re)
{
        int i, j, k;
        double a = nx - 2 * nghost + Approx - 1;
	int dim = nx - 2 * nghost;
        double c = 1.0 / a / a / a;
        int num = Approx;

        i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

        elas_small[(k + nghost) * nx * ny + (j + nghost) * nx + (i + nghost)] = c * 2.0 * tmpy_re[k * dim * dim + j * dim + i] * elas_big[(k + num - 1) * (NX + num - 1) * (NY + num - 1) + (j + num - 1) * (NX + num - 1) + i + num - 1];
}
#if 0
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

__global__ void
conv_left_right_unpack(double *fieldr, double *fieldr_left, double *fieldr_right)
{
  int f_j, f_k;
  int lr_k, lr_j;
  int i, j, k;

  f_j = (nx - 2 * nghost) + Approx * 2;
  f_k = ((nx - 2 * nghost) + Approx * 2) * ((ny - 2 * nghost) + Approx * 2);
  lr_j = Approx;
  lr_k = Approx * ((ny - 2 * nghost) + Approx * 2);

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

// if (cart_id[0] == 0)
// {
//   if (right >= 0)
//   {
//     fieldr[f_k * k + f_j * j + i + (nx - 2 * nghost) + Approx] = fieldr_right[lr_k * k + lr_j * j + i];
//   }
// }
// else
// {
    if (left >= 0)
    {
      fieldr[f_k * k + f_j * j + i] = fieldr_left[lr_k * k + lr_j * j + i];
    }
    if (right >= 0)
    {
      fieldr[f_k * k + f_j * j + i + (nx - 2 * nghost) + Approx] = fieldr_right[lr_k * k + lr_j * j + i];
    }
//  }
}

__global__ void
conv_top_bottom_unpack(double *fieldr, double *fieldr_top, double *fieldr_bottom)
{
  int f_j, f_k;
  int tb_k, tb_j;
  int i, j, k;
  f_j = (nx - 2 * nghost) + Approx * 2;
  f_k = ((nx - 2 * nghost) + Approx * 2) * ((ny - 2 * nghost) + Approx * 2);
  tb_j = nx - 2 * nghost;
  tb_k = (nx - 2 * nghost) * Approx;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

// if (cart_id[1] == 0)
// {
//   if (bottom >= 0)
//   {
//     fieldr[f_k * k + f_j * (j + (ny - 2 * nghost) + Approx) + i + Approx] = fieldr_bottom[tb_k * k + tb_j * j + i];
//   }
// }
// else
// {
    if (top >= 0)
    {
      fieldr[f_k * k + f_j * j + i + Approx] = fieldr_top[tb_k * k + tb_j * j + i];
    }
    if (bottom >= 0)
    {
      fieldr[f_k * k + f_j * (j + (ny - 2 * nghost) + Approx) + i + Approx] = fieldr_bottom[tb_k * k + tb_j * j + i];
    }
//  }
}

__global__ void
conv_front_back_unpack(double *fieldr, double *fieldr_front, double *fieldr_back)
{
  int f_j, f_k;
  int fb_j, fb_k;
  int i, j, k;

  f_j = (nx - 2 * nghost) + Approx * 2;
  f_k = ((nx - 2 * nghost) + Approx * 2) * ((ny - 2 * nghost) + Approx * 2);
  fb_j = nx - 2 * nghost;
  fb_k = (nx - 2 * nghost) * (ny - 2 * nghost);
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
// if (cart_id[2] == 0)
// {
//   if (back >= 0)
//   {
//     fieldr[f_k * (k + (nz - 2 * nghost) + Approx) + f_j * (j + Approx) + i + Approx] = fieldr_back[fb_k * k + fb_j * j + i];
//   }
// }
// else
// {
    if (front >= 0)
    {
      fieldr[f_k * k + f_j * (j + Approx) + i + Approx] = fieldr_front[fb_k * k + fb_j * j + i];
    }
    if (back >= 0)
    {
      fieldr[f_k * (k + (nz - 2 * nghost) + Approx) + f_j * (j + Approx) + i + Approx] = fieldr_back[fb_k * k + fb_j * j + i];
    }
//  }
}

__global__ void
conv_inner_unpack(double *fieldr, double *field)
{
  int f_j, f_k;
  int i, j, k;
  f_j = (nx - 2 * nghost) + Approx * 2;
  f_k = ((nx - 2 * nghost) + Approx * 2) * ((ny - 2 * nghost) + Approx * 2);
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  fieldr[f_k * (k + Approx) + f_j * (j + Approx) + (i + Approx)] = field[(nx - 2 * nghost) * (ny - 2 * nghost) * k + (nx - 2 * nghost) * j + i];
}

void conv_transfer(int n)
{
  int threads_x, threads_y, threads_z;
  hipMemcpy(conv_s_front, &tmpy_RE2[0], sizeof(Dtype) * conv_fb_size, hipMemcpyDeviceToHost);
  hipMemcpy(conv_s_back, &tmpy_RE2[NX * NY * (NZ - Approx)], sizeof(Dtype) * conv_fb_size, hipMemcpyDeviceToHost);

  MPI_Startall(4, conv_ireq_front_back);
  MPI_Waitall(4, conv_ireq_front_back, status);

  hipMemcpy(conv_R_front, conv_r_front, sizeof(Dtype) * conv_fb_size, hipMemcpyHostToDevice);
  hipMemcpy(conv_R_back, conv_r_back, sizeof(Dtype) * conv_fb_size, hipMemcpyHostToDevice);

#if 1
  dim3 blocks_conv_tb_pack(NX / THREADS_PER_BLOCK_X, Approx / THREADS_PER_BLOCK_Y, (NZ + Approx * 2) / THREADS_PER_BLOCK_Z);
  dim3 threads_conv_tb_pack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(conv_top_bottom_pack, blocks_conv_tb_pack, threads_conv_tb_pack, 0, 0,
                     tmpy_RE2, conv_S_top, conv_S_bottom, conv_R_front, conv_R_back);

  hipMemcpy(conv_s_top, conv_S_top, sizeof(Dtype) * conv_tb_size, hipMemcpyDeviceToHost);
  hipMemcpy(conv_s_bottom, conv_S_bottom, sizeof(Dtype) * conv_tb_size, hipMemcpyDeviceToHost);

  MPI_Startall(4, conv_ireq_top_bottom);
  MPI_Waitall(4, conv_ireq_top_bottom, status);

  hipMemcpy(conv_R_top, conv_r_top, sizeof(Dtype) * conv_tb_size, hipMemcpyHostToDevice);
  hipMemcpy(conv_R_bottom, conv_r_bottom, sizeof(Dtype) * conv_tb_size, hipMemcpyHostToDevice);

  dim3 blocks_conv_lr_pack(Approx / THREADS_PER_BLOCK_X, ((ny - 2 * nghost) + Approx * 2) / THREADS_PER_BLOCK_Y, ((nz - 2 * nghost) + Approx * 2) / THREADS_PER_BLOCK_Z);
  dim3 threads_conv_lr_pack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(conv_left_right_pack, blocks_conv_lr_pack, threads_conv_lr_pack, 0, 0,
                     tmpy_RE2, conv_S_left, conv_S_right, conv_R_top, conv_R_bottom, conv_R_front, conv_R_back);
  hipMemcpy(conv_s_left, conv_S_left, sizeof(Dtype) * conv_lr_size, hipMemcpyDeviceToHost);
  hipMemcpy(conv_s_right, conv_S_right, sizeof(Dtype) * conv_lr_size, hipMemcpyDeviceToHost);

  MPI_Startall(4, conv_ireq_left_right);
  MPI_Waitall(4, conv_ireq_left_right, status);

  hipMemcpy(conv_R_left, conv_r_left, sizeof(Dtype) * conv_lr_size, hipMemcpyHostToDevice);
  hipMemcpy(conv_R_right, conv_r_right, sizeof(Dtype) * conv_lr_size, hipMemcpyHostToDevice);
//need fix
  threads_x = Approx;
  dim3 blocks_conv_lr_unpack(Approx / threads_x, ((ny - 2 * nghost) + 2 * Approx) / THREADS_PER_BLOCK_Y, ((nz - 2 * nghost) + 2 * Approx) / THREADS_PER_BLOCK_Z);
  dim3 threads_conv_lr_unpack(threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(conv_left_right_unpack, blocks_conv_lr_unpack, threads_conv_lr_unpack, 0, 0,
                     conv_e, conv_R_left, conv_R_right);
  threads_y = Approx;
  dim3 blocks_conv_tb_unpack((nx - 2 * nghost) / THREADS_PER_BLOCK_X, Approx / threads_y, ((nz - 2 * nghost) + 2 * Approx) / THREADS_PER_BLOCK_Z);
  dim3 threads_conv_tb_unpack(THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(conv_top_bottom_unpack, blocks_conv_tb_unpack, threads_conv_tb_unpack, 0, 0,
                     conv_e, conv_R_top, conv_R_bottom);
  threads_z = Approx;
  dim3 blocks_conv_fb_unpack((nx - 2 * nghost) / THREADS_PER_BLOCK_X, (ny - 2 * nghost) / THREADS_PER_BLOCK_Y, Approx / threads_z);
  dim3 threads_conv_fb_unpack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);
  hipLaunchKernelGGL(conv_front_back_unpack, blocks_conv_fb_unpack, threads_conv_fb_unpack, 0, 0,
                     conv_e, conv_R_front, conv_R_back);
  dim3 blocks_conv_inner_unpack((nx - 2 * nghost) / THREADS_PER_BLOCK_X, (ny - 2 * nghost) / THREADS_PER_BLOCK_Y, (nz - 2 * nghost) / THREADS_PER_BLOCK_Z);
  dim3 threads_conv_inner_unpack(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  hipLaunchKernelGGL(conv_inner_unpack, blocks_conv_inner_unpack, threads_conv_inner_unpack, 0, 0,
                     conv_e, tmpy_RE2);
#endif
}

__global__ void
conv_calculate(Dtype *elas, Dtype *ifftBn1, Dtype *conv_e, Dtype *tmpy_re)
{
        int i, j, k;
        int x, y, z;
        int num = Approx;
        int dim = (nx - 2 * nghost);
        int dp0 = dim * procs[0];
        int dp1 = dim * procs[1];
        int dp2 = dim * procs[2];
        double c = 1.0 / dp0 / dp1 / dp2;
        int stride = 16;
        int f_k = (NX + 2 * Approx) * (NY + 2 * Approx);
        int f_j = (NX + 2 * Approx);
        int size = num * num * num;

        i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

        double temp = 0.0;
//      if (i >= Approx && j >= Approx && k >= Approx) {
        for (z = 0; z < Approx; z++) {
                for (y = 0; y < Approx; y++) {
                        for (x = 0; x < Approx; x++) {
                          temp += c * conv_e[f_k * (k + z + 1) + f_j * (j + y + 1) + (i + x + 1)] * ifftBn1[(Approx - 1 - z) * num * num + (Approx - 1 - y) * num + (Approx - 1 - x)];
                        }
                }
        }
//      }
#if 1
//      if (i < NX - Approx && j >= Approx && k >= Approx) {
        for (z = 0; z < Approx; z++) {
                for (y = 0; y < Approx; y++) {
                        for (x = 0; x < Approx; x++) {
                          temp += c * conv_e[f_k * (k + z + 1) + f_j * (j + y + 1) + (i + x + Approx + 1)] * ifftBn1[(Approx - 1 - z) * num * num + (Approx - 1 - y) * num + (Approx - 1 - x) + 4 * size];
                        }
                }
        }
//      }
//      if (j < NX - Approx && i >= Approx && k >= Approx) {
        for (z = 0; z < Approx; z++) {
                for (y = 0; y < Approx; y++) {
                        for (x = 0; x < Approx; x++) {
                          temp += c * conv_e[f_k * (k + z + 1) + f_j * (j + y + Approx + 1) + (i + x + 1)] * ifftBn1[(Approx - 1 - z) * num * num + (Approx - 1 - y) * num + (Approx - 1 - x) + 2 * size];
                        }
                }
        }
//      }
//      if (j < NX - Approx && i < NX - Approx && k >= Approx) {
        for (z = 0; z < Approx; z++) {
                for (y = 0; y < Approx; y++) {
                        for (x = 0; x < Approx; x++) {
                          temp += c * conv_e[f_k * (k + z + 1) + f_j * (j + y + Approx + 1) + (i + x + Approx + 1)] * ifftBn1[(Approx - 1 - z) * num * num + (Approx - 1 - y) * num + (Approx - 1 - x) + 6 * size];
                        }
                }
        }
//      }
//      if (k < NX - Approx && i >= Approx && j >= Approx) {
        for (z = 0; z < Approx; z++) {
                for (y = 0; y < Approx; y++) {
                        for (x = 0; x < Approx; x++) {
                          temp += c * conv_e[f_k * (k + z + Approx + 1) + f_j * (j + y + 1) + (i + x + 1)] * ifftBn1[(Approx - 1 - z) * num * num + (Approx - 1 - y) * num + (Approx - 1 - x) + size];
                        }
                }
        }
//      }
//      if (k < NX - Approx && i < NX - Approx && j >= Approx) {
        for (z = 0; z < Approx; z++) {
                for (y = 0; y < Approx; y++) {
                        for (x = 0; x < Approx; x++) {
                          temp += c * conv_e[f_k * (k + z + Approx + 1) + f_j * (j + y + 1) + (i + x + Approx + 1)] * ifftBn1[(Approx - 1 - z) * num * num + (Approx - 1 - y) * num + (Approx - 1 - x) + 5 * size];
                        }
                }
        }
//      }
//      if (k < NX - Approx && j < NX - Approx && i >= Approx) {
        for (z = 0; z < Approx; z++) {
                for (y = 0; y < Approx; y++) {
                        for (x = 0; x < Approx; x++) {
                          temp += c * conv_e[f_k * (k + z + Approx + 1) + f_j * (j + y + Approx + 1) + (i + x + 1)] * ifftBn1[(Approx - 1 - z) * num * num + (Approx - 1 - y) * num + (Approx - 1 - x) + 3 * size];
                        }
                }
        }
//      }
#endif
#if 1
//      if (k < NX - Approx && j < NX - Approx && i < NX - Approx) {
        for (z = 0; z < Approx; z++) {
                for (y = 0; y < Approx; y++) {
                        for (x = 0; x < Approx; x++) {
                          temp += c * conv_e[f_k * (k + z + Approx + 1) + f_j * (j + y + Approx + 1) + (i + x + Approx + 1)] * ifftBn1[(Approx - 1 - z) * num * num + (Approx - 1 - y) * num + (Approx - 1 - x) + 7 * size];
                        }
                }
        }
//      }
#endif
        elas[(k+nghost) * nx * ny + (j+nghost) * nx + (i+nghost)] = 2.0 * temp * tmpy_re[k * dim * dim + j * dim + i];
}
#endif
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
	dim3 blocks_exp_conv((NX + Approx) / THREADS_PER_BLOCK_X, Approx / THREADS_PER_BLOCK_Y, Approx / THREADS_PER_BLOCK_Z);
	dim3 threads_exp_conv(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
	dim3 blocks_exp_conv1((NX + Approx) / THREADS_PER_BLOCK_X, (NX + Approx) / THREADS_PER_BLOCK_Y, Approx / THREADS_PER_BLOCK_Z);
	dim3 threads_exp_conv1(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
        dim3 blocks_exp_conv2((NX + Approx) / THREADS_PER_BLOCK_X, (NX + Approx) / THREADS_PER_BLOCK_Y, (NX + Approx) / THREADS_PER_BLOCK_Z);
        dim3 threads_exp_conv2(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
	hipMemcpy (fieldE, ac[0].fieldE, sizeof (Dtype) * offset, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(down_sample_eta, blocks_ela2, threads_ela2, 0, 0, tmpy_RE1, fieldE);
        hipLaunchKernelGGL(eta_eta, blocks_ela2, threads_ela2, 0, 0, tmpy_RE2, tmpy_RE1);

	if (iter == 0)
	{
        	hipLaunchKernelGGL(initial_elas, blocks, threads, 0, 0, Elas);
        	hipLaunchKernelGGL(initial_fftre_im, blocks_ela3, threads_ela3, 0, 0, tmpy_fftRE);
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
                        rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, Gz_n, Gz_k, &alpha, temp7, Gz_n, &beta2, temp8, Gz_n, tmpy_fftRE + flag * ela_size, Gz_n);
                        hipLaunchKernelGGL(ifft_multiply_scale, blocks_scale, threads_scale, 0, 0, tmpy_fftRE + flag * ela_size);
                }
                hipMemcpy(tmpy_fftre, tmpy_fftRE, sizeof(Dtype) * 8 * ela_size, hipMemcpyDeviceToHost);
                //kernel matrix
                MPI_Allreduce(&tmpy_fftre[0], &fftre[0], 8 * ela_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
 	        //write_Bn(fftre);
                hipMemcpy(fftRE, fftre, sizeof(Dtype) * 8 * ela_size, hipMemcpyHostToDevice);
#if 0
	}
		int p = 0;
		conv_transfer(p);
		hipLaunchKernelGGL(conv_calculate, blocks_ela2, threads_ela2, 0, 0, Elas, fftRE, conv_e, tmpy_RE1);
#endif
#if 1
		hipLaunchKernelGGL(FFT_IFFT_Bn_exp, blocks_exp_conv, threads_exp_conv, 0, 0, conv_x_re, conv_x_im, conv_y_re, conv_y_im, conv_z_re, conv_z_im);
                for (flag = 0; flag < 8; flag++)
                {
                        // x -- dim
                        //(ifftBN)^T*(conv_x_re) <--> ((32*32)*32)*(32*287)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_x_n, conv_x_m, conv_x_k, &alpha, fftRE + flag * ela_size, conv_x_k, conv_x_re, conv_x_k, &beta, conv_temp5, conv_x_n);
                        //(ifftBN)^T*(conv_x_im) <--> ((32*32)*32)*(32*287)
                        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_x_n, conv_x_m, conv_x_k, &alpha, fftRE + flag * ela_size, conv_x_k, conv_x_im, conv_x_k, &beta, conv_temp6, conv_x_n);

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
		}
		hipLaunchKernelGGL(conv_exp_matrix, blocks_exp_conv1, threads_exp_conv1, 0, 0, conv_big_x_re, conv_big_x_im, conv_big_y_re, conv_big_y_im, conv_big_z_re, conv_big_z_im);
	}
        conv_transfer(tmpy_RE2);
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
                hipLaunchKernelGGL(fft_ifftBN_multiply_fft_eta, blocks_exp_conv2, threads_exp_conv2, 0, 0, bn_RE, bn_IM, BN_RE + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), BN_IM + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), eta_RE + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), eta_IM + flag * (NX + Approx - 1) * (NY + Approx - 1) * (NZ + Approx - 1), flag);
	}
        // x -- dim
        //(conv_e_big_re)^T*(Gx_re) <--> ((319*319)*319)*(319*nx)
        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_x_n, conv_big_x_m, conv_big_x_k, &alpha, bn_RE, conv_big_x_k, conv_big_x_re, conv_big_x_k, &beta, conv_temp9, conv_big_x_n);
        //(conv_e_big_im)^T*(Gx_im) <--> ((319*319)*319)*(319*nx)
        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_x_n, conv_big_x_m, conv_big_x_k, &alpha, bn_IM, conv_big_x_k, conv_big_x_im, conv_big_x_k, &beta, conv_temp12, conv_big_x_n);
        // caculate real part
        rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, conv_big_x_n, conv_big_x_m, &alpha, conv_temp9, conv_big_x_n, &beta2, conv_temp12, conv_big_x_n, conv_e, conv_big_x_n);
        //(conv_e_big_im)^T*(Gx_re) <--> ((319*319)*319)*(319*nx)
        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_x_n, conv_big_x_m, conv_big_x_k, &alpha, bn_IM, conv_big_x_k, conv_big_x_re, conv_big_x_k, &beta, conv_temp9, conv_big_x_n);
        //(conv_e_big_re)^T*(Gx_im) <--> ((319*319)*319)*(319*nx)
        rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_none, conv_big_x_n, conv_big_x_m, conv_big_x_k, &alpha1, bn_RE, conv_big_x_k, conv_big_x_im, conv_big_x_k, &beta, conv_temp12, conv_big_x_n);
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
        hipLaunchKernelGGL(elas_cal, blocks_ela2, threads_ela2, 0, 0, Elas, conv_temp10, flag, tmpy_RE1);
#endif
	hipMemcpy (ac[0].felas, Elas, sizeof (Dtype) * offset, hipMemcpyDeviceToHost);
//#ifdef SCLETD_DEBUG
//  hipEventRecord(ed, NULL);
//  hipEventSynchronize(ed);
//  hipEventElapsedTime(&timer, st, ed);
//  conv_calculate_time += timer;
//#endif
}
