#include "ScLETD.h"
#include "ScLETD_hip.h"
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>

void init_para(void)
{
  if (left < 0)
  {
    ix1 = 0;
    ix2 = ix1;
    ix3 = ix1 + nx - nghost;
    ix4 = nx;
  }
  else if (right < 0)
  {
    ix1 = 0;
    ix2 = ix1 + nghost;
    ix3 = ix1 + nx;
    ix4 = nx;
  }
  else
  {
    ix1 = 0;
    ix2 = ix1 + nghost;
    ix3 = ix1 + nx - nghost;
    ix4 = nx;
  }
  lnx = ix4 - ix1;
  if (periodic)
  {
    gnx = lnx * procs[0] - 2 * nghost * procs[0];
  }
  else
  {
    gnx = lnx * procs[0] - 2 * nghost * (procs[0] - 1);
  }

  if (top < 0)
  {
    iy1 = 0;
    iy2 = iy1;
    iy3 = iy1 + ny - nghost;
    iy4 = ny;
  }
  else if (bottom < 0)
  {
    iy1 = 0;
    iy2 = iy1 + nghost;
    iy3 = iy1 + ny;
    iy4 = ny;
  }
  else
  {
    iy1 = 0;
    iy2 = iy1 + nghost;
    iy3 = iy1 + ny - nghost;
    iy4 = ny;
  }
  lny = iy4 - iy1;
  if (periodic)
  {
    gny = lny * procs[1] - 2 * nghost * procs[1];
  }
  else
  {
    gny = lny * procs[1] - 2 * nghost * (procs[1] - 1);
  }

  if (front < 0)
  {
    iz1 = 0;
    iz2 = iz1;
    iz3 = iz1 + nz - nghost;
    iz4 = nz;
  }
  else if (back < 0)
  {
    iz1 = 0;
    iz2 = iz1 + nghost;
    iz3 = iz1 + nz;
    iz4 = nz;
  }
  else
  {
    iz1 = 0;
    iz2 = iz1 + nghost;
    iz3 = iz1 + nz - nghost;
    iz4 = nz;
  }
  lnz = iz4 - iz1;
  if (periodic)
  {
    gnz = lnz * procs[2] - 2 * nghost * procs[2];
  }
  else
  {
    gnz = lnz * procs[2] - 2 * nghost * (procs[2] - 1);
  }

  hx = (xmax - xmin) / (gnx - 1);
  hy = (ymax - ymin) / (gny - 1);
  hz = (zmax - zmin) / (gnz - 1);
}

void init_KL()
{
  ac[0].LE = 1.0;
  ac[1].LE = 1.0;
  ac[2].LE = 1.0;
  ac[3].LE = 1.0;
  ac[0].KE = 42.0;
  ac[1].KE = 42.0;
  ac[2].KE = 42.0;
  ac[3].KE = 42.0;
}

void init_vars(void)
{
  alpha = 1.0;
  beta = 0.0;
  alpha1 = -1.0;
  beta1 = -1.0;
  beta2 = 1.0;
  init_KL();

  offset2 = 3 * 3;
  offset = nx * ny * nz;
  lr_size = (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2);
  tb_size = nx * (nghost + 2) * (nz + (nghost + 2) * 2);
  fb_size = nx * ny * (nghost + 2);
  u_fb = nx * ny;
  u_tb = nx * nz;
  u_lr = ny * nz;
  x_m = nx;
  x_n = ny * nz;
  x_k = nx;
  y_m = ny;
  y_n = nz * nx;
  y_k = ny;
  z_m = nz;
  z_n = nx * ny;
  z_k = nz;

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

  //convolution offset
  conv_lr_size = Approx * (NY + Approx * 2) * (NZ + Approx * 2);
  conv_tb_size = NX * Approx * (NZ + Approx * 2);
  conv_fb_size = NX * NY * Approx;
}

void alloc_vars(void)
{
  int n;
  elas = (double *)_mm_malloc(sizeof(double) * NX * NY * NZ, 256);
  ac = (struct Allen_Cahn *)_mm_malloc(nac * sizeof(struct Allen_Cahn), 256);
  for (n = 0; n < nac; n++)
  {
    ac[n].field2b = (Stype *)_mm_malloc(sizeof(Stype) * nx * ny * nz, 256);
    ac[n].fieldE = (double *)_mm_malloc(sizeof(double) * nx * ny * nz, 256);
    ac[n].fieldEs_left = (double *)_mm_malloc(sizeof(double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEr_left = (double *)_mm_malloc(sizeof(double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEs_right = (double *)_mm_malloc(sizeof(double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEr_right = (double *)_mm_malloc(sizeof(double) * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEs_top = (double *)_mm_malloc(sizeof(double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEr_top = (double *)_mm_malloc(sizeof(double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEs_bottom = (double *)_mm_malloc(sizeof(double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEr_bottom = (double *)_mm_malloc(sizeof(double) * nx * (nghost + 2) * (nz + (nghost + 2) * 2), 256);
    ac[n].fieldEr_front = (double *)_mm_malloc(sizeof(double) * nx * ny * (nghost + 2), 256);
    ac[n].fieldEr_back = (double *)_mm_malloc(sizeof(double) * nx * ny * (nghost + 2), 256);
    ac[n].fieldEs_front = (double *)_mm_malloc(sizeof(double) * nx * ny * (nghost + 2), 256);
    ac[n].fieldEs_back = (double *)_mm_malloc(sizeof(double) * nx * ny * (nghost + 2), 256);
    ac[n].ireq_left_right_fieldE = (MPI_Request *)calloc(4, sizeof(MPI_Request));
    ac[n].ireq_top_bottom_fieldE = (MPI_Request *)calloc(4, sizeof(MPI_Request));
    ac[n].ireq_front_back_fieldE = (MPI_Request *)calloc(4, sizeof(MPI_Request));
  }
  theta = (double *)_mm_malloc(sizeof(double) * ORI * DIM, 256);
  lambdar1 = (double *)_mm_malloc(sizeof(double) * nac * DIM * DIM, 256);
  lambdar = (double *)_mm_malloc(sizeof(double) * DIM * DIM, 256);  

  MPX = (double *)_mm_malloc(sizeof(double) * nx * nx, 256);
  MPY = (double *)_mm_malloc(sizeof(double) * ny * ny, 256);
  MPZ = (double *)_mm_malloc(sizeof(double) * nz * nz, 256);
  MPXI = (double *)_mm_malloc(sizeof(double) * nx * nx, 256);
  MPYI = (double *)_mm_malloc(sizeof(double) * ny * ny, 256);
  MPZI = (double *)_mm_malloc(sizeof(double) * nz * nz, 256);
  DDX = (double *)_mm_malloc(sizeof(double) * nx, 256);
  DDY = (double *)_mm_malloc(sizeof(double) * ny, 256);
  DDZ = (double *)_mm_malloc(sizeof(double) * nz, 256);
  MPX_b = (double *)_mm_malloc(sizeof(double) * nx * nx * 4, 256);
  MPY_b = (double *)_mm_malloc(sizeof(double) * ny * ny * 4, 256);
  MPZ_b = (double *)_mm_malloc(sizeof(double) * nz * nz * 4, 256);
  MPXI_b = (double *)_mm_malloc(sizeof(double) * nx * nx * 4, 256);
  MPYI_b = (double *)_mm_malloc(sizeof(double) * ny * ny * 4, 256);
  MPZI_b = (double *)_mm_malloc(sizeof(double) * nz * nz * 4, 256);
  DDX_b = (double *)_mm_malloc(sizeof(double) * nx * 4, 256);
  DDY_b = (double *)_mm_malloc(sizeof(double) * ny * 4, 256);
  DDZ_b = (double *)_mm_malloc(sizeof(double) * nz * 4, 256);

  status = (MPI_Status *)calloc(4, sizeof(MPI_Status));
  rocblas_create_handle(&handle);
  hipMalloc(&fieldEr, (nx + 2 * 2) * (ny + 2 * 2) * (nz + 2 * 2) * sizeof(Dtype));
  // Malloc F1
  hipMalloc(&f1, nx * ny * nz * sizeof(Dtype));
  hipMalloc(&fieldE, nac * nx * ny * nz * sizeof(Dtype));

  hipMalloc(&field2B, nac * nx * ny * nz * sizeof(Stype));
  hipMalloc(&fielde, nac * nx * ny * nz * sizeof(Dtype));
  hipMalloc(&fieldEu_left, nac * ny * nz * sizeof(Dtype));
  hipMalloc(&fieldEu_right, nac * ny * nz * sizeof(Dtype));
  hipMalloc(&fieldEu_front, nac * nx * ny * sizeof(Dtype));
  hipMalloc(&fieldEu_back, nac * nx * ny * sizeof(Dtype));
  hipMalloc(&fieldEu_top, nac * nx * nz * sizeof(Dtype));
  hipMalloc(&fieldEu_bottom, nac * nx * nz * sizeof(Dtype));
  // Malloc F2
  hipMalloc(&f2, nx * ny * nz * sizeof(Dtype));
  hipMalloc(&lambda, nac * DIM * DIM * sizeof(Dtype));

  // Malloc dgemm
  hipMalloc(&mpxi, nx * nx * sizeof(Dtype));
  hipMalloc(&mpyi, ny * ny * sizeof(Dtype));
  hipMalloc(&mpzi, nz * nz * sizeof(Dtype));
  hipMalloc(&mpx, nx * nx * sizeof(Dtype));
  hipMalloc(&mpy, ny * ny * sizeof(Dtype));
  hipMalloc(&mpz, nz * nz * sizeof(Dtype));
  // Malloc update
  hipMalloc(&ddx, nx * sizeof(Dtype));
  hipMalloc(&ddy, ny * sizeof(Dtype));
  hipMalloc(&ddz, nz * sizeof(Dtype));
  hipMalloc(&fieldEr_left, nac * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2) * sizeof(Dtype));
  hipMalloc(&fieldEs_left, (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2) * sizeof(Dtype));
  hipMalloc(&fieldEr_right, nac * (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2) * sizeof(Dtype));
  hipMalloc(&fieldEs_right, (nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2) * sizeof(Dtype));
  hipMalloc(&fieldEr_top, nac * nx * (nghost + 2) * (nz + (nghost + 2) * 2) * sizeof(Dtype));
  hipMalloc(&fieldEs_top, nx * (nghost + 2) * (nz + (nghost + 2) * 2) * sizeof(Dtype));
  hipMalloc(&fieldEr_bottom, nac * nx * (nghost + 2) * (nz + (nghost + 2) * 2) * sizeof(Dtype));
  hipMalloc(&fieldEs_bottom, nx * (nghost + 2) * (nz + (nghost + 2) * 2) * sizeof(Dtype));
  hipMalloc(&fieldEr_front, nac * nx * ny * (nghost + 2) * sizeof(Dtype));
  hipMalloc(&fieldEr_back, nac * nx * ny * (nghost + 2) * sizeof(Dtype));
  hipMalloc(&fieldEe_front, (nx + 4) * (ny + 4) * (nghost + 2) * sizeof(Dtype));
  hipMalloc(&fieldEe_back, (nx + 4) * (ny + 4) * (nghost + 2) * sizeof(Dtype));
  hipMalloc(&fieldEe_top, (nx + 4) * (nghost + 2) * (nz + 4) * sizeof(Dtype));
  hipMalloc(&fieldEe_bottom, (nx + 4) * (nghost + 2) * (nz + 4) * sizeof(Dtype));
  hipMalloc(&fieldEe_left, (nghost + 2) * (ny + 4) * (nz + 4) * sizeof(Dtype));
  hipMalloc(&fieldEe_right, (nghost + 2) * (ny + 4) * (nz + 4) * sizeof(Dtype));
  hipEventCreate(&st);
  hipEventCreate(&ed);
  hipEventCreate(&st2);
  hipEventCreate(&ed2);
}
void read_matrices(void)
{
  int n;
  char filename[1024], dirname[1024], wirname[1024];
  int i, j, k, l, id, mp_ofst, d_ofst;
  FILE *file;

  if (cart_id[0] == 0 and cart_id[1] == 0)
  {
    sprintf(dirname, "%s%02d", data_dir, cart_id[2]);
    if (mkdir(dirname, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH) < 0)
    {
      // printf("mkdir failed\n");
      // return 2;
    }
  }
  if (cart_id[0] == 0 and cart_id[1] == 0)
  {
    sprintf(wirname, "%s%02d", work_dir, cart_id[2]);
    if (mkdir(wirname, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH) < 0)
    {
      // printf("mkdir failed\n");
      // return 2;
    }
  }

  if (cart_id[1] == 0 && cart_id[2] == 0)
  {
    for (k = 0; k < 4; k++)
    {
      mp_ofst = k * nx * nx;
      d_ofst = k * nx;

      sprintf(filename, "%sd%d%d.dat", "input_data/", k, nx);
      file = fopen(filename, "r");
      for (l = 0; l < nx; l++)
      {
        fscanf(file, "%lf", DDX_b + d_ofst + l);
        DDX_b[d_ofst + l] = DDX_b[d_ofst + l];
      }
      fclose(file);

      sprintf(filename, "%sv%d%d.dat", "input_data/", k, nx);
      file = fopen(filename, "r");
      for (j = 0; j < nx; j++)
      {
        for (i = 0; i < nx; i++)
        {
          l = i * nx + j;
          fscanf(file, "%lf", MPX_b + mp_ofst + l);
        }
      }
      fclose(file);

      sprintf(filename, "%svi%d%d.dat", "input_data/", k, nx);
      file = fopen(filename, "r");
      for (j = 0; j < nx; j++)
      {
        for (i = 0; i < nx; i++)
        {
          l = i * nx + j;
          fscanf(file, "%lf", MPXI_b + mp_ofst + l);
        }
      }
      fclose(file);

      mp_ofst = k * ny * ny;
      d_ofst = k * ny;

      sprintf(filename, "%sd%d%d.dat", "input_data/", k, ny);
      file = fopen(filename, "r");
      for (l = 0; l < ny; l++)
      {
        fscanf(file, "%lf", DDY_b + d_ofst + l);
      }
      fclose(file);

      sprintf(filename, "%sv%d%d.dat", "input_data/", k, ny);
      file = fopen(filename, "r");
      for (j = 0; j < ny; j++)
      {
        for (i = 0; i < ny; i++)
        {
          l = i * ny + j;
          fscanf(file, "%lf", MPY_b + mp_ofst + l);
        }
      }
      fclose(file);

      sprintf(filename, "%svi%d%d.dat", "input_data/", k, ny);
      file = fopen(filename, "r");
      for (j = 0; j < ny; j++)
      {
        for (i = 0; i < ny; i++)
        {
          l = i * ny + j;
          fscanf(file, "%lf", MPYI_b + mp_ofst + l);
        }
      }
      fclose(file);

      mp_ofst = k * nz * nz;
      d_ofst = k * nz;

      sprintf(filename, "%sd%d%d.dat", "input_data/", k, nz);
      file = fopen(filename, "r");
      for (l = 0; l < nz; l++)
      {
        fscanf(file, "%lf", DDZ_b + d_ofst + l);
      }
      fclose(file);

      sprintf(filename, "%sv%d%d.dat", "input_data/", k, nz);
      file = fopen(filename, "r");
      for (j = 0; j < nz; j++)
      {
        for (i = 0; i < nz; i++)
        {
          l = i * nz + j;
          fscanf(file, "%lf", MPZ_b + mp_ofst + l);
        }
      }
      fclose(file);

      sprintf(filename, "%svi%d%d.dat", "input_data/", k, nz);
      file = fopen(filename, "r");
      for (j = 0; j < nz; j++)
      {
        for (i = 0; i < nz; i++)
        {
          l = i * nz + j;
          fscanf(file, "%lf", MPZI_b + mp_ofst + l);
        }
      }
      fclose(file);
    }
  }
  MPI_Bcast(&DDX_b[0], nx * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast(&MPX_b[0], nx * nx * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast(&MPXI_b[0], nx * nx * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast(&DDY_b[0], ny * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast(&MPY_b[0], ny * ny * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast(&MPYI_b[0], ny * ny * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast(&DDZ_b[0], nz * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast(&MPZ_b[0], nz * nz * 4, MPI_DOUBLE, 0, YZ_COMM);
  MPI_Bcast(&MPZI_b[0], nz * nz * 4, MPI_DOUBLE, 0, YZ_COMM);
  if (left < 0)
  {
    k = 1;
  }
  else
  {
    k = 2;
  }

  if (right < 0)
  {
    l = 1;
  }
  else
  {
    l = 2;
  }

  id = (k - 1) + (l - 1) * 2;
  mp_ofst = id * nx * nx;
  d_ofst = id * nx;

  for (l = 0; l < nx; l++)
  {
    DDX[l] = DDX_b[d_ofst + l] / hx / hx;
  }
  for (l = 0; l < nx * nx; l++)
  {
    MPX[l] = MPX_b[mp_ofst + l];
    MPXI[l] = MPXI_b[mp_ofst + l];
  }

  if (top < 0)
  {
    k = 1;
  }
  else
  {
    k = 2;
  }

  if (bottom < 0)
  {
    l = 1;
  }
  else
  {
    l = 2;
  }

  id = (k - 1) + (l - 1) * 2;
  mp_ofst = id * ny * ny;
  d_ofst = id * ny;

  for (l = 0; l < ny; l++)
  {
    DDY[l] = DDY_b[d_ofst + l] / hy / hy;
  }
  for (l = 0; l < ny * ny; l++)
  {
    MPY[l] = MPY_b[mp_ofst + l];
    MPYI[l] = MPYI_b[mp_ofst + l];
  }

  if (front < 0)
  {
    k = 1;
  }
  else
  {
    k = 2;
  }

  if (back < 0)
  {
    l = 1;
  }
  else
  {
    l = 2;
  }

  id = (k - 1) + (l - 1) * 2;
  mp_ofst = id * nz * nz;
  d_ofst = id * nz;

  for (l = 0; l < nz; l++)
  {
    DDZ[l] = DDZ_b[d_ofst + l] / hz / hz;
  }
  for (l = 0; l < nz * nz; l++)
  {
    MPZ[l] = MPZ_b[mp_ofst + l];
    MPZI[l] = MPZI_b[mp_ofst + l];
  }
  hipMemcpy(mpxi, MPXI, nx * nx * sizeof(Dtype), hipMemcpyHostToDevice);
  hipMemcpy(mpyi, MPYI, ny * ny * sizeof(Dtype), hipMemcpyHostToDevice);
  hipMemcpy(mpzi, MPZI, nz * nz * sizeof(Dtype), hipMemcpyHostToDevice);
  hipMemcpy(mpx, MPX, nx * nx * sizeof(Dtype), hipMemcpyHostToDevice);
  hipMemcpy(mpy, MPY, ny * ny * sizeof(Dtype), hipMemcpyHostToDevice);
  hipMemcpy(mpz, MPZ, nz * nz * sizeof(Dtype), hipMemcpyHostToDevice);

  hipMemcpy(ddx, DDX, nx * sizeof(Dtype), hipMemcpyHostToDevice);
  hipMemcpy(ddy, DDY, ny * sizeof(Dtype), hipMemcpyHostToDevice);
  hipMemcpy(ddz, DDZ, nz * sizeof(Dtype), hipMemcpyHostToDevice);
}

void dealloc_vars(void)
{
  int l;
  int n;
  _mm_free(elas);
  for (n = 0; n < nac; n++)
  {
    _mm_free(ac[n].field2b);
    _mm_free(ac[n].fieldE);
    _mm_free(ac[n].fieldEs_left);
    _mm_free(ac[n].fieldEr_left);
    _mm_free(ac[n].fieldEs_right);
    _mm_free(ac[n].fieldEr_right);
    _mm_free(ac[n].fieldEs_top);
    _mm_free(ac[n].fieldEr_top);
    _mm_free(ac[n].fieldEs_bottom);
    _mm_free(ac[n].fieldEr_bottom);
    _mm_free(ac[n].fieldEr_front);
    _mm_free(ac[n].fieldEr_back);
    _mm_free(ac[n].fieldEs_front);
    _mm_free(ac[n].fieldEs_back);
    for (l = 0; l < 4; l++)
    {
      MPI_Request_free(&ac[n].ireq_left_right_fieldE[l]);
      MPI_Request_free(&ac[n].ireq_top_bottom_fieldE[l]);
      MPI_Request_free(&ac[n].ireq_front_back_fieldE[l]);
    }
    free(ac[n].ireq_left_right_fieldE);
    free(ac[n].ireq_top_bottom_fieldE);
    free(ac[n].ireq_front_back_fieldE);
  }
  _mm_free(ac);
  _mm_free(theta);
  _mm_free(lambdar);
  _mm_free(lambdar1);

  _mm_free(MPX);
  _mm_free(MPY);
  _mm_free(MPZ);
  _mm_free(MPXI);
  _mm_free(MPYI);
  _mm_free(MPZI);
  _mm_free(DDX);
  _mm_free(DDY);
  _mm_free(DDZ);
  _mm_free(MPX_b);
  _mm_free(MPY_b);
  _mm_free(MPZ_b);
  _mm_free(MPXI_b);
  _mm_free(MPYI_b);
  _mm_free(MPZI_b);
  _mm_free(DDX_b);
  _mm_free(DDY_b);
  _mm_free(DDZ_b);
  _mm_free(ac_fieldE);
  free(status);
  rocblas_destroy_handle(handle);
  hipFree(fieldEr_front);
  hipFree(fieldEr_back);
  hipFree(fieldEr_top);
  hipFree(fieldEs_top);
  hipFree(fieldEr_bottom);
  hipFree(fieldEs_bottom);
  hipFree(fieldEr_left);
  hipFree(fieldEs_left);
  hipFree(fieldEr_right);
  hipFree(fieldEs_right);
  hipFree(fieldEe_front);
  hipFree(fieldEe_back);
  hipFree(fieldEe_top);
  hipFree(fieldEe_bottom);
  hipFree(fieldEe_left);
  hipFree(fieldEe_right);
  hipFree(f1);
  hipFree(fieldEu_left);
  hipFree(fieldEu_right);
  hipFree(fieldEu_top);
  hipFree(fieldEu_bottom);
  hipFree(fieldEu_front);
  hipFree(fieldEu_back);
  hipFree(fieldEr);
  // free F2
  hipFree(f2);
  hipFree(lambda);
  // free dgemm
  hipFree(mpxi);
  hipFree(mpyi);
  hipFree(mpzi);
  // free update
  hipFree(ddx);
  hipFree(ddy);
  hipFree(ddz);
  hipFree(mpx);
  hipFree(mpy);
  hipFree(mpz);
  hipFree(fieldE);
  hipFree(field2B);
  hipEventDestroy(st);
  hipEventDestroy(ed);
  hipEventDestroy(st2);
  hipEventDestroy(ed2);
}

void write_chk(void)
{
  int i, j, k, n;
  char filename[1024];
  FILE *file;
  int x, l;
  //  for(l = 0; l < (nprocs / 2); l++)
  //{
  //  if (myrank / 2 == l){
  for (n = 0; n < nac; n++)
  {
    sprintf(filename, "%s%02d/eta%d_chk%d_%02d%02d%02d.dat", work_dir, cart_id[2], n, chk, cart_id[0], cart_id[1], cart_id[2]);
    file = fopen(filename, "wb");

    hipMemcpy(ac[n].fieldE, fielde + n * nx * ny * nz, sizeof(Dtype) * nx * ny * nz, hipMemcpyDeviceToHost);

    fwrite(&irun, sizeof(int), 1, file);
  // for (i = 0; i < ORI; i++){
  //         for (j = 0; j < 3; j++){
  //                 fwrite (&theta[i*DIM+j], sizeof (Dtype), 1, file);
  //         }
  // }

    for (k = 0; k < nz; k++)
    {
      for (j = 0; j < ny; j++)
      {
        for (i = 0; i < nx; i++)
        {
          fwrite(&ac[n].fieldE[k * nx * ny + j * nx + i], sizeof(Dtype), 1, file);
        }
      }
    }
    fclose(file);
  }
  //}
  //}
  MPI_Barrier(MPI_COMM_WORLD);
}

void read_chk(void)
{
  int i, j, k, n;
  char filename[1024];
  FILE *file;

  for (n = 0; n < nac; n++)
  {
    sprintf(filename, "%s%02d/eta%d_chk%d_%02d%02d%02d.dat", work_dir, cart_id[2], n, chk, cart_id[0], cart_id[1], cart_id[2]);
    file = fopen(filename, "rb");

    fread(&irun, sizeof(int), 1, file);
   //for (i = 0; i < ORI; i++){
   //    for (j = 0; j < 3; j++){
   //        fread (&theta[i*DIM+j], sizeof (Dtype), 1, file);
   //    }
   //}

    for (k = 0; k < nz; k++)
    {
      for (j = 0; j < ny; j++)
      {
        for (i = 0; i < nx; i++)
        {
          fread(&ac[n].fieldE[k * nx * ny + j * nx + i], sizeof(Dtype), 1, file);
        }
      }
    }
    fclose(file);

    hipMemcpy (fieldE + n * nx * ny * nz, ac[n].fieldE, sizeof (Dtype) * nx * ny * nz, hipMemcpyHostToDevice);
  }
}

void write_field2B(int irun)
{
  int i, j, k, n;
  char filename[1024];
  FILE *file;
  if (prank == myrank)
  {
    printf("write_chk %d\n", irun);
  }

  for (n = 0; n < nac; n++)
  {
    hipMemcpy(ac[n].field2b, field2B + n * offset, sizeof(Stype) * offset, hipMemcpyDeviceToHost);

    sprintf(filename, "%s%02d/eta%d_%06d_%02d%02d%02d.dat", data_dir, cart_id[2], n, irun, cart_id[0], cart_id[1], cart_id[2]);
    file = fopen(filename, "wb");
    for (k = iz1; k < iz4; k++)
    {
      for (j = iy1; j < iy4; j++)
      {
        for (i = ix1; i < ix4; i++)
        {
          fwrite(&ac[n].field2b[k * nx * ny + j * nx + i], sizeof(Stype), 1, file);
        }
      }
    }
    fclose(file);
  }
}
