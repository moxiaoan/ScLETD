// test driver on 3d FFT from FFT library
// for benchmarking, timing purposes
// see test3d.cpp for command-line args

// include files

#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include "fft3d_wrap.h"
#include "./fft3d.h"
#include "./ScLETD.h"
#include <time.h>

//#define DEBUG
// intra functions
void error_all(const char *);
void error_one(const char *);
void error_mpi_time(double);
void *smalloc(int64_t);
void sfree(void *ptr);
/*
void fft_test()
{
  fft_forward(ac[0].fieldE, ac[0].fieldE1, ac[0].fieldE2);
  fft_backward(ac[0].fieldE1, ac[0].fieldE2, ac[0].fieldEt);
  fft_check(ac[0].fieldE, ac[0].fieldEt);
}
*/
/* ----------------------------------------------------------------------
   fft setup
------------------------------------------------------------------------- */
void fft_setup()
{

  MPI_Barrier(MPI_COMM_WORLD);
  nfftx = elas_x;
  nffty = elas_y;
  nfftz = elas_z;
  // partition FFT grid across procs, for both input and output
  xlo = (int)(1.0 * cart_id[0] * nfftx);
  xhi = (int)(1.0 * (cart_id[0] + 1) * nfftx) - 1;

  ylo = (int)(1.0 * cart_id[1] * nffty);
  yhi = (int)(1.0 * (cart_id[1] + 1) * nffty) - 1;

  zlo = (int)(1.0 * cart_id[2] * nfftz);
  zhi = (int)(1.0 * (cart_id[2] + 1) * nfftz) - 1;

  nfft = (xhi - xlo + 1) * (yhi - ylo + 1) * (zhi - zlo + 1);
#ifdef DEBUG
  printf("fft:(%d,%d,%d)-(%d,%d,%d)=(%d,%d,%d)\n", xlo, ylo, zlo, xhi, yhi, zhi, nfftx, nffty, nfftz);
#endif

  MPI_Barrier(MPI_COMM_WORLD);

  // create FFT plan
  // set fft precision double
  fft3d_create(MPI_COMM_WORLD, 2, &fft);
  // 0 or 1
  fft3d_set(fft, "remaponly", 0);
  // enum{POINT,ALL2ALL,COMBO};
  fft3d_set(fft, "collective", 2);
  // enum{PENCIL,BRICK};
  fft3d_set(fft, "exchange", 0);
  // enum{ARRAY,POINTER,MEMCPY};
  fft3d_set(fft, "pack", 2);

  int permute = 0;
  int sendsize, recvsize;

  MPI_Barrier(MPI_COMM_WORLD);

  fft3d_setup(fft, nfftx * procs[0], nffty * procs[1], nfftz * procs[2],
              xlo, xhi, ylo, yhi, zlo, zhi,
              xlo, xhi, ylo, yhi, zlo, zhi,
              permute, &fftsize, &sendsize, &recvsize);

  MPI_Barrier(MPI_COMM_WORLD);

  // allocate memory for FFT grid
  int64_t nbytes = ((int64_t)sizeof(FFT_SCALAR)) * 2 * fftsize;
  fft_in = (FFT_SCALAR *)smalloc(nbytes);
  fft_out = (FFT_SCALAR *)smalloc(nbytes);
  if ((nbytes && fft_in == NULL) || (nbytes && fft_out == NULL))
    error_one("Failed malloc for FFT grid");

  MPI_Barrier(MPI_COMM_WORLD);
}

/* ----------------------------------------------------------------------
   fft forward
------------------------------------------------------------------------- */
void fft_forward_A(double *in, double *out_re, double *out_im)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int m;

  double start1, end1;

  start1 = MPI_Wtime();  
  for (m = 0; m < nfftx * nffty * nfftz; m++)
  {
    fft_in[2 * m] = in[m];
    fft_in[2 * m + 1] = 0.0;
  }
  end1 = MPI_Wtime();

//  printf("myrank = %d, fft3d_forward_copyin_time = %9.7f s\n",myrank, end1 - start1);

  start1 = MPI_Wtime();  
  fft3d_compute_prefast_p2p(fft,fft_in,fft_out,1);
  end1 = MPI_Wtime();

//  printf("myrank = %d, fft3d_forward_compute_time = %9.7f s\n",myrank, end1 - start1);

}
void fft_forward_B(double *in, double *out_re, double *out_im)
{
  double start1, end1;

  start1 = MPI_Wtime();  
  fft3d_compute_prefast_a2a_fastmid_p2p(fft,fft_in,fft_out,1);
  end1 = MPI_Wtime();

//  printf("myrank = %d, fft3d_forward_compute_time = %9.7f s\n",myrank, end1 - start1);

}
void fft_forward_C(double *in, double *out_re, double *out_im)
{
  double start1, end1;

  start1 = MPI_Wtime();  
  fft3d_compute_fastmid_a2a_midslow_p2p(fft,fft_in,fft_out,1);
  end1 = MPI_Wtime();

//  printf("myrank = %d, fft3d_forward_compute_time = %9.7f s\n",myrank, end1 - start1);

}
void fft_forward_D(double *in, double *out_re, double *out_im)
{
  int m;

  double start1, end1;

  start1 = MPI_Wtime();  
  fft3d_compute_midslow_a2a_postslow_p2p(fft,fft_in,fft_out,1);
  fft3d_compute_postslow_a2a(fft,fft_in,fft_out,1);
  end1 = MPI_Wtime();

//  printf("myrank = %d, fft3d_forward_compute_time = %9.7f s\n",myrank, end1 - start1);

  start1 = MPI_Wtime();  
  for (m = 0; m < nfftx * nffty * nfftz; m++)
  {
    out_re[m] = fft_out[2 * m];
    out_im[m] = fft_out[2 * m + 1];
  }
  end1 = MPI_Wtime();

//  printf("myrank = %d, fft3d_forward_copyout_time = %9.7f s\n",myrank, end1 - start1);
  MPI_Barrier(MPI_COMM_WORLD);
}

/* ----------------------------------------------------------------------
   fft backward
------------------------------------------------------------------------- */
void fft_backward(double *in_re, double *in_im, double *out)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int m;

  double start1, end1;

  start1 = MPI_Wtime();  
  for (m = 0; m < nfftx * nffty * nfftz; m++)
  {
    fft_in[2 * m] = in_re[m];
    fft_in[2 * m + 1] = in_im[m];
  }
  end1 = MPI_Wtime();

//  printf("myrank = %d, fft3d_backward_copyin_time = %9.7f s\n",myrank, end1 - start1);


  start1 = MPI_Wtime();  
  // perform FFTs
  //fft3d_compute(fft,fft_in,fft_out,-1);
  fft3d_compute_prefast_p2p(fft,fft_in,fft_out,-1);
  fft3d_compute_prefast_a2a_fastmid_p2p(fft,fft_in,fft_out,-1);
  fft3d_compute_fastmid_a2a_midslow_p2p(fft,fft_in,fft_out,-1);
  fft3d_compute_midslow_a2a_postslow_p2p(fft,fft_in,fft_out,-1);
  fft3d_compute_postslow_a2a(fft,fft_in,fft_out,-1);
  end1 = MPI_Wtime();

//  printf("myrank = %d, fft3d_backward_compute_time is: %f s\n",myrank, end1 - start1);


  start1 = MPI_Wtime();  
  // copy out
  for (m = 0; m < nfftx * nffty * nfftz; m++)
  {
    out[m] = fft_out[2 * m];
  }
  end1 = MPI_Wtime();

//  printf("myrank = %d, fft3d_backward_copyout_time = %9.7f s\n",myrank, end1 - start1);
  MPI_Barrier(MPI_COMM_WORLD);
}

void fft_finish()
{
  // deallocate grid and plan
  sfree(fft_in);
  sfree(fft_out);
  fft3d_destroy(fft);
}

/* ----------------------------------------------------------------------
   fft check forward + backward
------------------------------------------------------------------------- */

void fft_check(double *in, double *out)
{
  int m;
  double delta;
  double epsilon = 0.0;
  double epsmax;
  for (m = 0; m < nfftx * nffty * nfftz; m++)
  {
    delta = fabs(out[m] - in[m]);
    if (delta > epsilon)
      epsilon = delta;
  }
  MPI_Allreduce(&epsilon, &epsmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (myrank == 0)
    printf("fft_check:%lf\n", epsmax);
}

/* ----------------------------------------------------------------------
   must be called by all procs in MPI_COMM_WORLD 
   shuts down MPI and exits
------------------------------------------------------------------------- */

void error_all(const char *str)
{
  MPI_Barrier(MPI_COMM_WORLD);

  if (myrank == 0)
    printf("ERROR: %s\n", str);
  MPI_Finalize();
  exit(1);
}

/* ----------------------------------------------------------------------
   called by one proc in MPI_COMM_WORLD
   forces abort of entire MPI_COMM_WORLD if any proc in MPI_COMM_WORLD calls
------------------------------------------------------------------------- */

void error_one(const char *str)
{
  printf("ERROR on proc %d: %s\n", myrank, str);
  MPI_Abort(MPI_COMM_WORLD, 1);
}


void error_mpi_time(double t)
{
  MPI_Barrier(MPI_COMM_WORLD);

  if (myrank == 0)
    printf("the time of fftmpi is : %lf\n", t);
  MPI_Finalize();
  exit(1);
}
/* ----------------------------------------------------------------------
   safe malloc
------------------------------------------------------------------------- */

void *smalloc(int64_t nbytes)
{
  if (nbytes == 0)
    return NULL;

  void *ptr;

  int retval = posix_memalign(&ptr, FFT_MEMALIGN, nbytes);
  if (retval)
    ptr = NULL;

  return ptr;
}

/* ----------------------------------------------------------------------
   safe free
------------------------------------------------------------------------- */

void sfree(void *ptr)
{
  if (ptr == NULL)
    return;
  free(ptr);
  ptr = NULL;
}
