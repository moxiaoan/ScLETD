#include "ScLETD.h"
#include <assert.h>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "rocfft.h"
#include "hipfft.h"
//#include "hip_complex.h"
#include <complex.h>
#include <rocfft-export.h>
#include <rocfft-version.h>
#define SCLETD_DEBUG

__global__ void 
anisotropic_calc_dev_rc (double *f, double *fa, double *lambda, int n)
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

  int f_k = (ny + 2*2) * (nx+2*2);
  int f_j = (nx + 2*2);
  int offset = 2 * f_k + 2 * f_j + 2; 

  // Id of thread in the warp.
  int WARPSIZE_X = hipBlockDim_x;
  int WARPSIZE_Y = hipBlockDim_y;
  int WARPSIZE_Z = hipBlockDim_z;
  int localId_x = hipThreadIdx_x % WARPSIZE_X;
  int localId_y = hipThreadIdx_y % WARPSIZE_Y;
  int localId_z = hipThreadIdx_z % WARPSIZE_Z;

  // The first index of output element computed by this warp.
  int startOfWarp_x = hipBlockDim_x * hipBlockIdx_x + WARPSIZE_X*(hipThreadIdx_x / WARPSIZE_X);
  int startOfWarp_y = hipBlockDim_y * hipBlockIdx_y + WARPSIZE_Y*(hipThreadIdx_y / WARPSIZE_Y);
  int startOfWarp_z = hipBlockDim_z * hipBlockIdx_z + WARPSIZE_Z*(hipThreadIdx_z / WARPSIZE_Z);

  // The Id of the thread in the scope of the grid.
  i = localId_x + startOfWarp_x;
  j = localId_y + startOfWarp_y;
  k = localId_z + startOfWarp_z;
 // int globalId = k * mesh.nx * mesh.ny + j * mesh.nx + i;


//////////////--left--middle--right///////////////////////////////////////////////////
    double rc[19];
    int globalId = k * f_k  + j * f_j + i + offset;
    rc[0] = fa[k * f_k  + j * f_j + i + offset];

   // if (localId_x ==(WARPSIZE_X-1) && globalId - WARPSIZE_X >=0 ) //left
    if (localId_x ==(WARPSIZE_X-1) && globalId - offset >=0 )
    {
            rc[1]  = fa[globalId - WARPSIZE_X];//left
            rc[7]  = fa[globalId - (nx+4) - WARPSIZE_X];//left_top
            rc[9]  = fa[globalId + (nx+4) - WARPSIZE_X];//left_bottom
            rc[11] = fa[globalId - (ny+4)*(nx+4) - WARPSIZE_X];//left_front
			rc[13] = fa[globalId + (ny+4)*(nx+4) - WARPSIZE_X];//left_back
    }
    if (localId_x ==0 && WARPSIZE_X + globalId < (nx+4)*(ny+4)*(nz+4))
    {
            rc[2]  = fa[WARPSIZE_X + globalId];//right
            rc[8]  = fa[globalId - (nx+4) + WARPSIZE_X];//right_top
            rc[10] = fa[globalId + (nx+4) + WARPSIZE_X];//right_bottom
            rc[12] = fa[globalId - (ny+4)*(nx+4) + WARPSIZE_X];//right_front
            rc[14] = fa[globalId + (ny+4)*(nx+4) + WARPSIZE_X];//right_back
    }
    if (localId_y == (WARPSIZE_Y-1) && globalId - offset >=0 )
    {
            rc[3]  = fa[globalId - WARPSIZE_Y * (nx+4)];//top
            rc[15] = fa[globalId - (ny+4)*(nx+4) - WARPSIZE_Y * (nx+4)];//top_front
            rc[16] = fa[globalId + (ny+4)*(nx+4) - WARPSIZE_Y * (nx+4)];//top_back
    }
    if (localId_y == 0 && globalId + WARPSIZE_Y * (nx+4) < (nx+4)*(ny+4)*(nz+4))
    {
            rc[4]  = fa[globalId + WARPSIZE_Y * (nx+4)];//bottom  
            rc[17] = fa[globalId - (ny+4)*(nx+4) + WARPSIZE_Y * (nx+4)]; //bottom_back
            rc[18] = fa[globalId + (ny+4)*(nx+4) + WARPSIZE_Y * (nx+4)]; //bottom_back
    }
    if (localId_z == (WARPSIZE_Z-1) && globalId - offset >= 0)
    {
            rc[5] =  fa[globalId - WARPSIZE_Z*(ny+4)*(nx+4)];//front
    }
    if (localId_z == 0 && globalId + WARPSIZE_Z*(ny+4)*(nx+4) < (nx+4)*(ny+4)*(nz+4))
    {
            rc[6] =  fa[WARPSIZE_Z*(ny+4)*(nx+4) + globalId];//back
    }
	double toShare = rc[0];
    a_middle =  __shfl(toShare, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + localId_y * WARPSIZE_X + localId_x), 64);
    if (localId_x ==(WARPSIZE_X-1)) //left
    {
        toShare = rc[1];
    }
    a_left =  __shfl(toShare, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + localId_y * WARPSIZE_X + localId_x-1 + WARPSIZE_X) % WARPSIZE_X , WARPSIZE_X);
 ////// 
    if (localId_x ==(WARPSIZE_X-1))
    {
        toShare = rc[0];
    }
    if (localId_x == 0)//right 
    {
        toShare = rc[2];
    }
    a_right =  __shfl(toShare, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + localId_y * WARPSIZE_X + localId_x+1 + WARPSIZE_X) % WARPSIZE_X , WARPSIZE_X);
 ////// 
    if (localId_x == 0)//right 
    {
        toShare = rc[0];
    }
    if (localId_y == (WARPSIZE_Y-1)) //top
    {
        toShare = rc[3];
    }
    a_top =  __shfl(toShare, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y-1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y* WARPSIZE_X );
 ////// 
    if (localId_y == (WARPSIZE_Y-1)) //top
    {
	    toShare = rc[0];
    }
    if (localId_y == 0) //bottom  
    {
        toShare = rc[4];
    }
    a_bottom =  __shfl(toShare, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y+1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y* WARPSIZE_X );
 ////// 
    if (localId_y == 0) //bottom  
    {
        toShare = rc[0];
    }
    if (localId_z == (WARPSIZE_Z-1)) //front
    {
        toShare = rc[5];
    }
    a_front = __shfl(toShare, ((localId_z - 1) * WARPSIZE_Y * WARPSIZE_X + localId_y * WARPSIZE_X + localId_x), 64);
 ////// 
    if (localId_z == (WARPSIZE_Z-1)) //front
    {
        toShare = rc[0];
    }
    if (localId_z == 0) //back
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
                if (i == ix4-1)
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
    if (localId_x ==(WARPSIZE_X-1))
    {
        a_top_temp = rc[7];   //left_top
        a_bottom_temp = rc[9];//left_bottom
        a_front_temp = rc[11];//left_front
        a_back_temp = rc[13];//left_back
	}
    a_left_top =  __shfl(a_top_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y) * WARPSIZE_X + localId_x -1 + WARPSIZE_X) % ( WARPSIZE_X), WARPSIZE_X );
    a_left_bottom =  __shfl(a_bottom_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y) * WARPSIZE_X + localId_x -1 + WARPSIZE_X) % ( WARPSIZE_X), WARPSIZE_X );
    a_left_front =  __shfl(a_front_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y) * WARPSIZE_X + localId_x -1 + WARPSIZE_X) % ( WARPSIZE_X), WARPSIZE_X );
    a_left_back =  __shfl(a_back_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y) * WARPSIZE_X + localId_x -1 + WARPSIZE_X) % ( WARPSIZE_X), WARPSIZE_X );

 ////// 
    if (localId_x == (WARPSIZE_X-1))
    {
        a_top_temp = a_top;//left_top
        a_bottom_temp = a_bottom;//left_bottom
        a_front_temp = a_front;//left_front
        a_back_temp = a_back;//left_back
    }
    if (localId_x == 0)
    {
        a_top_temp = rc[8];//right_top
        a_bottom_temp = rc[10];//right_bottom
        a_front_temp = rc[12];//right_front
        a_back_temp = rc[14];//right_back
    }
    a_right_top =  __shfl(a_top_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y) * WARPSIZE_X + localId_x +1 + WARPSIZE_X) % ( WARPSIZE_X), WARPSIZE_X );
    a_right_bottom =  __shfl(a_bottom_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y) * WARPSIZE_X + localId_x +1 + WARPSIZE_X) % ( WARPSIZE_X), WARPSIZE_X );
    a_right_front =  __shfl(a_front_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y) * WARPSIZE_X + localId_x +1 + WARPSIZE_X) % ( WARPSIZE_X), WARPSIZE_X );
    a_right_back =  __shfl(a_back_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y) * WARPSIZE_X + localId_x +1 + WARPSIZE_X) % ( WARPSIZE_X), WARPSIZE_X );
	
 ////// 
    
    a_top_front_temp = a_front;
    a_top_back_temp = a_back;
    if (localId_y == (WARPSIZE_Y-1))
    {
        a_top_front_temp = rc[15];//top_front
        a_top_back_temp = rc[16];//top_back
    }
	a_top_front =  __shfl(a_top_front_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y-1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y* WARPSIZE_X );
    a_top_back =  __shfl(a_top_back_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y-1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y* WARPSIZE_X );
 ////// 
    a_bottom_front_temp = a_front;
    a_bottom_back_temp = a_back;
    if (localId_y == 0)
    {
        a_bottom_front_temp = rc[17];//bottom_front
        a_bottom_back_temp = rc[18];//bottom_back
    }
	a_bottom_front =  __shfl(a_bottom_front_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y+1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y* WARPSIZE_X );
    a_bottom_back =  __shfl(a_bottom_back_temp, ((localId_z) * WARPSIZE_Y * WARPSIZE_X + (localId_y+1) * WARPSIZE_X + localId_x + WARPSIZE_Y * WARPSIZE_X) % (WARPSIZE_Y * WARPSIZE_X), WARPSIZE_Y* WARPSIZE_X );
	
	double tmp = 0.0;
        fxx = (a_right - 2 * a_middle + a_left) / hx / hx;
        fyy = (a_bottom - 2 * a_middle + a_top) / hy / hy;
        fzz = (a_back - 2 * a_middle + a_front) / hz / hz;
        fxy = ((a_right_bottom - a_left_bottom) - (a_right_top - a_left_top)) * 0.25 / hx  / hy;
        fxz = ((a_right_back - a_left_back) - (a_right_front - a_left_front)) * 0.25 / hx  / hz;
        fyz = ((a_bottom_back - a_top_back) - (a_bottom_front - a_top_front)) * 0.25 / hy  / hz;
        tmp += lambda[0*3 + 0] * fxx;
        tmp += lambda[0*3 + 1] * fxy;
        tmp += lambda[0*3 + 2] * fxz;
        tmp += lambda[1*3 + 0] * fxy;
        tmp += lambda[1*3 + 1] * fyy;
        tmp += lambda[1*3 + 2] * fyz;
        tmp += lambda[2*3 + 0] * fxz;
        tmp += lambda[2*3 + 1] * fyz;
        tmp += lambda[2*3 + 2] * fzz;
        f[k * nx * ny + j * nx + i] = tmp;
}

__global__ void
anisotropic_calc_dev (double *f, double *fa, double *lambda, int n)
{
  int i, j, k;
  int n_left, n_right, n_top, n_bottom, n_front, n_back;
  double a_left, a_right, a_top, a_bottom, a_front, a_back, a_middle;
  double a_left_top, a_left_bottom, a_left_front, a_left_back;
  double a_right_top, a_right_bottom, a_right_front, a_right_back;
  double a_top_front, a_top_back, a_bottom_front, a_bottom_back;
  double fxx, fyy, fzz, fxy, fxz, fyz;

  int f_k = (ny + 2*2) * (nx+2*2);
  int f_j = (nx + 2*2);
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
                if (i == ix4-1)
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


        a_middle = fa[k * f_k  + j * f_j + i + offset];

        a_front = fa[n_front * f_k + j * f_j + i + offset];
        a_back = fa[n_back * f_k + j * f_j + i + offset];
        a_top = fa[k * f_k + n_top * f_j + i + offset];
        a_bottom = fa[k * f_k + n_bottom * f_j + i + offset];
        a_right = fa[k * f_k + j * f_j + n_right + offset];
        a_left = fa[k * f_k + j * f_j + n_left + offset];

        a_left_top = fa[k * f_k + n_top * f_j + n_left + offset];
        a_left_bottom = fa[k * f_k +n_bottom * f_j + n_left + offset];
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
        tmp += lambda[0*3 + 0] * fxx;
        tmp += lambda[0*3 + 1] * fxy;
        tmp += lambda[0*3 + 2] * fxz;
        tmp += lambda[1*3 + 0] * fxy;
        tmp += lambda[1*3 + 1] * fyy;
        tmp += lambda[1*3 + 2] * fyz;
        tmp += lambda[2*3 + 0] * fxz;
        tmp += lambda[2*3 + 1] * fyz;
        tmp += lambda[2*3 + 2] * fzz;
        f[k * nx * ny + j * nx + i] = tmp;
}

__global__ void
top_bottom_pack (double *field, double *fields_top, double *fields_bottom, double *fieldr_front, double *fieldr_back)
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

  if (k < k_s) {
    if (front >= 0) {
            fields_top[tb_k * k + tb_j * j + i] = fieldr_front[f_k * k + f_j * j + i + t_ofst];
            fields_bottom[tb_k * k + tb_j * j + i] = fieldr_front[f_k * k + f_j * j + i + b_ofst];

    }
  }
  else if (k >= k_e) {
    if (back >= 0) {
            fields_top[tb_k * k + tb_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * j + i + t_ofst];
            fields_bottom[tb_k * k + tb_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * j + i + b_ofst];

    }
  }
  else {
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

  if (j < j_s) {
    if (top >= 0) {
          fields_left[lr_k * k + lr_j * j + i] = fieldr_top[tb_k * k + tb_j * j + i + l_ofst];
          fields_right[lr_k * k + lr_j * j + i] = fieldr_top[tb_k * k + tb_j * j + i + r_ofst];
    }
  }
  else if (j >= j_e) {
    if (bottom >= 0) {
          fields_left[lr_k * k + lr_j * j + i] = fieldr_bottom[tb_k * k + tb_j * (j - j_e) + i + l_ofst];
          fields_right[lr_k * k + lr_j * j + i] = fieldr_bottom[tb_k * k + tb_j * (j - j_e) + i + r_ofst];
    }
  }
  else {
    if (k < k_s) {
      if (front >= 0) {
            fields_left[lr_k * k + lr_j * j + i] = fieldr_front[f_k * k + f_j * (j - j_s) + i + l_ofst];
            fields_right[lr_k * k + lr_j * j + i] = fieldr_front[f_k * k + f_j * (j - j_s) + i + r_ofst];
      }
    }
    else if (k >= k_e) {
      if (back >= 0) {
            fields_left[lr_k * k + lr_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * (j - j_s) + i + l_ofst];
            fields_right[lr_k * k + lr_j * j + i] = fieldr_back[f_k * (k - k_e) + f_j * (j - j_s) + i + r_ofst];
      }
    }
    else {
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

  if (left >= 0) {
          field[f_k * k + f_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * (j + (nghost + 2)) + i + 2];
  }
  if (right >= 0) {
          field[f_k * k + f_j * j + i + (nx - nghost)] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * (j + (nghost + 2)) + i];
  }

}

__global__ void
top_bottom_unpack(double *field,double *fieldr_top, double *fieldr_bottom)
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

  if (top >= 0) {
          field[f_k * k + f_j * j + i] = fieldr_top[tb_k * (k + (nghost + 2)) + tb_j * (j + 2) + i];
  }
  if (bottom >= 0) {
          field[f_k * k + f_j * (j + ny - nghost) + i] = fieldr_bottom[tb_k * (k + (nghost + 2)) + tb_j * j + i];
  }

}

__global__ void
front_back_unpack (double *field,double *fieldr_front, double *fieldr_back)
{

  int f_j, f_k;
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;


  f_j = nx;
  f_k = nx * ny;

  if (front >= 0) {
          field[f_k * k + f_j * j + i] = fieldr_front[f_k * (k + 2) + f_j * j + i];
  }
  if (back >= 0) {
          field[f_k * (k + nz - nghost) + f_j * j + i] = fieldr_back[f_k * k + f_j * j + i];
  }

}

__global__ void
left_right_enlarge (double *fielde_left, double *fielde_right, double *fieldr_left, double *fieldr_right)
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

  if (left >= 0) {
    if (front >= 0 && k < k_s) {
      if (j >= 2 && j < ny + 2) {
            fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * k + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2) {
            fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * k + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny) {
            fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * k + lr_j * (j + (nghost + 2)) + i];
      }
    }
    if (k >= k_s && k < k_e) {
      if (j >= 2 && j < ny + 2) {
            fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + nghost) + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2) {
            fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + nghost) + lr_j * j + i];
      }
	  if (bottom >= 0 && j >= ny) {
            fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + nghost) + lr_j * (j + (nghost + 2)) + i];
      }
    }
    if (back >= 0 && k >= k_e) {
      if (j >= 2 && j < ny + 2) {
            fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2) {
            fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny) {
            fielde_left[elr_k * k + elr_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * (j + (nghost + 2)) + i];
      }
    }
  }

  if (right >= 0) {
    if (front >= 0 && k < k_s) {
      if (j >= 2 && j < ny + 2) {
            fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * k + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2) {
            fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * k + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny) {
            fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * k + lr_j * (j + (nghost + 2)) + i];
      }
    }
	if (k >= k_s && k < k_e) {
      if (j >= 2 && j < ny + 2) {
            fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + nghost) + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2) {
            fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + nghost) + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny) {
            fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + nghost) + lr_j * (j + (nghost + 2)) + i];
      }
    }
    if (back >= 0 && k >= k_e) {
      if (j >= 2 && j < ny + 2) {
            fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * (j + nghost) + i];
      }
      if (top >= 0 && j < nghost + 2) {
            fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * j + i];
      }
      if (bottom >= 0 && j >= ny) {
            fielde_right[elr_k * k + elr_j * j + i] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * (j + (nghost + 2)) + i];
      }
    }
  }

}

__global__ void
top_bottom_enlarge (double *fielde_top, double *fielde_bottom, double *fieldr_left, double *fieldr_right, double *fieldr_top, double *fieldr_bottom)
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

  if (top >= 0) {
    if (front >= 0 && k < k_s) {
      if (i >= 2 && i < nx + 2) {
            fielde_top[etb_k * k + etb_j * j + i] = fieldr_top[tb_k * k + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_top[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * k + lr_j * j + i];
      }
      if (right >= 0 && i >= nx) {
            fielde_top[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * k + lr_j * j + (i - nx)];
      }
    }
    if (k >= k_s && k < k_e) {
      if (i >= 2 && i < nx + 2) {
            fielde_top[etb_k * k + etb_j * j + i] = fieldr_top[tb_k * (k + nghost) + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_top[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * (k + nghost) + lr_j * j + i];
      }
	  if (right >= 0 && i >= nx) {
            fielde_top[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * (k + nghost) + lr_j * j + (i - nx)];
      }
    }
    if (back >= 0 && k >= k_e) {
      if (i >= 2 && i < nx + 2) {
            fielde_top[etb_k * k + etb_j * j + i] = fieldr_top[tb_k * (k + (nghost + 2)) + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_top[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * j + i];
      }
      if (right >= 0 && i >= nx) {
            fielde_top[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * j + (i - nx)];
      }
    }
  }

  if (bottom >= 0) {
    if (front >= 0 && k < k_s) {
      if (i >= 2 && i < nx + 2) {
            fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_bottom[tb_k * k + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * k + lr_j * (j + (ny + (nghost + 2))) + i];
      }
      if (right >= 0 && i >= nx) {
            fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * k + lr_j * (j + (ny + (nghost + 2))) + (i - nx)];
      }
    }
	if (k >= k_s && k < k_e) {
      if (i >= 2 && i < nx + 2) {
            fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_bottom[tb_k * (k + nghost) + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * (k + nghost) + lr_j * (j + (ny + (nghost + 2))) + i];
      }
      if (right >= 0 && i >= nx) {
            fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * (k + nghost) + lr_j * (j + (ny + (nghost + 2))) + (i - nx)];      }
    }
    if (back >= 0 && k >= k_e) {
      if (i >= 2 && i < nx + 2) {
            fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_bottom[tb_k * (k + (nghost + 2)) + tb_j * j + (i - 2)];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_left[lr_k * (k + (nghost + 2)) + lr_j * (j + (ny + (nghost + 2))) + i];
      }
      if (right >= 0 && i >= nx) {
            fielde_bottom[etb_k * k + etb_j * j + i] = fieldr_right[lr_k * (k + (nghost + 2)) + lr_j * (j + (ny + (nghost + 2))) + (i - nx)];
      }
    }
  }
}

__global__ void
front_back_enlarge (double *fielde_front, double *fielde_back, double *fieldr_left, double *fieldr_right, double *fieldr_top, double *fieldr_bottom, double *fieldr_front, double *fieldr_back)
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

  if (front >= 0) {
    if (top >= 0 && j < j_s) {
      if (i >= 2 && i < nx + 2) {
            fielde_front[efb_k * k + efb_j * j + i] = fieldr_top[tb_k * k + tb_j * j + i - 2];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_front[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * k + lr_j * j + i];
      }
      if (right >= 0 && i >= nx) {
            fielde_front[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * k + lr_j * j + i - nx];
      }
    }
    if (j >= j_s && j < j_e) {
      if (i >= 2 && i < nx + 2) {
            fielde_front[efb_k * k + efb_j * j + i] = fieldr_front[fb_k * k + fb_j * (j - nghost) + i - 2];
      }
	  if (left >= 0 && i < nghost + 2) {
            fielde_front[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * k + lr_j * (j + nghost) + i];
      }
      if (right >= 0 && i >= nx) {
            fielde_front[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * k + lr_j * (j + nghost) + i - nx];
      }
    }
    if (bottom >= 0 && j >= j_e) {
      if (i >= 2 && i < nx + 2) {
            fielde_front[efb_k * k + efb_j * j + i] = fieldr_bottom[tb_k * k + tb_j * (j - j_e) + i - 2];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_front[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * k + lr_j * (j + (nghost + 2)) + i];
      }
      if (right >= 0 && i >= nx) {
            fielde_front[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * k + lr_j * (j + (nghost + 2)) + i - nx];
      }
    }
  }
  if (back >= 0) {
    if (top >= 0 && j < j_s) {
      if (i >= 2 && i < nx + 2) {
            fielde_back[efb_k * k + efb_j * j + i] = fieldr_top[tb_k * (k + b_ofst) + tb_j * j + i - 2];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_back[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * (k + b_ofst) + lr_j * j + i];
      }
      if (right >= 0 && i >= nx) {
            fielde_back[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * (k + b_ofst) + lr_j * j + i - nx];
      }
    }
    if (j >= j_s && j < j_e) {
      if (i >= 2 && i < nx + 2) {
            fielde_back[efb_k * k + efb_j * j + i] = fieldr_back[fb_k * k + fb_j * (j - nghost) + i - 2];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_back[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * (k + b_ofst) + lr_j * (j + nghost) + i];
      }
      if (right >= 0 && i >= nx) {
            fielde_back[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * (k + b_ofst) + lr_j * (j + nghost) + i - nx];
      }
    }
    if (bottom >= 0 && j >= j_e) {
      if (i >= 2 && i < nx + 2) {
            fielde_back[efb_k * k + efb_j * j + i] = fieldr_bottom[tb_k * (k + b_ofst) + tb_j * (j - j_e) + i - 2];
      }
      if (left >= 0 && i < nghost + 2) {
            fielde_back[efb_k * k + efb_j * j + i] = fieldr_left[lr_k * (k + b_ofst) + lr_j * (j + (nghost + 2)) + i];
      }
	  if (right >= 0 && i >= nx) {
            fielde_back[efb_k * k + efb_j * j + i] = fieldr_right[lr_k * (k + b_ofst) + lr_j * (j + (nghost + 2)) + i - nx];
      }
    }
  }

}

__global__ void
left_right_mu (double *Eu_left, double *Eu_right, double *Ee_left, double *Ee_right)
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

  if (left >= 0) {

    if (k == 2 && front < 0) {
      l_e = k * e_k + j * e_j + 1;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_left[l_mu] = Ee_left[l_e];
    }
    else if (k == nz + 1 && back < 0) {
      l_e = k * e_k + j * e_j + 1;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_left[l_mu] = Ee_left[l_e];
    }
	else {
      l_e = k * e_k + j * e_j + 1;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_left[l_mu] = Ee_left[l_e];
    }
  }

  if (right >= 0) {

    if (k == 2 && front < 0) {
      l_e = k * e_k + j * e_j + 2;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_right[l_mu] = Ee_right[l_e];
    }
    else if (k == nz + 1 && back < 0) {
      l_e = k * e_k + j * e_j + 2;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_right[l_mu] = Ee_right[l_e];
    }
    else {
      l_e = k * e_k + j * e_j + 2;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_right[l_mu] = Ee_right[l_e];
    }
  }

}

__global__ void
top_bottom_mu (double *Eu_top, double *Eu_bottom, double *Ee_top, double *Ee_bottom)
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

  if (top >= 0) {

    if (k == 2 && front < 0) {
      l_e = k * e_k + e_j + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_top[l_mu] = Ee_top[l_e];
    }
    else if (k == nz + 1 && back < 0) {
      l_e = k * e_k + e_j + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_top[l_mu] = Ee_top[l_e];
    }
	else {
      l_e = k * e_k + e_j + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_top[l_mu] = Ee_top[l_e];
    }
  }

  if (bottom >= 0) {

    if (k == 2 && front < 0) {
      l_e = k * e_k + e_j * nghost + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_bottom[l_mu] = Ee_bottom[l_e];
    }
    else if (k == nz + 1 && back < 0) {
      l_e = k * e_k + e_j * nghost + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_bottom[l_mu] = Ee_bottom[l_e];
    }
    else {
      l_e = k * e_k + e_j * nghost + j;
      l_mu = (k - 2) * mu_k + j - 2;
      Eu_bottom[l_mu] = Ee_bottom[l_e];
    }
  }

}

__global__ void
front_back_mu (Dtype *Eu_front, Dtype *Eu_back, Dtype *Ee_front, Dtype *Ee_back)
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

  if (front >= 0) {

    l_e = k * e_k + e_j + j;
    l_mu = (k - 2) * mu_k + j - 2;
    Eu_front[l_mu] = Ee_front[l_e];
  }

  if (back >= 0) {

    l_e = k * e_k + e_j * nghost + j;
    l_mu = (k - 2) * mu_k + j - 2;
    Eu_back[l_mu] = Ee_back[l_e];
  }

}

//tsb
__global__ void
unpack_lr (double *fieldr, double *fieldr_left, double *fieldr_right)
{
  int f_j, f_k;
  int lr_k, lr_j;
  int i, j, k;

  f_j = nx + 2 * 2;
  f_k = (nx+ 2 * 2) * (ny+ 2 * 2);
  lr_j = nghost + 2;
  lr_k = (nghost + 2) * (ny + (nghost + 2) * 2);

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; //  nz+2*2 , ny+2*2 , nghost+2 (k,j,i) 
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  if (left >= 0) {
        fieldr[f_k * k + f_j * j + i] = fieldr_left[lr_k * (k+2) + lr_j * (j+2) + i];
  }
  if (right >= 0) {
        fieldr[f_k * k + f_j * j + i + nx] = fieldr_right[lr_k * (k+2) + lr_j * (j+2) + i];
  }
}

__global__ void
unpack_tb (double *fieldr, double *fieldr_top, double *fieldr_bottom)
{
  int f_j, f_k;
  int tb_k, tb_j;
  int i, j, k;
  f_j = nx + 2 * 2;
  f_k = (nx+ 2 * 2) * (ny+ 2 * 2);
  tb_j = nx;
  tb_k = nx * (nghost + 2);
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; //  nz+2*2 ,  nghost+2, nx (k,j,i) 
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  if (top >= 0) {
        fieldr[f_k * k + f_j * j + i+2] = fieldr_top[tb_k * (k+2) + tb_j * j + i];
  }
  if (bottom >= 0) {
        fieldr[f_k * k + f_j * (j + ny) + i+2] = fieldr_bottom[tb_k * (k+2) + tb_j * j + i];
  }
}

__global__ void
unpack_fb (double *fieldr, double *fieldr_front, double *fieldr_back)
{
  int f_j, f_k;
  int fb_j, fb_k;
  int i, j, k;

  f_j = nx + 2 * 2;
  f_k = (nx+ 2 * 2) * (ny+ 2 * 2);
  fb_j = nx;
  fb_k = nx * ny;
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; //  nghost+2, ny, nx (k,j,i) 
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;


  if (front >= 0) {
        fieldr[f_k * k + f_j * (j+2) + (i+2)] = fieldr_front[fb_k * k + fb_j * j + i];
  }

  if (back >= 0) {
        fieldr[f_k * (k + nz) + f_j * (j+2) + (i+2)] = fieldr_back[fb_k * k + fb_j * j + i];
  }
}

__global__ void
unpack_all (double *fieldr, double *field)
{
  int f_j, f_k;
  int i, j, k;
  f_j = nx + 2 * 2;
  f_k = (nx+ 2 * 2) * (ny+ 2 * 2);
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; //  nz, ny, nx (k,j,i) 
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

        fieldr[f_k * (k+2) + f_j * (j+2) + (i+2)] = field[nx*ny * k + nx * j + i];
}
__global__ void
Ee_dev (double *fieldE, double *fielde)
{
  int i, j, k;
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  fieldE[k*nx*ny + j* nx + i] = fielde[nx*ny * k + nx * j + i];
}

void elastic_transfer(void)
{
  int n = 0;

  int threads_x, threads_y, threads_z;
//  if (ELASTIC == 1) {
    hipMemcpy (s_front, &Elas[n * offset + nx * ny * nghost], sizeof (Dtype) * fb_size, hipMemcpyDeviceToHost);
    hipMemcpy (s_back, &Elas[n * offset + nx * ny * (nz - nghost - (nghost + 2))], sizeof (Dtype) * fb_size, hipMemcpyDeviceToHost);

    MPI_Startall(4, ireq_front_back);
    MPI_Waitall(4, ireq_front_back, status);

    hipMemcpy (R_front, r_front, sizeof (Dtype) * fb_size, hipMemcpyHostToDevice);
    hipMemcpy (R_back, r_back, sizeof (Dtype) * fb_size, hipMemcpyHostToDevice);

    dim3 blocks_tb_pack (nx / THREADS_PER_BLOCK_X, (nghost + 2) / THREADS_PER_BLOCK_Y, (nz + (nghost + 2) * 2) / THREADS_PER_BLOCK_Z);
    dim3 threads_tb_pack (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (top_bottom_pack, blocks_tb_pack, threads_tb_pack, 0, 0,
                        Elas + n * offset, S_top, S_bottom, R_front, R_back);

    hipMemcpy (s_top, S_top, sizeof (Dtype) * tb_size, hipMemcpyDeviceToHost);
    hipMemcpy (s_bottom, S_bottom, sizeof (Dtype) * tb_size, hipMemcpyDeviceToHost);

    MPI_Startall(4, ireq_top_bottom);
    MPI_Waitall(4, ireq_top_bottom, status);

    hipMemcpy (R_top, r_top, sizeof (Dtype) * tb_size, hipMemcpyHostToDevice);
    hipMemcpy (R_bottom, r_bottom, sizeof (Dtype) * tb_size, hipMemcpyHostToDevice);
        
    dim3 blocks_lr_pack ((nghost + 2) / THREADS_PER_BLOCK_X, (ny + (nghost + 2) * 2) / THREADS_PER_BLOCK_Y, (nz + (nghost + 2) * 2) / THREADS_PER_BLOCK_Z);
    dim3 threads_lr_pack (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (left_right_pack, blocks_lr_pack, threads_lr_pack, 0, 0,
                        Elas + n * offset, S_left, S_right, R_top, R_bottom, R_front, R_back);

    hipMemcpy (s_left, S_left, sizeof (Dtype) * lr_size, hipMemcpyDeviceToHost);
    hipMemcpy (s_right, S_right, sizeof (Dtype) * lr_size, hipMemcpyDeviceToHost);

    MPI_Startall(4, ireq_left_right);
    MPI_Waitall(4, ireq_left_right, status);

    hipMemcpy (R_left, r_left, sizeof (Dtype) * lr_size, hipMemcpyHostToDevice);
    hipMemcpy (R_right, r_right, sizeof (Dtype) * lr_size, hipMemcpyHostToDevice);

    threads_x = nghost;
    dim3 blocks_lr_unpack (nghost / threads_x, ny / THREADS_PER_BLOCK_Y, nz / THREADS_PER_BLOCK_Z);
    dim3 threads_lr_unpack (threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (left_right_unpack, blocks_lr_unpack, threads_lr_unpack, 0, 0,
                        Elas + n * offset, R_left, R_right);
    threads_y = nghost;
    dim3 blocks_tb_unpack (nx / THREADS_PER_BLOCK_X, nghost / threads_y, nz / THREADS_PER_BLOCK_Z);
    dim3 threads_tb_unpack (THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (top_bottom_unpack, blocks_tb_unpack, threads_tb_unpack, 0, 0,
                        Elas + n * offset, R_top, R_bottom);

    threads_z = nghost;
    dim3 blocks_fb_unpack (nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, nghost / threads_z);
    dim3 threads_fb_unpack (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);
    hipLaunchKernelGGL (front_back_unpack, blocks_fb_unpack, threads_fb_unpack, 0, 0,
                      Elas + n * offset, R_front, R_back);
  //}

}

void transfer (void)
{
  int n;

  int threads_x, threads_y, threads_z;

  for (n = 0; n < nac; n++)
  {

#ifdef SCLETD_DEBUG
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    hipMemcpy (ac[n].fieldEs_front, &fieldE[n * offset + nx * ny * nghost], sizeof (Dtype) * fb_size, hipMemcpyDeviceToHost);
    hipMemcpy (ac[n].fieldEs_back, &fieldE[n * offset + nx * ny * (nz - nghost - (nghost + 2))], sizeof (Dtype) * fb_size, hipMemcpyDeviceToHost);

#ifdef SCLETD_DEBUG
   hipEventRecord (ed, NULL);
   hipEventSynchronize (ed);
   hipEventElapsedTime (&timer, st, ed);
   trans_Memcpy_time += timer;
   hipEventRecord (st, NULL);
   hipEventSynchronize (st);
#endif

    MPI_Startall(4, ac[n].ireq_front_back_fieldE);
    MPI_Waitall(4, ac[n].ireq_front_back_fieldE, status);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
	hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_MPI_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    hipMemcpy (fieldEr_front, ac[n].fieldEr_front, sizeof (Dtype) * fb_size, hipMemcpyHostToDevice);
    hipMemcpy (fieldEr_back, ac[n].fieldEr_back, sizeof (Dtype) * fb_size, hipMemcpyHostToDevice);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    dim3 blocks_tb_pack (nx / THREADS_PER_BLOCK_X, (nghost + 2) / THREADS_PER_BLOCK_Y, (nz + (nghost + 2) * 2) / THREADS_PER_BLOCK_Z);
    dim3 threads_tb_pack (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (top_bottom_pack, blocks_tb_pack, threads_tb_pack, 0, 0,
                        fieldE + n * offset, fieldEs_top, fieldEs_bottom, fieldEr_front, fieldEr_back);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_pack_time += timer;
    hipEventRecord (st, NULL);
	hipEventSynchronize (st);
#endif

    hipMemcpy (ac[n].fieldEs_top, fieldEs_top, sizeof (Dtype) * tb_size, hipMemcpyDeviceToHost);
    hipMemcpy (ac[n].fieldEs_bottom, fieldEs_bottom, sizeof (Dtype) * tb_size, hipMemcpyDeviceToHost);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    MPI_Startall(4, ac[n].ireq_top_bottom_fieldE);
    MPI_Waitall(4, ac[n].ireq_top_bottom_fieldE, status);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_MPI_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    hipMemcpy (fieldEr_top, ac[n].fieldEr_top, sizeof (Dtype) * tb_size, hipMemcpyHostToDevice);
    hipMemcpy (fieldEr_bottom, ac[n].fieldEr_bottom, sizeof (Dtype) * tb_size, hipMemcpyHostToDevice);
	
#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    dim3 blocks_lr_pack ((nghost + 2) / THREADS_PER_BLOCK_X, (ny + (nghost + 2) * 2) / THREADS_PER_BLOCK_Y, (nz + (nghost + 2) * 2) / THREADS_PER_BLOCK_Z);
    dim3 threads_lr_pack (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (left_right_pack, blocks_lr_pack, threads_lr_pack, 0, 0,\
                        fieldE + n * offset, fieldEs_left, fieldEs_right, fieldEr_top,\
                        fieldEr_bottom, fieldEr_front, fieldEr_back);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_pack_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    hipMemcpy (ac[n].fieldEs_left, fieldEs_left, sizeof (Dtype) * lr_size, hipMemcpyDeviceToHost);
    hipMemcpy (ac[n].fieldEs_right, fieldEs_right, sizeof (Dtype) * lr_size, hipMemcpyDeviceToHost);
#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    MPI_Startall(4, ac[n].ireq_left_right_fieldE);
    MPI_Waitall(4, ac[n].ireq_left_right_fieldE, status);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_MPI_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    hipMemcpy (fieldEr_left, ac[n].fieldEr_left, sizeof (Dtype) * lr_size, hipMemcpyHostToDevice);
    hipMemcpy (fieldEr_right, ac[n].fieldEr_right, sizeof (Dtype) * lr_size, hipMemcpyHostToDevice);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_Memcpy_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    dim3 blocks_lr_enlarge ((nghost + 2) / THREADS_PER_BLOCK_X, (ny + 4) / THREADS_PER_BLOCK_Y, (nz + 4) / THREADS_PER_BLOCK_Z);
    dim3 threads_lr_enlarge (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (left_right_enlarge, blocks_lr_enlarge, threads_lr_enlarge, 0, 0,\
                        fieldEe_left, fieldEe_right, fieldEr_left, fieldEr_right);


    dim3 blocks_tb_enlarge ((nx + 4) / THREADS_PER_BLOCK_X, (nghost + 2) / THREADS_PER_BLOCK_Y, (nz + 4) / THREADS_PER_BLOCK_Z);
    dim3 threads_tb_enlarge (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (top_bottom_enlarge, blocks_tb_enlarge, threads_tb_enlarge, 0, 0,\
                        fieldEe_top, fieldEe_bottom, fieldEr_left, fieldEr_right, fieldEr_top, fieldEr_bottom);


    dim3 blocks_fb_enlarge ((nx + 4) / THREADS_PER_BLOCK_X, (ny + 4) / THREADS_PER_BLOCK_Y, (nghost + 2) / THREADS_PER_BLOCK_Z);
    dim3 threads_fb_enlarge (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (front_back_enlarge, blocks_fb_enlarge, threads_fb_enlarge, 0, 0,\
                        fieldEe_front, fieldEe_back, fieldEr_left, fieldEr_right,\
                        fieldEr_top, fieldEr_bottom, fieldEr_front, fieldEr_back);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_enlarge_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

    threads_x = 1;
    dim3 blocks_lr_mu (1 / threads_x, ny / THREADS_PER_BLOCK_Y, nz / THREADS_PER_BLOCK_Z);
    dim3 threads_lr_mu (threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (left_right_mu, blocks_lr_mu, threads_lr_mu, 0, 0,\
                        fieldEu_left + n * u_lr, fieldEu_right + n * u_lr, fieldEe_left, fieldEe_right);

    threads_y = 1;
    dim3 blocks_tb_mu (nx / THREADS_PER_BLOCK_X, 1 / threads_y, nz / THREADS_PER_BLOCK_Z);
    dim3 threads_tb_mu (THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (top_bottom_mu, blocks_tb_mu, threads_tb_mu, 0, 0,\
                        fieldEu_top + n * u_tb, fieldEu_bottom + n * u_tb, fieldEe_top, fieldEe_bottom);

    threads_z = 1;
    dim3 blocks_fb_mu (nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, 1 / threads_z);
    dim3 threads_fb_mu (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);
    hipLaunchKernelGGL (front_back_mu, blocks_fb_mu, threads_fb_mu, 0, 0,\
                        fieldEu_front + n * u_fb, fieldEu_back + n * u_fb, fieldEe_front, fieldEe_back);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_mu_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif


    threads_x = nghost;
    dim3 blocks_lr_unpack (nghost / threads_x, ny / THREADS_PER_BLOCK_Y, nz / THREADS_PER_BLOCK_Z);
    dim3 threads_lr_unpack (threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (left_right_unpack, blocks_lr_unpack, threads_lr_unpack, 0, 0,\
                        fieldE + n * offset, fieldEr_left, fieldEr_right);
    threads_y = nghost;
    dim3 blocks_tb_unpack (nx / THREADS_PER_BLOCK_X, nghost / threads_y, nz / THREADS_PER_BLOCK_Z);
    dim3 threads_tb_unpack (THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);
    hipLaunchKernelGGL (top_bottom_unpack, blocks_tb_unpack, threads_tb_unpack, 0, 0,\
                        fieldE + n * offset, fieldEr_top, fieldEr_bottom);

    threads_z = nghost;
    dim3 blocks_fb_unpack (nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, nghost / threads_z);
    dim3 threads_fb_unpack (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);
    hipLaunchKernelGGL (front_back_unpack, blocks_fb_unpack, threads_fb_unpack, 0, 0,\
                        fieldE + n * offset, fieldEr_front, fieldEr_back);

//tsb
     threads_x = nghost;
     dim3 blocks3 ((nghost+2) / threads_x, (ny+2*2) / THREADS_PER_BLOCK_Y, (nz+2*2) / THREADS_PER_BLOCK_Z);
     dim3 threads3 (threads_x, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
     hipLaunchKernelGGL (unpack_lr, blocks3, threads3, 0, 0,\
                        fieldEr + n * offset_Er , fieldEr_left, fieldEr_right);

     threads_y = nghost;
     dim3 blocks4 ((nx) / THREADS_PER_BLOCK_X, (nghost+2) / threads_y, (nz+2*2) / THREADS_PER_BLOCK_Z);
     dim3 threads4 (THREADS_PER_BLOCK_X, threads_y, THREADS_PER_BLOCK_Z);
     hipLaunchKernelGGL (unpack_tb, blocks4, threads4, 0, 0,\
                    fieldEr + n * offset_Er , fieldEr_top, fieldEr_bottom);

     threads_z = nghost;
     dim3 blocks5 ((nx) / THREADS_PER_BLOCK_X, (ny) / THREADS_PER_BLOCK_Y, (nghost+2) / threads_z);
     dim3 threads5 (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, threads_z);
     hipLaunchKernelGGL (unpack_fb, blocks5, threads5, 0, 0,\
                    fieldEr + n * offset_Er , fieldEr_front, fieldEr_back);

     dim3 blocks6 ((nx) / THREADS_PER_BLOCK_X, (ny) / THREADS_PER_BLOCK_Y, (nz) / THREADS_PER_BLOCK_Z);
     dim3 threads6 (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
     hipLaunchKernelGGL (unpack_all, blocks6, threads6, 0, 0,\
                    fieldEr + n * offset_Er , fieldE + n * offset);

#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    trans_unpack_time += timer;
#endif

  }

}

__global__ void
ac_calc_F1_dev (Dtype *fieldE, Dtype *f1, Dtype *fieldEu_left, Dtype *fieldEu_right,\
                Dtype *fieldEu_top, Dtype *fieldEu_bottom, Dtype *fieldEu_front, Dtype *fieldEu_back,\
                Dtype LE, Dtype KE , int n)
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
        u1 = fieldE[k * nx * ny + j * nx + i + offset*1];
        u2 = fieldE[k * nx * ny + j * nx + i + offset*2];
        u3 = fieldE[k * nx * ny + j * nx + i + offset*3];

        if(n==0) un = u0;
        if(n==1) un = u1;
        if(n==2) un = u2;
        if(n==3) un = u3;
        f1[k * nx * ny + j * nx + i] = LE * KE * un;

        double f;
        double sumh22;
        sumh22  = u0 * u0 +  u1 * u1 + u2 * u2 + u3 * u3;
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
        int n, i, j;

        double m, p, q;
    if (checkpoint == 1){
        m = rand() / (double)(RAND_MAX);
        p = rand() / (double)(RAND_MAX);
        q = rand() / (double)(RAND_MAX);


        theta[0] = m * 180;
        theta[1] = p * 180;
        theta[2] = q * 180;
        //theta[0] = 45.0;
        //theta[1] = 45.0;
        //theta[2] = 45.0;
        MPI_Bcast (&theta[0], 3, MPI_DOUBLE, 0, R_COMM);
     }
//      printf("id = %d, %d, %d, %d, theta0 = %lf, theta1 = %lf, theta2 = %lf\n",myrank,cart_id[0],cart_id[1],cart_id[2],theta[0],theta[1],theta[2]);
        for (n = 0; n < nac; n++)
        {

                ac[n].lambdar[0][0] = ac[n].lambda[0][0] * cos(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                    + ac[n].lambda[1][0] * sin(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                    - ac[n].lambda[2][0] * sin(theta[1]*PI/180);
                ac[n].lambdar[0][1] = ac[n].lambda[0][1] * cos(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                    + ac[n].lambda[1][1] * sin(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                    - ac[n].lambda[2][1] * sin(theta[1]*PI/180);
		ac[n].lambdar[0][2] = ac[n].lambda[0][2] * cos(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                    + ac[n].lambda[1][2] * sin(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                    - ac[n].lambda[2][2] * sin(theta[1]*PI/180);
                ac[n].lambdar[1][0] = ac[n].lambda[0][0] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * sin(theta[0]*PI/180) - sin(theta[2]*PI/180 * cos(theta[0]*PI/180)))
                                    + ac[n].lambda[1][0] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) *sin(theta[0]*PI/180) + cos(theta[2]*PI/180) * cos(theta[0]*PI/180))
                                    + ac[n].lambda[2][0] * cos(theta[1]*PI/180) * sin(theta[0]*PI/180);
                ac[n].lambdar[1][1] = ac[n].lambda[0][1] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * sin(theta[0]*PI/180) - sin(theta[2]*PI/180 * cos(theta[0]*PI/180)))
                                    + ac[n].lambda[1][1] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) *sin(theta[0]*PI/180) + cos(theta[2]*PI/180) * cos(theta[0]*PI/180))
                                    + ac[n].lambda[2][1] * cos(theta[1]*PI/180) * sin(theta[0]*PI/180);
                ac[n].lambdar[1][2] = ac[n].lambda[0][2] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * sin(theta[0]*PI/180) - sin(theta[2]*PI/180 * cos(theta[0]*PI/180)))
                                    + ac[n].lambda[1][2] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) *sin(theta[0]*PI/180) + cos(theta[2]*PI/180) * cos(theta[0]*PI/180))
                                    + ac[n].lambda[2][2] * cos(theta[1]*PI/180) * sin(theta[0]*PI/180);
                ac[n].lambdar[2][0] = ac[n].lambda[0][0] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) + sin(theta[2]*PI/180 * sin(theta[0]*PI/180)))
                                    + ac[n].lambda[1][0] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) - cos(theta[2]*PI/180) * sin(theta[0]*PI/180))
                                    + ac[n].lambda[2][0] * cos(theta[1]*PI/180) * cos(theta[0]*PI/180);
                ac[n].lambdar[2][1] = ac[n].lambda[0][1] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) + sin(theta[2]*PI/180 * sin(theta[0]*PI/180)))
                                    + ac[n].lambda[1][1] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) - cos(theta[2]*PI/180) * sin(theta[0]*PI/180))
                                    + ac[n].lambda[2][1] * cos(theta[1]*PI/180) * cos(theta[0]*PI/180);
                ac[n].lambdar[2][2] = ac[n].lambda[0][2] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) + sin(theta[2]*PI/180 * sin(theta[0]*PI/180)))
				                    + ac[n].lambda[1][2] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) - cos(theta[2]*PI/180) * sin(theta[0]*PI/180))
                                    + ac[n].lambda[2][2] * cos(theta[1]*PI/180) * cos(theta[0]*PI/180);
                ac[n].lambdar1[0][0] = ac[n].lambdar[0][0] * cos(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                     + ac[n].lambdar[0][1] * sin(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                     - ac[n].lambdar[0][2] * sin(theta[1]*PI/180);
                ac[n].lambdar1[0][1] = ac[n].lambdar[0][0] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * sin(theta[0]*PI/180) - sin(theta[2]*PI/180 * cos(theta[0]*PI/180)))
                                     + ac[n].lambdar[0][1] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) *sin(theta[0]*PI/180) + cos(theta[2]*PI/180) * cos(theta[0]*PI/180))
                                     + ac[n].lambdar[0][2] * cos(theta[1]*PI/180) * sin(theta[0]*PI/180);
                ac[n].lambdar1[0][2] = ac[n].lambdar[0][0] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) + sin(theta[2]*PI/180 * sin(theta[0]*PI/180)))
                                     + ac[n].lambdar[0][1] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) - cos(theta[2]*PI/180) * sin(theta[0]*PI/180))
                                     + ac[n].lambdar[0][2] * cos(theta[1]*PI/180) * cos(theta[0]*PI/180);
                ac[n].lambdar1[1][0] = ac[n].lambdar[1][0] * cos(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                     + ac[n].lambdar[1][1] * sin(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                     - ac[n].lambdar[1][2] * sin(theta[1]*PI/180);
                ac[n].lambdar1[1][1] = ac[n].lambdar[1][0] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * sin(theta[0]*PI/180) - sin(theta[2]*PI/180 * cos(theta[0]*PI/180)))
                                     + ac[n].lambdar[1][1] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) *sin(theta[0]*PI/180) + cos(theta[2]*PI/180) * cos(theta[0]*PI/180))
                                     + ac[n].lambdar[1][2] * cos(theta[1]*PI/180) * sin(theta[0]*PI/180);
                ac[n].lambdar1[1][2] = ac[n].lambdar[1][0] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) + sin(theta[2]*PI/180 * sin(theta[0]*PI/180)))
                                     + ac[n].lambdar[1][1] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) - cos(theta[2]*PI/180) * sin(theta[0]*PI/180))
                                     + ac[n].lambdar[1][2] * cos(theta[1]*PI/180) * cos(theta[0]*PI/180);
		ac[n].lambdar1[2][0] = ac[n].lambdar[2][0] * cos(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                     + ac[n].lambdar[2][1] * sin(theta[2]*PI/180) * cos(theta[1]*PI/180)
                                     - ac[n].lambdar[2][2] * sin(theta[1]*PI/180);
                ac[n].lambdar1[2][1] = ac[n].lambdar[2][0] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * sin(theta[0]*PI/180) - sin(theta[2]*PI/180 * cos(theta[0]*PI/180)))
                                     + ac[n].lambdar[2][1] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) *sin(theta[0]*PI/180) + cos(theta[2]*PI/180) * cos(theta[0]*PI/180))
                                     + ac[n].lambdar[2][2] * cos(theta[1]*PI/180) * sin(theta[0]*PI/180);
                ac[n].lambdar1[2][2] = ac[n].lambdar[2][0] * (cos(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) + sin(theta[2]*PI/180 * sin(theta[0]*PI/180)))
                                     + ac[n].lambdar[2][1] * (sin(theta[2]*PI/180) * sin(theta[1]*PI/180) * cos(theta[0]*PI/180) - cos(theta[2]*PI/180) * sin(theta[0]*PI/180))
                                     + ac[n].lambdar[2][2] * cos(theta[1]*PI/180) * cos(theta[0]*PI/180);
        }


}

// divergence((D-lambda)gradient(eta))
void anisotropic_input(void)
{
        int i, j, n;
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

/*      fp = fopen("anisotropic_input.txt", "r");
        if (fp == NULL)
        {
                printf("open anisotropic_input.txt Fail!");
                exit(EXIT_FAILURE);
        }
*/
        sprintf(filename, "anisotropic_input.txt");
        fp = fopen(filename, "r");
        for (n = 0; n < nac; n++)
        {
                for (i = 0; i < 3; i++)
                {
						for (j = 0; j < 3; j++)
                        {
                                fscanf(fp, "%lf", &(ac[n].lambda[i][j]));
                        }
                }
        }
        fclose(fp);
		
		if (rotation == 1)
        {
                rotation_matrix();
                for (n = 0; n < nac; n++)
                {
                        for (i = 0; i < 3; i++)
                        {
                                for (j = 0; j < 3; j++)
                                {
                                        ac[n].lambda_check[i][j] = ac[n].lambdar1[i][j];
                                        ac[n].lambda[i][j] = D[i][j] - ac[n].lambdar1[i][j];
                                }
                        }
                }
        }

        else
        {
                for (n = 0; n < nac; n++)
                {
                        for (i = 0; i < 3; i++)
                        {
                                for (j = 0; j < 3; j++)
                                {
                                        ac[n].lambda_check[i][j] = ac[n].lambda[i][j];
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
}

__global__ void
ac_add_F1_dev (Dtype *fieldE1, Dtype *f1, int n)
{
        int i, j, k;
        i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

        fieldE1[k * nx * ny + j * nx + i] = f1[k * nx * ny + j * nx + i];

}

__global__ void
ac_add_F1_F2_dev (Dtype *f1, Dtype *f2, Dtype *Elas, Dtype epn2, Dtype LE, int ANISOTROPIC, int ELASTIC, Dtype ElasticScale, int n)
{
        int i, j, k;
        i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
        //fieldE1[k * nx * ny + j * nx + i] = f1[k * nx * ny + j * nx + i];
        if (ANISOTROPIC == 1)
        {
                f1[k * nx * ny + j * nx + i] -= epn2 * LE * f2[k * nx * ny + j * nx + i];

        }

        if (ELASTIC == 1 && n == 0)
        {
                f1[k * nx * ny + j * nx + i] -= LE * ElasticScale * Elas[k * ny * nx + j * nx + i];
        }

}

__global__ void
ac_add_F1_F2_F3_dev (Dtype *fieldE1, Dtype *f3, int iter, int n)
{
        int i, j, k;
        i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
       if (iter < 200)
       {
        fieldE1[k * nx * ny + j * nx + i] += f3[k * nx * ny + j * nx + i];
       }
}

//xyz
__global__ void
xyz_yzx (Dtype *II, Dtype *JJ)
{
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  JJ[i * ny * nz + k * ny + j] = II[k * nx * ny + j * nx + i];
}

//yzx
__global__ void
yzx_zxy (Dtype *II, Dtype *JJ)
{
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  JJ[j * nz * nx + i * nz + k] = II[i * ny * nz + k * ny + j];
}

//zxy
__global__ void
zxy_xyz (Dtype *II, Dtype *JJ)
{
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  JJ[k * nx * ny + j * nx + i] = II[j * nz * nx + i * nz + k];
}

__global__ void
ac_updateU_new(int n, Dtype *field, Dtype *field1, Dtype *DDX, Dtype *DDY, Dtype *DDZ, Dtype LE, Dtype KE, Dtype epn2, Dtype *phiE)
{
  double tmp, Hijk;
  int i, j, k, l;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  l = k * ny * nx + j * nx + i;
  tmp = kkz * DDZ[k] + kky * DDY[j] + kkx * DDX[i];
  Hijk = -LE * (tmp * epn2 - KE);
  if (fabs(Hijk) < 1.0e-8)
  {
        Hijk = 0.0;
  }
  if (fabs(Hijk) > 1.0e-8)
  {
    tmp = exp(-dt * Hijk);
    phiE[l] = (1.0 - tmp) / Hijk;
  }
  else
  {
    phiE[l] = dt;
  }
  tmp = 1.0 - phiE[l] * Hijk;
  field[l] = tmp * field[l] + phiE[l] * field1[l];
}

extern  void hip_init()
{
        deviceId = myrank % 4;
        //hipGetDevice(&deviceId);
        hipGetDeviceProperties(&props, deviceId);

}
__global__ void
Dtype2B_E (Dtype *E, Stype *F2B)
{
  int i, j, k, l;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  l = k * nx * ny + j * nx + i;
  F2B[l] = (Stype) ((E[l] + 0.1) * 30000);
}

__global__ void
elastic_copyin (Dtype *f, Dtype *fe)
{
  int i, j, k;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f[k * NX * NY + j * NX + i] = fe[(k + nghost) * ny * nx + (j + nghost) * nx + i + nghost];

}

__global__ void
skip_copyin (Dtype *f, Dtype *fe)
{
  int i, j, k;
  int dim = 64;
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  if (i % 4 == 0 && j % 4 == 0 && k % 4 == 0)
  {
        f[(k / 4) * dim * dim + (j / 4) * dim + (i / 4)] = (1.0 / 64.0) *\
        (fe[k * NY * NX + j * NX + i] +\
        fe[k * NY * NX + j * NX + i + 1] +\
        fe[k * NY * NX + j * NX + i + 2] +\
        fe[k * NY * NX + j * NX + i + 3] +\
        fe[k * NY * NX + (j + 1) * NX + i] +\
        fe[k * NY * NX + (j + 1) * NX + i + 1] +\
        fe[k * NY * NX + (j + 1) * NX + i + 2] +\
        fe[k * NY * NX + (j + 1) * NX + i + 3] +\
        fe[k * NY * NX + (j + 2) * NX + i] +\
        fe[k * NY * NX + (j + 2) * NX + i + 1] +\
        fe[k * NY * NX + (j + 2) * NX + i + 2] +\
        fe[k * NY * NX + (j + 2) * NX + i + 3] +\
        fe[k * NY * NX + (j + 3) * NX + i] +\
        fe[k * NY * NX + (j + 3) * NX + i + 1] +\
        fe[k * NY * NX + (j + 3) * NX + i + 2] +\
        fe[k * NY * NX + (j + 3) * NX + i + 3] +\
        fe[(k + 1) * NY * NX + j * NX + i] +\
        fe[(k + 1) * NY * NX + j * NX + i + 1] +\
        fe[(k + 1) * NY * NX + j * NX + i + 2] +\
        fe[(k + 1) * NY * NX + j * NX + i + 3] +\
	fe[(k + 1) * NY * NX + (j + 1) * NX + i] +\
        fe[(k + 1) * NY * NX + (j + 1) * NX + i + 1] +\
        fe[(k + 1) * NY * NX + (j + 1) * NX + i + 2] +\
        fe[(k + 1) * NY * NX + (j + 1) * NX + i + 3] +\
        fe[(k + 1) * NY * NX + (j + 2) * NX + i] +\
        fe[(k + 1) * NY * NX + (j + 2) * NX + i + 1] +\
        fe[(k + 1) * NY * NX + (j + 2) * NX + i + 2] +\
        fe[(k + 1) * NY * NX + (j + 2) * NX + i + 3] +\
        fe[(k + 1) * NY * NX + (j + 3) * NX + i] +\
        fe[(k + 1) * NY * NX + (j + 3) * NX + i + 1] +\
        fe[(k + 1) * NY * NX + (j + 3) * NX + i + 2] +\
        fe[(k + 1) * NY * NX + (j + 3) * NX + i + 3] +\
        fe[(k + 2) * NY * NX + j * NX + i] +\
        fe[(k + 2) * NY * NX + j * NX + i + 1] +\
        fe[(k + 2) * NY * NX + j * NX + i + 2] +\
        fe[(k + 2) * NY * NX + j * NX + i + 3] +\
        fe[(k + 2) * NY * NX + (j + 1) * NX + i] +\
        fe[(k + 2) * NY * NX + (j + 1) * NX + i + 1] +\
        fe[(k + 2) * NY * NX + (j + 1) * NX + i + 2] +\
        fe[(k + 2) * NY * NX + (j + 1) * NX + i + 3] +\
        fe[(k + 2) * NY * NX + (j + 2) * NX + i] +\
        fe[(k + 2) * NY * NX + (j + 2) * NX + i + 1] +\
        fe[(k + 2) * NY * NX + (j + 2) * NX + i + 2] +\
        fe[(k + 2) * NY * NX + (j + 2) * NX + i + 3] +\
        fe[(k + 2) * NY * NX + (j + 3) * NX + i] +\
        fe[(k + 2) * NY * NX + (j + 3) * NX + i + 1] +\
        fe[(k + 2) * NY * NX + (j + 3) * NX + i + 2] +\
        fe[(k + 2) * NY * NX + (j + 3) * NX + i + 3] +\
        fe[(k + 3) * NY * NX + j * NX + i] +\
        fe[(k + 3) * NY * NX + j * NX + i + 1] +\
        fe[(k + 3) * NY * NX + j * NX + i + 2] +\
	fe[(k + 3) * NY * NX + j * NX + i + 3] +\
        fe[(k + 3) * NY * NX + (j + 1) * NX + i] +\
        fe[(k + 3) * NY * NX + (j + 1) * NX + i + 1] +\
        fe[(k + 3) * NY * NX + (j + 1) * NX + i + 2] +\
        fe[(k + 3) * NY * NX + (j + 1) * NX + i + 3] +\
        fe[(k + 3) * NY * NX + (j + 2) * NX + i] +\
        fe[(k + 3) * NY * NX + (j + 2) * NX + i + 1] +\
        fe[(k + 3) * NY * NX + (j + 2) * NX + i + 2] +\
        fe[(k + 3) * NY * NX + (j + 2) * NX + i + 3] +\
        fe[(k + 3) * NY * NX + (j + 3) * NX + i] +\
        fe[(k + 3) * NY * NX + (j + 3) * NX + i + 1] +\
        fe[(k + 3) * NY * NX + (j + 3) * NX + i + 2] +\
        fe[(k + 3) * NY * NX + (j + 3) * NX + i + 3]);

  }
}
__global__ void
skip_copyin_new (Dtype *f, Dtype *fe)
{
  int i, j, k;
  int dim = elas_x;
  int dim1 = elas_z;
  int stride = 4;

  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  f[k * dim * dim + j * dim + i] = (1.0 / 64.0) *\
  (fe[(k * stride + nghost) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 1) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 2) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 1) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 2) * nx + i * stride + nghost + 3] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 1] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 2] +\
  fe[(k * stride + nghost + 3) * ny * nx + (j * stride + nghost + 3) * nx + i * stride + nghost + 3]);

}

__global__ void
trilinear_interpolation (Dtype *elas_bk, Dtype *Bn_stariftre)
{
  int x, y, z;
  int tx, ty, tz;
  int dim = elas_x;
  int dim1 = elas_z;

  x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
  
  if (z == 0)
  {
    if (y == 0)
    {
      if (x == 0)
      {
        for (tz = 1; tz < 3; tz++)
        {
          for (ty = 1; ty < 3; ty++)
          {
            for (tx = 1; tx < 3; tx++)
            {
                 elas_bk[(z + tz - 1 + nghost) * nx * ny + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
                (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                (1.0-0.33*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                (1.0-0.33*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
           }
         }
       }
     }
     if (x == NX - 2)
     {
        for (tz = 1; tz < 3; tz++)
        {
          for (ty = 1; ty < 3; ty++)
          {
            for (tx = 1; tx < 3; tx++)
            {
                 elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
                (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                (1.0-0.33*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                (1.0-0.33*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
           }
         }
       }
     }
     if (x > 0 && x < NX - 2 && x % 4 == 0)
     {
        for (tz = 1; tz < 3; tz++)
        {
          for (ty = 1; ty < 3; ty++)
          {
            for (tx = 1; tx < 5; tx++)
            {
                 elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + ((x/4) * 4 - 3 + tx + nghost)] =\
                (1.0-0.2*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                0.2*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                (1.0-0.2*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                0.2*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                (1.0-0.2*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                0.2*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*ty*0.33*tz*(1.0-0.2*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                0.33*ty*0.33*tz*0.2*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
           }
         }
       }
     }
   }
   if (y == NY - 2)
   {
     if (x == 0)
     {
       for (tz = 1; tz < 3; tz++)
       {
         for (ty = 1; ty < 3; ty++)
         {
           for (tx = 1; tx < 3; tx++)
           {
                elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
               (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               (1.0-0.33*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               (1.0-0.33*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
           }
         }
       }
     }
     if (x == NX - 2)
     {
       for (tz = 1; tz < 3; tz++)
       {
         for (ty = 1; ty < 3; ty++)
         {
           for (tx = 1; tx < 3; tx++)
           {
                elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
               (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               (1.0-0.33*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               (1.0-0.33*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
           }
         }
       }
     }
     if (x > 0 && x < NX - 2 && x % 4 == 0)
     {
       for (tz = 1; tz < 3; tz++)
       {
         for (ty = 1; ty < 3; ty++)
         {
           for (tx = 1; tx < 5; tx++)
           {
                elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + ((x/4) * 4 - 3 + tx + nghost)] =\
               (1.0-0.2*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
               0.2*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               (1.0-0.2*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
               0.2*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               (1.0-0.2*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
               0.2*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*ty*0.33*tz*(1.0-0.2*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
               0.33*ty*0.33*tz*0.2*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
           }
         }
       }
     }
   }
   if (y > 0 && y < NY - 2 && y % 4 == 0)
   {
     if (x == 0)
     {
       for (tz = 1; tz < 3; tz++)
       {
         for (ty = 1; ty < 5; ty++)
         {
           for (tx = 1; tx < 3; tx++)
           {
                elas_bk[(z + tz - 1 + nghost) * ny * nx + ((y/4) * 4 - 3 + ty + nghost) * nx + (x + tx - 1 + nghost)] =\
               (1.0-0.33*tx)*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
               0.33*tx*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
               (1.0-0.33*tx)*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.33*tx*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               (1.0-0.33*tx)*(1.0-0.2*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
               0.33*tx*0.33*tz*(1.0-0.2*ty)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
               0.2*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
               0.2*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
          }
        }
      }
    }
    if (x == NX - 2)
    {
      for (tz = 1; tz < 3; tz++)
      {
        for (ty = 1; ty < 5; ty++)
        {
          for (tx = 1; tx < 3; tx++)
          {
               elas_bk[(z + tz - 1 + nghost) * ny * nx + ((y/4) * 4 - 3 + ty + nghost) * nx + (x + tx - 1 + nghost)] =\
              (1.0-0.33*tx)*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
              0.33*tx*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
              (1.0-0.33*tx)*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
              0.33*tx*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
              (1.0-0.33*tx)*(1.0-0.2*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
              0.33*tx*0.33*tz*(1.0-0.2*ty)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
              0.2*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
              0.2*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
          }
        }
      }
    }
    if (x > 0 && x < NX - 2 && x % 4 == 0)
    {
      for (tz = 1; tz < 3; tz++)
      {
        for (ty = 1; ty < 5; ty++)
        {
          for (tx = 1; tx < 5; tx++)
          {
               elas_bk[(z + tz - 1 + nghost) * ny * nx + ((y/4) * 4 - 3 + ty + nghost) * nx + ((x/4) * 4 - 3 + tx + nghost)] =\
              (1.0-0.2*tx)*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4) - 1] +\
              0.2*tx*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
              (1.0-0.2*tx)*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
              0.2*tx*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
              (1.0-0.2*tx)*(1.0-0.2*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4) - 1] +\
              0.2*tx*0.33*tz*(1.0-0.2*ty)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
              0.2*ty*0.33*tz*(1.0-0.2*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
              0.2*ty*0.33*tz*0.2*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
          }
        }
      }
    }
   }
  }
  if (z == NZ - 2)
  {
    if (y == 0)
    {
      if (x == 0)
      {
        for (tz = 1; tz < 3; tz++)
        {
          for (ty = 1; ty < 3; ty++)
          {
            for (tx = 1; tx < 3; tx++)
            {
                 elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
                (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                (1.0-0.33*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                (1.0-0.33*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                0.33*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
            }
          }
        }
      }
      if (x == NX - 2)
      {
         for (tz = 1; tz < 3; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
       if (x > 0 && x < NX - 2 && x % 4 == 0)
       {
         for (tz = 1; tz < 3; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 5; tx++)
             {
                  elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + ((x/4) * 4 - 3 + tx + nghost)] =\
                 (1.0-0.2*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.2*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.2*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.33*tz*(1.0-0.2*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.33*ty*0.33*tz*0.2*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
     }
     if (y == NY - 2)
     {
       if (x == 0)
       {
         for (tz = 1; tz < 3; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
       if (x == NX - 2)
       {
         for (tz = 1; tz < 3; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
       if (x > 0 && x < NX - 2 && x % 4 == 0)
       {
         for (tz = 1; tz < 3; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 5; tx++)
             {
                  elas_bk[(z + tz - 1 + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + ((x/4) * 4 - 3 + tx + nghost)] =\
                 (1.0-0.2*tx)*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*(1.0-0.33*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.2*tx)*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*0.33*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.2*tx)*(1.0-0.33*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*0.33*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.33*tz*(1.0-0.2*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.33*ty*0.33*tz*0.2*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
     }
     if (y > 0 && y < NY - 2 && y % 4 == 0)
     {
       if (x == 0)
       {
         for (tz = 1; tz < 3; tz++)
         {
           for (ty = 1; ty < 5; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[(z + tz - 1 + nghost) * ny * nx + ((y/4) * 4 - 3 + ty + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*(1.0-0.2*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.33*tx*0.33*tz*(1.0-0.2*ty)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.2*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.2*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
       if (x == NX - 2)
       {
         for (tz = 1; tz < 3; tz++)
         {
           for (ty = 1; ty < 5; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[(z + tz - 1 + nghost) * ny * nx + ((y/4) * 4 - 3 + ty + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*(1.0-0.2*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.33*tx*0.33*tz*(1.0-0.2*ty)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.2*ty*0.33*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.2*ty*0.33*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
       if (x > 0 && x < NX - 2 && x % 4 == 0)
       {
         for (tz = 1; tz < 3; tz++)
         {
           for (ty = 1; ty < 5; ty++)
           {
             for (tx = 1; tx < 5; tx++)
             {
                  elas_bk[(z + tz - 1 + nghost) * ny * nx + ((y/4) * 4 - 3 + ty + nghost) * nx + ((x/4) * 4 - 3 + tx + nghost)] =\
                 (1.0-0.2*tx)*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4) - 1] +\
                 0.2*tx*(1.0-0.2*ty)*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 (1.0-0.2*tx)*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*0.2*ty*(1.0-0.33*tz)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.2*tx)*(1.0-0.2*ty)*0.33*tz*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4) - 1] +\
                 0.2*tx*0.33*tz*(1.0-0.2*ty)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.2*ty*0.33*tz*(1.0-0.2*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*ty*0.33*tz*0.2*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
     }
   }
   if (z > 0 && z < NZ - 2 && z % 4 == 0)
   {
     if (y == 0)
     {
       if (x == 0)
       {
         for (tz = 1; tz < 5; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[((z/4) * 4 - 3 + tz + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*0.2*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.2*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.2*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.2*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
       if (x == NX - 2)
       {
         for (tz = 1; tz < 5; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[((z/4) * 4 - 3 + tz + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*0.2*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.2*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.2*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.2*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }  
       if (x > 0 && x < NX - 2 && x % 4 == 0)
       {
         for (tz = 1; tz < 5; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 5; tx++)
             {
                  elas_bk[((z/4) * 4 - 3 + tz + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + ((x/4) * 4 - 3 + tx + nghost)] =\
                 (1.0-0.2*tx)*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.2*tx)*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.2*tx)*(1.0-0.33*ty)*0.2*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*0.2*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.2*tz*(1.0-0.2*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.33*ty*0.2*tz*0.2*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
     }
     if (y == NY - 2)
     {
       if (x == 0)
       {
         for (tz = 1; tz < 5; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[((z/4) * 4 - 3 + tz + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*0.2*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.2*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.2*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.2*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
       if (x == NX - 2)
       {
         for (tz = 1; tz < 5; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[((z/4) * 4 - 3 + tz + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*(1.0-0.33*ty)*0.2*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.2*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.2*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.2*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
       if (x > 0 && x < NX - 2 && x % 4 == 0)
       {
         for (tz = 1; tz < 5; tz++)
         {
           for (ty = 1; ty < 3; ty++)
           {
             for (tx = 1; tx < 5; tx++)
             {
                  elas_bk[((z/4) * 4 - 3 + tz + nghost) * ny * nx + (y + ty - 1 + nghost) * nx + ((x/4) * 4 - 3 + tx + nghost)] =\
                 (1.0-0.2*tx)*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*(1.0-0.33*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.2*tx)*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*0.33*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.2*tx)*(1.0-0.33*ty)*0.2*tz*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*0.2*tz*(1.0-0.33*ty)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*ty*0.2*tz*(1.0-0.2*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.33*ty*0.2*tz*0.2*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
     }
     if (y > 0 && y < NY - 2 && y % 4 == 0)
     {
       if (x == 0)
       {
         for (tz = 1; tz < 5; tz++)
         {
           for (ty = 1; ty < 5; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[((z/4) * 4 - 3 + tz + nghost) * ny * nx + ((y/4) * 4 - 3 + ty + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.2*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.2*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.2*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.2*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*(1.0-0.2*ty)*0.2*tz*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.33*tx*0.2*tz*(1.0-0.2*ty)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.2*ty*0.2*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.2*ty*0.2*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
       if (x == NX - 2)
       {
         for (tz = 1; tz < 5; tz++)
         {
           for (ty = 1; ty < 5; ty++)
           {
             for (tx = 1; tx < 3; tx++)
             {
                  elas_bk[((z/4) * 4 - 3 + tz + nghost) * ny * nx + ((y/4) * 4 - 3 + ty + nghost) * nx + (x + tx - 1 + nghost)] =\
                 (1.0-0.33*tx)*(1.0-0.2*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.33*tx*(1.0-0.2*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 (1.0-0.33*tx)*0.2*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.33*tx*0.2*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.33*tx)*(1.0-0.2*ty)*0.2*tz*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.33*tx*0.2*tz*(1.0-0.2*ty)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.2*ty*0.2*tz*(1.0-0.33*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)] +\
                 0.2*ty*0.2*tz*0.33*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
       if (x > 0 && x < NX - 2 && x % 4 == 0)
       {
         for (tz = 1; tz < 5; tz++)
         {
           for (ty = 1; ty < 5; ty++)
           {
             for (tx = 1; tx < 5; tx++)
             {
                  elas_bk[((z/4) * 4 - 3 + tz + nghost) * ny * nx + ((y/4) * 4 - 3 + ty + nghost) * nx + ((x/4) * 4 - 3 + tx + nghost)] =\
                 (1.0-0.2*tx)*(1.0-0.2*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + ((y/4)-1) * dim + (x/4) - 1] +\
                 0.2*tx*(1.0-0.2*ty)*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 (1.0-0.2*tx)*0.2*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*tx*0.2*ty*(1.0-0.2*tz)*Bn_stariftre[((z/4)-1) * dim * dim + (y/4) * dim + (x/4)] +\
                 (1.0-0.2*tx)*(1.0-0.2*ty)*0.2*tz*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4) - 1] +\
                 0.2*tx*0.2*tz*(1.0-0.2*ty)*Bn_stariftre[(z/4) * dim * dim + ((y/4)-1) * dim + (x/4)] +\
                 0.2*ty*0.2*tz*(1.0-0.2*tx)*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4) - 1] +\
                 0.2*ty*0.2*tz*0.2*tx*Bn_stariftre[(z/4) * dim * dim + (y/4) * dim + (x/4)];
             }
           }
         }
       }
    }
  }
}



__global__ void
elastic_multiply_BN (Dtype *Bnre, Dtype *Bnim, Dtype *BN, Dtype *ftre, Dtype *ftim)
{
  int i, j, k;
  int dim = elas_x;
  int dim1 = elas_z;
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  if (i % 4 == 0 && j % 4 == 0 && k % 4 == 0)
  {
        Bnre[(k / 4) * dim * dim + (j / 4) * dim + (i / 4)] = BN[k * NX * NY + j * NX + i] * ftre[(k / 4) * dim * dim + (j / 4) * dim + (i / 4)];
        Bnim[(k / 4) * dim * dim + (j / 4) * dim + (i / 4)] = BN[k * NX * NY + j * NX + i] * ftim[(k / 4) * dim * dim + (j / 4) * dim + (i / 4)];
  }
}
/*
__global__ void
elastic_copyout (Dtype *elas, Dtype *elas_bk)
{
  int i, j, k;
  i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  k = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

  elas[(k + nghost) * ny * nx + (j + nghost) * nx + i + nghost] = elas_bk[k * NX * NY + j * NX + i];
}*/

void elastic_calculate(void)
{
  dim3 blocks ((ix4 - ix1) / THREADS_PER_BLOCK_X, (iy4 - iy1) / THREADS_PER_BLOCK_Y, (iz4 - iz1) / THREADS_PER_BLOCK_Z);
  dim3 threads (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  dim3 blocks_ela (NX / THREADS_PER_BLOCK_X, NY / THREADS_PER_BLOCK_Y, NZ / THREADS_PER_BLOCK_Z);
  dim3 threads_ela (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  dim3 blocks_ela1 (elas_x / THREADS_PER_BLOCK_X, elas_y / THREADS_PER_BLOCK_Y, elas_z / THREADS_PER_BLOCK_Z);
  dim3 threads_ela1 (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);

  if (ELASTIC == 1)
  {
    for(int p = 0; p < NELASTIC; p++)
    {

#ifdef SCLETD_DEBUG
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

      hipLaunchKernelGGL (skip_copyin_new, blocks_ela1, threads_ela1, 0, 0, tmpy_RE1, fieldE + p * offset);

#ifdef SCLETD_DEBUG
   hipEventRecord (ed, NULL);
   hipEventSynchronize (ed);
   hipEventElapsedTime (&timer, st, ed);
   skipin_copyin_time += timer;
   hipEventRecord (st, NULL);
   hipEventSynchronize (st);
#endif

      hipMemcpy (tmpy_re, tmpy_RE1, offset3 * sizeof (double), hipMemcpyDeviceToHost);

#ifdef SCLETD_DEBUG
   hipEventRecord (ed, NULL);
   hipEventSynchronize (ed);
   hipEventElapsedTime (&timer, st, ed);
   elastic_Memcpy_time += timer;
#endif

      fft_forward_A(tmpy_re, tmpy_fftre[p], tmpy_fftim[p]);
 
      fft_forward_B(tmpy_re, tmpy_fftre[p], tmpy_fftim[p]);
    }
  }

  for(int n = 0; n < nac; n++)
  {
    rocblas_dgemm (handle, rocblas_operation_transpose, rocblas_operation_transpose, x_n, x_k, x_m, &alpha, fieldE + n * offset, x_k, mpxi, x_m, &beta, f1 + n * offset, x_n);
    rocblas_dgemm (handle, rocblas_operation_transpose, rocblas_operation_transpose, y_n, y_k, y_m, &alpha, f1 + n * offset, y_k, mpyi, y_m, &beta, f2 + n * offset, y_n);
    rocblas_dgemm (handle, rocblas_operation_transpose, rocblas_operation_transpose, z_n, z_k, z_m, &alpha, f2 + n * offset, z_k, mpzi, z_m, &beta, fielde + n * offset, z_n);
  }

  if (ELASTIC == 1)
  {
    for(int p = 0; p < NELASTIC; p++)
    {
      fft_forward_C(tmpy_re, tmpy_fftre[p], tmpy_fftim[p]);
    }
  }

  for(int n = 0; n < nac; n++)
  {
    hipLaunchKernelGGL (ac_calc_F1_dev, blocks, threads, 0, 0,\
                        fieldE, f1 + n * offset, fieldEu_left + n * u_lr, fieldEu_right + n * u_lr,\
                        fieldEu_top + n * u_tb, fieldEu_bottom + n * u_tb, fieldEu_front + n * u_fb, \
                        fieldEu_back + n * u_fb, ac[n].LE, ac[n].KE, n);
    if (ANISOTROPIC == 1)
    {
      hipLaunchKernelGGL (anisotropic_calc_dev_rc, blocks, threads, 0, 0,\
                          f2 + n * offset, fieldEr + n * offset_Er, lambda + n * 3 * 3, n);
    }
  }

  if (ELASTIC == 1)
  {
    for(int p = 0; p < NELASTIC; p++)
    {
      fft_forward_D(tmpy_re, tmpy_fftre[p], tmpy_fftim[p]);

#ifdef SCLETD_DEBUG
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

      hipMemcpy (tmpy_fftRE[p], tmpy_fftre[p], offset3 * sizeof (double), hipMemcpyHostToDevice);
      hipMemcpy (tmpy_fftIM[p], tmpy_fftim[p], offset3 * sizeof (double), hipMemcpyHostToDevice);

#ifdef SCLETD_DEBUG
   hipEventRecord (ed, NULL);
   hipEventSynchronize (ed);
   hipEventElapsedTime (&timer, st, ed);
   elastic_Memcpy_time += timer;
   hipEventRecord (st, NULL);
   hipEventSynchronize (st);
#endif

      for(int q = 0; q < NELASTIC; q++)
      {
         hipLaunchKernelGGL (elastic_multiply_BN, blocks_ela, threads_ela, 0, 0, tmpy_fftRE[p], tmpy_fftIM[p], BN[p][q], tmpy_fftRE[q], tmpy_fftIM[q]);
      }

#ifdef SCLETD_DEBUG
   hipEventRecord (ed, NULL);
   hipEventSynchronize (ed);
   hipEventElapsedTime (&timer, st, ed);
   elastic_multiply_BN_time += timer;
   hipEventRecord (st, NULL);
   hipEventSynchronize (st);
#endif

      hipMemcpy (tmpy_fftre[p], tmpy_fftRE[p], offset3 * sizeof (double), hipMemcpyDeviceToHost);
      hipMemcpy (tmpy_fftim[p], tmpy_fftIM[p], offset3 * sizeof (double), hipMemcpyDeviceToHost);

#ifdef SCLETD_DEBUG
   hipEventRecord (ed, NULL);
   hipEventSynchronize (ed);
   hipEventElapsedTime (&timer, st, ed);
   elastic_Memcpy_time += timer;
   hipEventRecord (st, NULL);
   hipEventSynchronize (st);
#endif

      fft_backward(tmpy_fftre[p], tmpy_fftim[p], tmpy_re);
      
#ifdef SCLETD_DEBUG
   hipEventRecord (ed, NULL);
   hipEventSynchronize (ed);
   hipEventElapsedTime (&timer, st, ed);
   backward_mpi_time += timer;
   hipEventRecord (st, NULL);
   hipEventSynchronize (st);
#endif

      hipMemcpy (tmpy_RE1, tmpy_re, offset3 * sizeof (double), hipMemcpyHostToDevice);
      
#ifdef SCLETD_DEBUG
   hipEventRecord (ed, NULL);
   hipEventSynchronize (ed);
   hipEventElapsedTime (&timer, st, ed);
   elastic_Memcpy_time += timer;
   hipEventRecord (st, NULL);
   hipEventSynchronize (st);
#endif

      hipLaunchKernelGGL (trilinear_interpolation, blocks_ela, threads_ela, 0, 0, Elas + p * offset, tmpy_RE1);

#ifdef SCLETD_DEBUG
   hipEventRecord (ed, NULL);
   hipEventSynchronize (ed);
   hipEventElapsedTime (&timer, st, ed);
   trilinear_interpolation_time += timer;
#endif

   }
  }
}

// f1
void ac_calc_F1 ()
{

        dim3 blocks ((ix4 - ix1) / THREADS_PER_BLOCK_X, (iy4 - iy1) / THREADS_PER_BLOCK_Y, (iz4 - iz1) / THREADS_PER_BLOCK_Z);
        dim3 threads (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
        dim3 blocks2B (nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, nz / THREADS_PER_BLOCK_Z);
        dim3 threads2B (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
        double tt;
        double functime, runtime, walltime;
        int ot2 = 0;
        int ot1 = 0;
        iter = 0;
        tt = 0.0;
        MPI_Barrier (MPI_COMM_WORLD);
        walltime = MPI_Wtime ();
        int n;
        for (n = 0; n < nac; n++)
       {
          hipMemcpy (lambda + n* 3* 3,  ac[n].lambda, 3* 3* sizeof (Dtype), hipMemcpyHostToDevice);
       }
        //CPU --> DCU dgemm
        hipMemcpy (mpxi,  MPXI,  nx* nx* sizeof (Dtype), hipMemcpyHostToDevice);
        hipMemcpy (mpyi,  MPYI,  ny* ny* sizeof (Dtype), hipMemcpyHostToDevice);
        hipMemcpy (mpzi,  MPZI,  nz* nz* sizeof (Dtype), hipMemcpyHostToDevice);
        hipMemcpy (mpx,   MPX,   nx* nx* sizeof (Dtype), hipMemcpyHostToDevice);
	hipMemcpy (mpy,   MPY,   ny* ny* sizeof (Dtype), hipMemcpyHostToDevice);
        hipMemcpy (mpz,   MPZ,   nz* nz* sizeof (Dtype), hipMemcpyHostToDevice);
        //CPU --> DCU update
        hipMemcpy (ddx,  DDX,  nx * sizeof (Dtype), hipMemcpyHostToDevice);
        hipMemcpy (ddy,  DDY,  ny * sizeof (Dtype), hipMemcpyHostToDevice);
        hipMemcpy (ddz,  DDZ,  nz * sizeof (Dtype), hipMemcpyHostToDevice);
        dim3 blocksEe (nx / THREADS_PER_BLOCK_X, ny / THREADS_PER_BLOCK_Y, nac*nz / THREADS_PER_BLOCK_Z);
        dim3 threadsEe (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
        while   (tt   <   t_total)
       {
          dgemm_time = 0.0f;
          copy_time = 0.0f;
          F1_time = 0.0f;
          F2_time = 0.0f;
	  elastic_copyin_time = 0.0f;
	  skipin_copyin_time = 0.0f;
	  elastic_copyout_time = 0.0f;
	  elastic_Memcpy_time = 0.0f;
	  elastic_multiply_BN_time = 0.0f;
          fft_forward_A_time = 0.0f;
          fft_forward_B_time = 0.0f;
          fft_forward_C_time = 0.0f;
          fft_forward_D_time = 0.0f;
          forward_mpi_time = 0.0f;
          backward_mpi_time = 0.0f;
	  trilinear_interpolation_time = 0.0f;
          elastic_calculate_time = 0.0f;
          elastic_transfer_time = 0.0f;
          fieldE1_time = 0.0f;
          trans_time = 0.0f;
          trans_Memcpy_time = 0.0f;
          trans_MPI_time = 0.0f;
          trans_pack_time = 0.0f;
          trans_enlarge_time = 0.0f;
          trans_mu_time = 0.0f;
          trans_unpack_time = 0.0f;
          Tu_time = 0.0f;
          updateU_new_time  = 0.0f;

          if (iter > 0)
         {
             hipLaunchKernelGGL (Ee_dev, blocksEe, threadsEe, 0, 0,\
			            fieldE, fielde);
             for (int m = 0; m < nac; m++)
            {
               hipMemcpy (ac[m].fieldE, fieldE +m*offset, nx* ny* nz* sizeof (double), hipMemcpyDeviceToHost);
            }
             //print_info();
             check_soln_new(tt);
         }
          if (iter < 1000)
         {
            for (int t = 0; t < nac; t++)
           {
               myball(iter, rad, ac[t].fieldE);
           }
            for (n = 0; n < nac; n++)
           {
               hipMemcpy (fieldE+ n* offset, ac[n].fieldE, nx* ny* nz* sizeof (double), hipMemcpyHostToDevice);
           }
         }
          if (iter == 0)
         {
             //print_info();
             check_soln_new(tt);
         }
         tt   +=   dt;
         runtime = MPI_Wtime ();
#ifdef SCLETD_DEBUG 
    hipEventRecord (st2, NULL);
    hipEventSynchronize (st2);
#endif

         transfer();
#ifdef SCLETD_DEBUG
    hipEventRecord(ed2, NULL);
    hipEventSynchronize (ed2);
    hipEventElapsedTime (&timer2, st2, ed2);
    trans_time = timer2;
#endif

#ifdef SCLETD_DEBUG
    hipEventRecord (st2, NULL);
    hipEventSynchronize (st2);
#endif

         elastic_calculate();

#ifdef SCLETD_DEBUG
    hipEventRecord(ed2, NULL);
    hipEventSynchronize (ed2);
    hipEventElapsedTime (&timer2, st2, ed2);
    elastic_calculate_time = timer2;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

	 if (ELASTIC == 1)
	 {
           elastic_transfer();
	 }

#ifdef SCLETD_DEBUG
    hipEventRecord(ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    elastic_transfer_time = timer;
#endif

         for (n = 0; n < nac; n++)
        {

#ifdef SCLETD_DEBUG
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif
          hipLaunchKernelGGL (ac_add_F1_F2_dev, blocks, threads, 0, 0,\
                	      f1 + n * offset, f2 + n * offset, Elas, epn2, ac[n].LE,\
                              ANISOTROPIC, ELASTIC, ElasticScale, n);
#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    fieldE1_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif
          rocblas_dgemm (handle, rocblas_operation_transpose, rocblas_operation_transpose, x_n, x_k, x_m, &alpha, f1 + n * offset, x_k, mpxi, x_m, &beta, f2 + n * offset, x_n);
          rocblas_dgemm (handle, rocblas_operation_transpose, rocblas_operation_transpose, y_n, y_k, y_m, &alpha, f2 + n * offset, y_k, mpyi, y_m, &beta, f1 + n * offset, y_n);
          rocblas_dgemm (handle, rocblas_operation_transpose, rocblas_operation_transpose, z_n, z_k, z_m, &alpha, f1 + n * offset, z_k, mpzi, z_m, &beta, f2 + n * offset, z_n);
#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    dgemm_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif

          hipLaunchKernelGGL (ac_updateU_new, blocks, threads, 0, 0,\
                              n, fielde + n * offset, f2 + n * offset, ddx, ddy, ddz, ac[n].LE, ac[n].KE, epn2, phiE);
#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    updateU_new_time += timer;
    hipEventRecord (st, NULL);
    hipEventSynchronize (st);
#endif
          rocblas_dgemm (handle, rocblas_operation_transpose, rocblas_operation_transpose, x_n, x_k, x_m, &alpha, fielde + n * offset, x_k, mpx, x_m, &beta, f1 + n * offset, x_n);
          rocblas_dgemm (handle, rocblas_operation_transpose, rocblas_operation_transpose, y_n, y_k, y_m, &alpha, f1 + n * offset, y_k, mpy, y_m, &beta, f2 + n * offset, y_n);
          rocblas_dgemm (handle, rocblas_operation_transpose, rocblas_operation_transpose, z_n, z_k, z_m, &alpha, f2 + n * offset, z_k, mpz, z_m, &beta, fielde + n * offset, z_n);
#ifdef SCLETD_DEBUG
    hipEventRecord (ed, NULL);
    hipEventSynchronize (ed);
    hipEventElapsedTime (&timer, st, ed);
    dgemm_time += timer;
#endif
      }

    ot1 += 1;
    if (ot1 >= nout) {
      ot1 = 0;
      irun += 1;
      for (n = 0; n < nac; n++){
          hipLaunchKernelGGL (Dtype2B_E, blocks2B, threads2B, 0, 0, fielde + n * offset, field2B + n * offset);
      }
      write_field2B (irun);
    }

    ot2 += 1;
    if (ot2 >= nchk) {
      ot2 = 0;
      write_chk ();
      chk = (chk + 1) % 2;
    }
    MPI_Barrier (MPI_COMM_WORLD);
	


        if (myrank == prank) {
#ifdef SCLETD_DEBUG
		 calc_time = elastic_copyin_time+skipin_copyin_time+elastic_Memcpy_time+elastic_multiply_BN_time+backward_mpi_time+trilinear_interpolation_time+elastic_copyout_time;
                 printf ("*******************************\n");
                 printf ("dgemm_time\t%lf\n",dgemm_time);
                 //printf ("Tu_time\t%lf\n",Tu_time);
                 //printf ("copy_time\t%lf\n",copy_time);
                 //printf ("chemical_time\t%lf\n",F1_time);
                 //printf ("anisotropic_time\t%lf\n",F2_time);
                 //printf ("elastic_copyin_time\t%lf\n",elastic_copyin_time);
                 printf ("256to64_time\t%lf\n",skipin_copyin_time);
                 printf ("elastic_Memcpy_time\t%lf\n",elastic_Memcpy_time);
                 printf ("elastic_multiply_BN_time\t%lf\n",elastic_multiply_BN_time);
                 //printf ("fft_forward_A_time\t%lf\n",fft_forward_A_time);
                 //printf ("fft_forward_B_time\t%lf\n",fft_forward_B_time);
                 //printf ("fft_forward_C_time\t%lf\n",fft_forward_C_time);
                 //printf ("fft_forward_D_time\t%lf\n",fft_forward_D_time);
                 //printf ("forward_mpi_time\t%lf\n",forward_mpi_time);
                 printf ("backward_mpi_time\t%lf\n",backward_mpi_time);
                 printf ("64to256_time\t%lf\n",trilinear_interpolation_time);
                 //printf ("elastic_copyout_time\t%lf\n",elastic_copyout_time);
                 //printf ("check_elastic_calculate_time\t%lf\n",calc_time);
                 printf ("elastic_calculate_time\t%lf\n",elastic_calculate_time);
                 printf ("elastic_transfer_time\t%lf\n",elastic_transfer_time);
                 printf ("nonlinear_add_time\t%lf\n",fieldE1_time);
                 printf ("trans_time\t%lf\n",trans_time);
                 printf ("trans_Memcpy_time\t%lf\n",trans_Memcpy_time);
                 printf ("trans_MPI_time\t%lf\n",trans_MPI_time);
                 printf ("trans_pack_time\t%lf\n",trans_pack_time);
                 printf ("trans_enlarge_time\t%lf\n",trans_enlarge_time);
                 printf ("trans_mu_time\t%lf\n",trans_mu_time);
                 printf ("trans_unpack_time\t%lf\n",trans_unpack_time);
                 printf ("updateU_new_time\t%lf\n",updateU_new_time);
#endif
                 printf ("runtime\t%lf\n",MPI_Wtime () - runtime);
                 printf ("--------------------------------------------------!\n");
       }
       iter++;
     }
     MPI_Barrier (MPI_COMM_WORLD);

     walltime = MPI_Wtime () - walltime;
     MPI_Reduce (&walltime, &runtime, 1, MPI_DOUBLE, MPI_MAX, prank, MPI_COMM_WORLD);
     if (myrank == prank) {
       printf ("time\t\t%lf\n", tt);
       printf ("wall time\t%lf\n", runtime);
    }
    check_soln_new(tt);

}
	
			   
