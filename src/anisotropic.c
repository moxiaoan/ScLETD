#include <stdio.h>
#include "mpi.h"
#include "ScLETD.h"
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>

/*
 fa : field variable
 ft : some polynomial
 f : divergence(ft*gradient(fa))
 f = ((t_right + t_middle)*(a_right - a_middle) - (t_middle + t_left)*(a_middle - a_left)) / 2.0 / hx / hx;
   + ((t_bottom + t_middle)*(a_bottom - a_middle) - (t_middle + t_top)*(a_middle - a_top)) / 2.0 / hy / hy;
   + ((t_back + t_middle)*(a_back - a_middle) - (t_middle + t_front)*(a_middle - a_front)) / 2.0 / hz / hz;

*/

void central_finite_volume (int n, double *f, double *ft, double *fa)
{
	int i, j, k;
	int n_left, n_right, n_top, n_bottom, n_front, n_back;
	double a_left, a_right, a_top, a_bottom, a_front, a_back, a_middle;
	double t_left, t_right, t_top, t_bottom, t_front, t_back, t_middle;
  int f_k = (ny + 2*2) * (nx+2*2);
  int f_j = (nx + 2*2);
  int offset = 2 * f_k + 2 * f_j + 2;

	for (k = iz1; k < iz4; k++)
	{
		for (j = iy1; j < iy4; j++)
		{
			for (i = ix1; i < ix4; i++)
			{

		    n_left = i - 1;
				n_right = i + 1;
			  n_top = j - 1;
				n_bottom = j + 1;
				n_front = k - 1;
				n_back = k + 1;
				if (left < 0)
				{
					if (i == ix1)
					{
						n_left = i + 1;
					}
				} 
				if (right < 0)
				{
					if (i == ix4-1)
					{
						n_right = i - 1;
					}
				} 

				if (top < 0)
				{
					if (j == iy1)
					{
						n_top = j + 1;
					}
				}
        if (bottom < 0) 
				{
					if (j == iy4 - 1)
					{
						n_bottom = j - 1;
					}
				}

				if (front < 0)
				{
					if (k == iz1)
					{
						n_front = k + 1;
					}
				}
				if (back < 0) 
				{
          if (k == iz4 - 1)
					{
						n_back = k - 1;
					}
				}
        a_front = fa[n_front * f_k + j * f_j + i + offset];
				t_front = ft[n_front * f_k + j * f_j + i + offset];
				a_back = fa[n_back * f_k + j * f_j + i + offset];
				t_back = ft[n_back * f_k + j * f_j + i + offset];
				a_top = fa[k * f_k + n_top * f_j + i + offset];
				t_top = ft[k * f_k + n_top * f_j + i + offset];
				a_bottom = fa[k * f_k + n_bottom * f_j + i + offset];
				t_bottom = ft[k * f_k + n_bottom * f_j + i + offset];
    		a_right = fa[k * f_k + j * f_j + n_right + offset];
				t_right = ft[k * f_k + j * f_j + n_right + offset];
				a_left = fa[k * f_k + j * f_j + n_left + offset];
				t_left = ft[k * f_k + j * f_j + n_left + offset];
        a_middle = fa[k * f_k  + j * f_j + i + offset];
        t_middle = ft[k * f_k + j * f_j + i + offset];

        double tmp = 0;
        tmp += ((t_right + t_middle)*(a_right - a_middle) - (t_middle + t_left)*(a_middle - a_left)) / 2.0 / hx / hx;
        tmp += ((t_bottom + t_middle)*(a_bottom - a_middle) - (t_middle + t_top)*(a_middle - a_top)) / 2.0 / hy / hy;
        tmp += ((t_back + t_middle)*(a_back - a_middle) - (t_middle + t_front)*(a_middle - a_front)) / 2.0 / hz / hz;
        f[k * nx * ny + j * nx + i] = tmp;

			}
		}
	}
}

/*
divergence(ft*gradient(fa))
fxx = (a_right - 2 * a_middle + a_left) / hx / hx;
fyy = (a_bottom - 2 * a_middle + a_top) / hy / hy;
fzz = (a_back - 2 * a_middle + a_front) / hz / hz;
fxy = ((a_right_bottom - a_left_bottom) - (a_right_top - a_left_top)) / (2*hx) / (2*hy);
fxz = ((a_right_back - a_left_back) - (a_right_front - a_left_front)) / (2*hx) / (2*hz);
fyz = ((a_bottom_back - a_top_back) - (a_bottom_front - a_top_front)) / (2*hy) / (2*hz);
f   =   lambda[0][0] * fxx;
      +  lambda[0][1] * fxy;
      +  lambda[0][2] * fxz;
      +  lambda[1][0] * fxy;
      +  lambda[1][1] * fyy;
      +  lambda[1][2] * fyz;
      +  lambda[2][0] * fxz;
      +  lambda[2][1] * fyz;
      +  lambda[2][2] * fzz;
*/
void anisotropic_calc (int n, double *f, double *fa)
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
	for (k = iz1; k < iz4; k++)
	{
		for (j = iy1; j < iy4; j++)
		{
			for (i = ix1; i < ix4; i++)
			{

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

        double tmp = 0;
        fxx = (a_right - 2 * a_middle + a_left) / hx / hx;
        fyy = (a_bottom - 2 * a_middle + a_top) / hy / hy;
        fzz = (a_back - 2 * a_middle + a_front) / hz / hz;
        fxy = ((a_right_bottom - a_left_bottom) - (a_right_top - a_left_top)) / (2*hx) / (2*hy);
        fxz = ((a_right_back - a_left_back) - (a_right_front - a_left_front)) / (2*hx) / (2*hz);
        fyz = ((a_bottom_back - a_top_back) - (a_bottom_front - a_top_front)) / (2*hy) / (2*hz);
        tmp += ac[n].lambda[0][0] * fxx;
        tmp += ac[n].lambda[0][1] * fxy;
        tmp += ac[n].lambda[0][2] * fxz;
        tmp += ac[n].lambda[1][0] * fxy;
        tmp += ac[n].lambda[1][1] * fyy;
        tmp += ac[n].lambda[1][2] * fyz;
        tmp += ac[n].lambda[2][0] * fxz;
        tmp += ac[n].lambda[2][1] * fyz;
        tmp += ac[n].lambda[2][2] * fzz;
        f[k * nx * ny + j * nx + i] = tmp;
			}
		}
	}
}


// fieldEt = divergence(D_lambda*gradient(eta))
void ac_calc_F2(int n)
{
	int i, j, k;
  anisotropic_calc(n, ac[n].f2, ac[n].fieldEr);
}

// read D and lambda in 
// divergence((D-lambda)gradient(eta))
void anisotropic_input()
{
	int i, j, n;
	double D[3][3];
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

	fp = fopen("anisotropic_input.txt", "r");
	if (fp == NULL)
	{
		printf("open anisotropic_input.txt Fail!");
		exit(EXIT_FAILURE);
	}

  double tmp;
	for (n = 0; n < nac; n++)
	{
		for (i = 0; i < 3; i++)
		{
			for (j = 0; j < 3; j++)
			{
				fscanf(fp, "%lf", &tmp);
        ac[n].lambda[i][j] = tmp;
			}
		}
	}
	fclose(fp);

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

