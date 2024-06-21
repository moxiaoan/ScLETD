#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "ScLETD.h"
#include "mkl.h"

#define MoleVolume (1.4e-5)
// scale elastic times
#define SCALE_ELASTIC 4e7
// f1 = M (-f  + Wu)
/*
n : order number of function
this function calculate local energy desity 
and boundary condition of eta
*/
void ac_calc_F1(int n)
{
	int m;
	int i, j, k;
	double tmp1, tmp2;
  double f, eta, c, c_alpha, c_delta, f_alpha, f_delta;
  // 1.848e9 0.429e9
  double w = 0.429e9; //1.848e9;
  // calculate local free energy desity
	for (k = iz1; k < iz4; k++)
	{
		for (j = iy1; j < iy4; j++)
		{
			for (i = ix1; i < ix4; i++)
			{
        // move data to tmp variable
        eta = ac[n].fieldE[k * nx * ny + j * nx + i];
        c_alpha = ch[0].c_alpha[k * nx * ny + j * nx + i];
        c_delta = ch[0].c_delta[k * nx * ny + j * nx + i];
        // calculate f(c_alpha) and f(c_delta)
        f_alpha = (-0.0241 - 3.8879 * c_alpha + 3.9057 * c_alpha * c_alpha);
        f_delta = (3.5373 - 15.3401 * c_delta + 10.0004 * c_delta * c_delta);
        //f = fl = (1/Vm) * h'(eta) 
        // * (f_delta - f_alpha + (c_alpha - c_delta) * f_alpha'(c_alpha)) + w * g'(eta)
        f =  f_delta - f_alpha;
        f += (c_alpha - c_delta) * (-3.8879 + 7.8114 * c_alpha);
        f *= ((1.0e4/MoleVolume) * (6 * eta - 6 * eta * eta));
        f += w * (2 * eta - 6 * eta * eta + 4 * eta * eta * eta);
        // f1 = M(-f + K * eta)
        ac[n].f1[k *nx * ny + j * nx + i] = -f;
			}
		}
	}
  // add boundary condition of field variable at inner and periodic boundary
  // add left boundary
	if (left >= 0)
	{
		i = ix1;
		for (k = iz1; k < iz4; k++)
		{
			for (j = iy1; j < iy4; j++)
			{
				ac[n].f1[k * nx * ny + j * nx + i] += ac[n].epn2 * ac[n].fieldEu_left[k * ny + j] / hx / hx;
			}
		}
	}

  // add right boundary
	if (right >= 0)
	{
		i = ix4 - 1;
		for (k = iz1; k < iz4; k++)
		{
			for (j = iy1; j < iy4; j++)
			{
				ac[n].f1[k * nx * ny + j * nx + i] += ac[n].epn2 * ac[n].fieldEu_right[k * ny + j] / hx / hx;
			}
		}
	}

  // add top boundary
	if (top >= 0)
	{
		j = iy1;
		for (k = iz1; k < iz4; k++)
		{
			for (i = ix1; i < ix4; i++)
			{
				ac[n].f1[k * nx * ny + j * nx + i] += ac[n].epn2 * ac[n].fieldEu_top[k * nx + i] / hy / hy;
			}
		}
	}

  // add bottom boundary
	if (bottom >= 0)
	{
		j = iy4 - 1;
		for (k = iz1; k < iz4; k++)
		{
			for (i = ix1; i < ix4; i++)
			{
				ac[n].f1[k * nx * ny + j * nx + i] += ac[n].epn2 * ac[n].fieldEu_bottom[k * nx + i] / hy / hy;
			}
		}
	}

  // add front boundary
	if (front >= 0)
	{
		k = iz1;
		for (j = iy1; j < iy4; j++)
		{
			for (i = ix1; i < ix4; i++)
			{
				ac[n].f1[k * nx * ny + j * nx + i] += ac[n].epn2 * ac[n].fieldEu_front[j * nx + i] / hz / hz;
			}
		}
	}

  // add back boundary
	if (back >= 0)
	{
		k = iz4 - 1;
		for (j = iy1; j < iy4; j++)
		{
			for (i = ix1; i < ix4; i++)
			{
				ac[n].f1[k * nx * ny + j * nx + i] += ac[n].epn2 * ac[n].fieldEu_back[j * nx + i] / hz / hz;
			}
		}
	}
}

 
/*
calculate all nonlinear items of Allen-Cahn function
n : which function to calculate
f : sum of nonliear items
ac[n].f1 : chemical local energy density
ac[n].f2 : anisotropic item
ac[n].felas : elastic energy
*/
void
ac_calc_FU (int n, double *f)
{
 int i, j, k;
  // ac[n].f1
  // chemical local energy density
  {
    ac_calc_F1(n);
    for (k = iz1; k < iz4; k++)
    {
      for (j = iy1; j < iy4; j++)
      {
        for (i = ix1; i < ix4; i++)
        {
          f[k * ny * nx + j * nx + i] = ac[n].LE * ac[n].f1[k * ny * nx + j * nx + i];
        }
      }
    }
  }


  // ac[n].f2 anisotropic nonliear items
  if (ANISOTROPIC == 1)
  {
    ac_calc_F2(n);

    for (k = iz1; k < iz4; k++)
    {
      for (j = iy1; j < iy4; j++)
      {
        for (i = ix1; i < ix4; i++)
        {
          f[k * ny * nx + j * nx + i] -= ac[n].epn2 * ac[n].LE * ac[n].f2[k * ny * nx + j * nx + i];
        }
      }
    }
  }

  // elastic energy, scale SCALE_ELASTIC times, make sure it's large enough
  // check_max(f);
  if (ELASTIC == 1) {
    for (k = iz1; k < iz4; k++) {
      for (j = iy1; j < iy4; j++) {
        for (i = ix1; i < ix4; i++) {
          f[k * ny * nx + j * nx + i] -= ac[n].LE * ac[n].felas[k * ny * nx + j * nx + i] * SCALE_ELASTIC;  
        }
      }
    }
  }

  // parameter kappa to maintain stability
  // * kappa * eta
  {
    for (k = iz1; k < iz4; k++)
    {
      for (j = iy1; j < iy4; j++)
      {
        for (i = ix1; i < ix4; i++)
        {
          f[k * ny * nx + j * nx + i] += ac[n].LE * ac[n].KE * ac[n].fieldE[k * ny * nx + j * nx + i];
        }
      }
    }
  }
}

// calculate last step's c_alpha and c_delta using c and eta
void calc_c_alpha_delta()
{
  int i, j, k;
  double c, eta, tmp, c_alpha, c_delta;
  /* 
   c_alpha = (c - 0.5726 * (3 * eta * eta - 2 * eta * eta * eta))
             /(1 - 0.6094 * (3 * eta * eta - 2 * eta * eta * eta))
   c_delta = 0.5726 + 0.3906 * c_alpha
  */ 
  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
        c = ch[0].fieldCI[k * nx * ny + j * nx + i];
        eta = ac[0].fieldE[k * nx * ny + j * nx + i];
        tmp = 1 - 0.6094 * (3 * eta * eta - 2 * eta * eta * eta);
        if (tmp < 1e-10) {
          tmp = 1e-30;
        }
        c_alpha = (c - 0.5726 * (3 * eta * eta - 2 * eta * eta * eta)) / tmp;
        c_delta = 0.5726 + 0.3906 * c_alpha;
        ch[0].c_alpha[k * nx * ny + j * nx + i] = c_alpha;
        ch[0].c_delta[k * nx * ny + j * nx + i] = c_delta;
      }
    }
  }

  // enlarge varible for laplace
  int f_k = (ny + 2 * 2) * (nx + 2 * 2);
  int f_j = nx + 2 * 2;
  for (k = 0; k < nz + 2 * 2; k++) {
    for (j = 0; j < ny + 2 * 2; j++) {
      for (i = 0; i < nx + 2 * 2; i++) {
        c = ch[0].fieldCIr[k * f_k + j * f_j + i];
        eta = ac[0].fieldEr[k * f_k + j * f_j + i];
        tmp = 1 - 0.6094 * (3 * eta * eta - 2 * eta * eta * eta);
        if (tmp < 1e-8) {
          tmp = 1e-8;
        }
        c_alpha = (c - 0.5726 * (3 * eta * eta - 2 * eta * eta * eta)) / tmp;
        c_delta = 0.5726 + 0.3906 * c_alpha;
        ch[0].c_alpha_r[k * f_k + j * f_j + i] = c_alpha;
        ch[0].c_delta_r[k * f_k + j * f_j + i] = c_delta;
      }
    }
  }
}

void
ch_calc_FU (int n, double *fieldci1)
{
  int i, j, k;
  double eta, c, tmp, c_alpha, c_delta;
  int f_k = (ny + 2 * 2) * (nx + 2 * 2);
  int f_j = nx + 2 * 2;
  int offset = 2 * f_k + 2 * f_j + 2;

  // ft = (c_alpha - c_delta) * h'(eta) * grad(eta)
  for (k = 0; k < nz + 2*2; k++) {
    for (j = 0; j < ny + 2*2; j++) {
      for (i = 0; i < nx + 2*2; i++) {
        eta = ac[0].fieldEr[k * f_k + j * f_j + i];
        c_alpha = ch[0].c_alpha_r[k * f_k + j * f_j + i];
        c_delta = ch[0].c_delta_r[k * f_k + j * f_j + i];
        // tmp =  (c_alpha - c_delta) * h'(eta);  
        tmp = (c_alpha - c_delta) * (6 * eta - 6 * eta * eta);
        ch[0].ftr[k * f_k + j * f_j + i] = tmp;
      }
    }
  }
  //check_max(ch[0].ftr);
  // using central finit volume method calculate divergence(gradient)
  central_finite_volume (n, fieldci1, ch[0].ftr, ac[0].fieldEr);
  // kappa maintain stability
  // f += kappa * c
  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
        fieldci1[k * nx * ny + j * nx + i] += ch[n].KCI * ch[n].fieldCI[k * nx * ny + j * nx + i];
      }
    }
  }
  // add boundary condition of field variable at inner and periodic boundary
  // f += Wc
  // add left boundary condition
  if (left >= 0) {
    i = ix1;
    for (k = iz1; k < iz4; k++) {
      for (j = iy1; j < iy4; j++) {
        fieldci1[k * nx * ny + j * nx + i] += ch[n].epn2 * ch[n].fieldCIu_left[k * ny + j] / hx / hx;
      }
    }
  }

  // add right boundary condition
  if (right >= 0) {
    i = ix4 - 1;
    for (k = iz1; k < iz4; k++) {
      for (j = iy1; j < iy4; j++) {
        fieldci1[k * nx * ny + j * nx + i] += ch[n].epn2 * ch[n].fieldCIu_right[k * ny + j] / hx / hx;
      }
    }
  }

  // add top boundary condition
  if (top >= 0) {
    j = iy1;
    for (k = iz1; k < iz4; k++) {
      for (i = ix1; i < ix4; i++) {
        fieldci1[k * nx * ny + j * nx + i] += ch[n].epn2 * ch[n].fieldCIu_top[k * nx + i] / hy / hy;
      }
    }
  }

  // add bottom boundary condition
  if (bottom >= 0) {
    j = iy4 - 1;
    for (k = iz1; k < iz4; k++) {
      for (i = ix1; i < ix4; i++) {
        fieldci1[k * nx * ny + j * nx + i] += ch[n].epn2 * ch[n].fieldCIu_bottom[k * nx + i] / hy / hy;
      }
    }
  }

  // add front boundary condition
  if (front >= 0) {
    k = iz1;
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
        fieldci1[k * nx * ny + j * nx + i] += ch[n].epn2 * ch[n].fieldCIu_front[j * nx + i] / hz / hz;
      }
    }
  }

  // add back boundary condition
  if (back >= 0) {
    k = iz4 - 1;
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
        fieldci1[k * nx * ny + j * nx + i] += ch[n].epn2 * ch[n].fieldCIu_back[j * nx + i] / hz / hz;
      }
    }
  }
  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
        fieldci1[k * nx * ny + j * nx + i] *= ch[n].LCI;
      }
    }
  }
}

void
ac_updateU_new (int n, double *field, double *field1)
{
  double tmp, Hijk;
  int i, j, k, l;

  // calculate Hijkl and update 1 order eta
  // Hijk = -M * ((kkx * dx + kky * dy + kkz * dz) * k^2 - kappa);
  // eta = (exp(-dt * Hijkl) * eta + (1/H) * (1 - exp(-dt * Hijk)) * g(eta)
  for (j = iy1; j < iy4; j++) {
    for (i = ix1; i < ix4; i++) {
      for (k = iz1; k < iz4; k++) {
	    l = j * nz * nx + i * nz + k;
	    tmp = kkz * DDZ[k] + kky * DDY[j] + kkx * DDX[i];
	    Hijk = -ac[n].LE * (tmp * ac[n].epn2 - ac[n].KE);
	    if (fabs(Hijk) < 1.0e-8) {
	      Hijk = 0.0;
	    }
	    tmp = 1.0 - ac[n].phiE[l] * Hijk;
	    field[l] = tmp * field[l] + ac[n].phiE[l] * field1[l];
      }
    }
  }
}

void
ch_updateU_new (int n, double *field, double *field1)
{
  double tmp, Hijk;
  int i, j, k, l;

  // Hijk = -M * ((dx + dy + dz)  - kappa);
  // eta = (exp(-dt * Hijkl) * eta + (1/H) * (1 - exp(-dt * Hijkl)) * g(eta)
  for (j = iy1; j < iy4; j++) {
    for (i = ix1; i < ix4; i++) {
      for (k = iz1; k < iz4; k++) {
	      l = j * nz * nx + i * nz + k;
  	    tmp = DDZ[k] + DDY[j] + DDX[i];
        Hijk = -ch[n].LCI * (tmp * ch[n].epn2 - ch[n].KCI);
        if (fabs(Hijk) < 1.0e-8) {
          Hijk = 0.0;
        }
	      tmp = 1.0 - ch[n].phiCI[l] * Hijk;
	      field[l] = tmp * field[l] + ch[n].phiCI[l] * field1[l];
      }
    }
  }
}

// 2 order update
void
prepare_U1_new (double *field1, double *field2)
{
  int i, j, k, l;

  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
	    l = k * nx * ny + j * nx + i;
	    field2[l] = field2[l] - field1[l];
      }
    }
  }
}

// 2 order update
void
prepare_U2_new (double *phi, double *field1, double *field2)
{
  int i, j, k, l;

  for (j = iy1; j < iy4; j++) {
    for (i = ix1; i < ix4; i++) {
      for (k = iz1; k < iz4; k++) {
	    l = j * nx * nz + i * nz + k;
	    field2[l] = phi[l] * field2[l];
      }
    }
  }
}

// 2 order update
void
correct_U_new (double *field, double *field1)
{
  int i, j, k, l;

  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
	    l = k * nx * ny + j * nx + i;
	    field[l] += field1[l];
      }
    }
  }
}

// transform varibale from xyz to yzx
void
xyz_yzx (double *f, double *ft)
{
  int i, j, k;

  for (i = ix1; i < ix4; i++) {
    for (k = iz1; k < iz4; k++) {
      for (j = iy1; j < iy4; j++) {
	      ft[i * ny * nz + k * ny + j] = f[k * nx * ny + j * nx + i];
      }
    }
  }
}


// transform varibale from yzx to zxy
void
yzx_zxy (double *f, double *ft)
{
  int i, j, k;

  for (j = iy1; j < iy4; j++) {
    for (i = ix1; i < ix4; i++) {
      for (k = iz1; k < iz4; k++) {
	ft[j * nz * nx + i * nz + k] = f[i * ny * nz + k * ny + j];
      }
    }
  }
}


// transform varibale from zxy to xyz
void
zxy_xyz (double *f, double *ft)
{
  int i, j, k;

  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
	ft[k * nx * ny + j * nx + i] = f[j * nz * nx + i * nz + k];
      }
    }
  }
}


// multiply Px
void
PUX (double *A, double *B, double *C, double *D)
{
  int m, n, k;

  m = nx;
  n = ny * nz;
  k = nx;

  cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, D, m);
  xyz_yzx (D, C);
}


// multiply Px
void
PUY (double *A, double *B, double *C, double *D)
{
  int m, n, k;

  m = ny;
  n = nz * nx;
  k = ny;

  cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, D, m);
  yzx_zxy (D, C);
}


// multiply Px
void
PUZ (double *A, double *B, double *C, double *D, double *E)
{
  int m, n, k;

  m = nz;
  n = nx * ny;
  k = nz;

  switch (stage) {
  case 0:

    cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);

    break;

  case 1:

    cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, D, m);

    break;

  case 2:

    cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, D, m);
    xyz_yzx (D, C);

    break;
  }
}

