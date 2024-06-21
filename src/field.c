#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include "mpi.h"
#include "ScLETD.h"

// output field variable as binary double to file
void
outputfield ()
{
  int i, j, k, n;
  char filename[1024];
  FILE *file;
#if 0
  sprintf (filename, "./data/eta_%06d_%02d%02d%02d.dat", ioutput, cart_id[0], cart_id[1], cart_id[2]);
  file = fopen (filename, "wb");
  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
#endif
  sprintf (filename, "./data/eta_%06d_%02d%02d%02d.dat", ioutput, cart_id[0], cart_id[2], cart_id[1]);
  file = fopen (filename, "w");
  for (j = iy1; j < iy4; j++) {
    for (k = iz1; k < iz4; k++) {
      for (i = ix1; i < ix4; i++) {
        //fwrite (&ac[0].fieldE[k * nx * ny + j * nx + i], sizeof (double), 1, file);
        fprintf (file, "%+1.15lf\n", ac[0].fieldE[k * nx * ny + j * nx + i]);
      }
    }
  }
  fclose (file);
#if 0
  sprintf (filename, "./data/c_%06d_%02d%02d%02d.dat", ioutput, cart_id[0], cart_id[1], cart_id[2]);
  file = fopen (filename, "wb");
  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
#endif
  sprintf (filename, "./data/c_%06d_%02d%02d%02d.dat", ioutput, cart_id[0], cart_id[2], cart_id[1]);
  file = fopen (filename, "w");
  for (j = iy1; j < iy4; j++) {
    for (k = iz1; k < iz4; k++) {
      for (i = ix1; i < ix4; i++) {
        //fwrite (&ch[0].fieldCI[k * nx * ny + j * nx + i], sizeof (double), 1, file);
        fprintf (file, "%+1.15lf\n", ac[0].felas[k * nx * ny + j * nx + i]);
//        fprintf (file, "%+1.15lf\n", ch[0].fieldCI[k * nx * ny + j * nx + i]);
      }
    }
  }
  fclose (file);
  ioutput+=1;
}


/* write a section perpendicular to z axis */
void
write_section_z (char *fname, double *field)
{
  FILE *fp;
  int i, j, k;
  // middle of procs in z dimention
  if (cart_id[2] == (procs[2]/2)) {
    fp = fopen (fname, "w");
    if (fp == NULL) {
      printf ("fopen error %s!\n", strerror(errno));
      exit (1);
    }
    if (procs[2] == 1) {
      // middle of subdomain
      k = nz/2;
    } else {
      // first layer
      k = nghost;
    }
    for (j = nghost; j < ny-nghost; j++) {
      for (i = nghost; i < nx-nghost; i++) {
        fprintf (fp, "%+1.15lf ", field[k * nx * ny + j * nx + i]);
      }
      fprintf (fp, "\n");
    }
    fclose (fp);
  }
}

/* write a section perpendicular to x axis */
void
write_section_x (char *fname, double *field)
{
  FILE *fp;
  int i, j, k;
  // middle of procs in x dimention
  if (cart_id[0] == (procs[0]/2)) {
    fp = fopen (fname, "w");
    if (fp == NULL) {
      printf ("fopen error %s!\n", strerror(errno));
      exit (1);
    }
    if (procs[0] == 1) {
      // middle of subdomain
      i = nx/2;
    } else {
      // first layer
      i = nghost;
    }
    for (k = nghost; k < nz-nghost; k++) {
      for (j = nghost; j < ny-nghost; j++) {
        fprintf (fp, "%+1.15lf ", field[k * nx * ny + j * nx + i]);
      }
      fprintf (fp, "\n");
    }
    fclose (fp);
  }
}


/* write a section perpendicular to y axis */
void
write_section_y (char *fname, double *field)
{
  FILE *fp;
  int i, j, k;
  // middle of procs in y dimention
  if (cart_id[1] == (procs[1]/2)) {
    fp = fopen (fname, "w");
    if (fp == NULL) {
      printf ("fopen error %s!\n", strerror(errno));
      exit (1);
    }
    if (procs[1] == 1) {
      // middle of subdomain
      j = ny/2;
    } else {
      // first layer
      j = nghost;
    }
    for (k = nghost; k < nz-nghost; k++) {
      for (i = nghost; i < nx-nghost; i++) {
        fprintf (fp, "%+1.15lf ", field[k * nx * ny + j * nx + i]);
      }
      fprintf (fp, "\n");
    }
    fclose (fp);
  }
}

/*
fname : name of file
field : tensor of field variable 
 read field variable tensor from file when restart
*/
void
read_field (char *fname, double *field)
{
  FILE *fp;
  int i, j, k;
  fp = fopen (fname, "r");
  if (fp == NULL) {
    printf ("fopen error %s!\n", strerror(errno));
    exit (1);
  }
  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
        fscanf (fp, "%lf", &field[k * nx * ny + j * nx + i]);
      }
    }
  }
  fclose (fp);
}

/*
fname : name of file
field : tensor of field variable 
write field variable tensor to fiele when needed
*/
void
write_field (char* fname, double* field)
{
  FILE *fp;
  int i, j, k;
  fp = fopen (fname, "w");
  if (fp == NULL) {
    printf ("fopen error %s!\n", strerror(errno));
    exit (1);
  }
  for (k = iz1; k < iz4; k++) {
    for (j = iy1; j < iy4; j++) {
      for (i = ix1; i < ix4; i++) {
        fprintf (fp, "%+1.15lf ", field[k * nx * ny + j * nx + i]);
      }
      fprintf (fp, "\n");
    }
  }
  fclose (fp);
}

// init field variable of couple function
void
couple_init_field ()
{
  int n;
  int l;
  int i, j, k;
  double rad1 = 10.0;
  double cnt1 = (nx * procs[0] - 1) / 2.0;
  int x, y, z;
  for (n = 0; n < nac; n++) {
    for (l = 0; l < nx * ny * nz; l++) {
      ac[n].fieldE[l] = 0.0;
      ch[n].fieldCI[l] = 0.1; //0.0656;
    }
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        for (k = 0; k < nz; k++) {
          x = cart_id[0] * nx + i;
          y = cart_id[1] * ny + j;
          z = cart_id[2] * nz + k;
          // this is a sphere
          if ((x-cnt1) * (x-cnt1) + (y-cnt1) * (y-cnt1) + (z-cnt1) * (z-cnt1) < rad1*rad1)
          {
            ac[n].fieldE[k * nx * ny + j * nx + i] = 1.0;
            ch[n].fieldCI[k * nx * ny + j * nx + i] = 0.5982;
          }
        }
      }
    }
  }
}


// init field variable as cube
void
init_field_cube (double *field)
{
  int l;
  int i, j, k;
  int rad1;
  int cnt1;
  int x, y, z;
  cnt1 = nx * procs[0] / 2;
  rad1 = nx * procs[0] / 6;

  for (l = 0; l < nx * ny * nz; l++) {
    field[l] = -1.0;
  }

  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      for (k = 0; k < nz; k++) {
        x = cart_id[0] * nx + i;
        y = cart_id[1] * ny + j;
        z = cart_id[2] * nz + k;
        x = abs(x-cnt1);
        y = abs(y-cnt1);
        z = abs(z-cnt1);
        if ((x<rad1) && (y<rad1) && (z<rad1)){
          field[k * nx * ny + j * nx + i] = 1.0;
        }
      }
    }
  }
}

// init field variable as sphere at center of tensor
void
init_field_sphere (double *field, double epsilon)
{
  int l;
  int i, j, k;
  double x, y, z;
  for (l = 0; l < nx * ny * nz; l++) {
    field[l] = 0.0;
  }

  for (i = ix1; i < ix4; i++) {
    for (j = iy1; j < iy4; j++) {
      for (k = iz1; k < iz4; k++) {
        // none perme
        x = xmin + fieldgx[k * ny * nx + j * nx + i] * hx;
        y = ymin + fieldgy[k * ny * nx + j * nx + i] * hy;
        z = zmin + fieldgz[k * ny * nx + j * nx + i] * hz;
        field[k * nx * ny + j * nx + i] = tanh ((0.4 - sqrt(x * x + y * y + z * z)) / sqrt (2.0) / epsilon);
      }
    }
  }
}

// check variable and volume
void
check_soln_new (double time)
{
  if (myrank == prank) {
    printf ("--------------iter %d--------------\n", iter);
  }
  int n;
  int i, j, k;
  double tmp, maxtmp, mintmp, voltmp, engtmp, eng, engtmp_grad, mmax, mmin, vol, eng_grad, engtmp_local, eng_local, engtmp_elast, eng_elast;
  double f_val2[8], f_val[8], f_ux, f_uy, f_uz, c_alpha, c_delta, f_alpha, f_delta, tmp2;
  double w = 0.429e9;
  double MoleVolume = (1.0e4)/(1.4e-5);
  FILE *fp;
  char fname[1024];
  if (iter == 0) {
    // couple function write to csv file
    if (nac > 0 && nch > 0) {
        if (myrank == prank) {
        sprintf (fname, "%sphi_couple.csv", work_dir);
        fp = fopen (fname, "w");
        if (fp == NULL) {
          printf ("fopen error %s!\n", strerror(errno));
          exit (1);
        }
        fprintf (fp, "iter,eng\n");
        fclose (fp);
        }
    }
    // AC function write to csv file
    for (n = 0; n < nac; n++) {
      if (myrank == prank) {
        sprintf (fname, "%sphi_ac_%d.csv", work_dir, n);
        fp = fopen (fname, "w");
        if (fp == NULL) {
          printf ("fopen error %s!\n", strerror(errno));
          exit (1);
        }
        fprintf (fp, "iter,maxphiE,minphiE,volE,engE\n");
        fclose (fp);
      }
    }
    // CH function write to csv file
    for (n = 0; n < nch; n++) {
      if (myrank == prank) {
        sprintf (fname, "%sphi_ch_%d.csv", work_dir, n);
        fp = fopen (fname, "w");
        if (fp == NULL) {
          printf ("fopen error %s!\n", strerror(errno));
          exit (1);
        }
        fprintf (fp, "iter,maxphiCI,minphiCI,volCI,engCI\n");
        fclose (fp);
      }
    }
  }
  // AC function write to csv file
  for (n = 0; n < nac; n++) {
    if (myrank == prank) {
      sprintf (fname, "%sphi_ac_%d.csv", work_dir, n);
      errno = 0;
      fp = fopen (fname, "a");
      if (fp == NULL) {
        printf ("fopen error %s!\n", strerror(errno));
        exit (1);
      }
    }
    maxtmp = -1.0e30;
    mintmp = 1.0e30;
    voltmp = 0.0;
    engtmp = 0.0;
    // check min and max of field variable and sum volume
    for (k = iz2; k < iz3; k++) {
      for (j = iy2; j < iy3; j++) {
        for (i = ix2; i < ix3; i++) {
           tmp = ac[n].fieldE[k * nx * ny + j * nx + i];
          if (tmp > maxtmp) {
            maxtmp = tmp;
          }
          if (tmp < mintmp) {
            mintmp = tmp;
          }
          voltmp += tmp + 1.0;
        }
      }
    }
    voltmp *= hx * hy * hz;
    // calculate energy
    for (k = iz2; k < iz3 - 1; k++) {
      for (j = iy2; j < iy3 - 1; j++) {
        for (i = ix2; i < ix3 - 1; i++) {
          f_val[0] = ac[n].fieldE[k * nx * ny + j * nx + i];
          f_val[1] = ac[n].fieldE[k * nx * ny + j * nx + i + 1];
          f_val[2] = ac[n].fieldE[k * nx * ny + (j + 1) * nx + i];
          f_val[3] = ac[n].fieldE[k * nx * ny + (j + 1) * nx + i + 1];
          f_val[4] = ac[n].fieldE[(k + 1) * nx * ny + j * nx + i];
          f_val[5] = ac[n].fieldE[(k + 1) * nx * ny + j * nx + i + 1];
          f_val[6] = ac[n].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i];
          f_val[7] = ac[n].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i + 1];
          f_val2[0] = ch[n].fieldCI[k * nx * ny + j * nx + i];
          f_val2[1] = ch[n].fieldCI[k * nx * ny + j * nx + i + 1];
          f_val2[2] = ch[n].fieldCI[k * nx * ny + (j + 1) * nx + i];
          f_val2[3] = ch[n].fieldCI[k * nx * ny + (j + 1) * nx + i + 1];
          f_val2[4] = ch[n].fieldCI[(k + 1) * nx * ny + j * nx + i];
          f_val2[5] = ch[n].fieldCI[(k + 1) * nx * ny + j * nx + i + 1];
          f_val2[6] = ch[n].fieldCI[(k + 1) * nx * ny + (j + 1) * nx + i];
          f_val2[7] = ch[n].fieldCI[(k + 1) * nx * ny + (j + 1) * nx + i + 1];
          ac[n].u = (f_val[0] + f_val[1] + f_val[2] + f_val[3] + f_val[4] + f_val[5] + f_val[6] + f_val[7]) / 8;
          ch[n].c = (f_val2[0] + f_val2[1] + f_val2[2] + f_val2[3] + f_val2[4] + f_val2[5] + f_val2[6] + f_val2[7]) / 8;
	  tmp2 = 1 - 0.6094 * (3 * ac[n].u * ac[n].u - 2 * ac[n].u * ac[n].u * ac[n].u);
	  if (tmp2 < 1e-10) {
  	    tmp2 = 1e-30;
          }
	  c_alpha = (ch[n].c - 0.5726 * (3 * ac[n].u * ac[n].u - 2 * ac[n].u * ac[n].u * ac[n].u)) / tmp2;
	  c_delta = 0.5726 + 0.3906 * c_alpha;
	  f_alpha = (-0.0241 - 3.8879 * c_alpha + 3.9057 * c_alpha * c_alpha);
          f_delta = (3.5373 - 15.3401 * c_delta + 10.0004 * c_delta * c_delta);
          f_ux = ((f_val[1] - f_val[0]) + (f_val[3] - f_val[2]) + (f_val[5] - f_val[4]) + (f_val[7] - f_val[6])) / 4 / hx;
          f_uy = ((f_val[2] - f_val[0]) + (f_val[3] - f_val[1]) + (f_val[6] - f_val[4]) + (f_val[7] - f_val[5])) / 4 / hy;
          f_uz = ((f_val[4] - f_val[0]) + (f_val[5] - f_val[1]) + (f_val[6] - f_val[2]) + (f_val[7] - f_val[3])) / 4 / hz;
	  engtmp_grad += ac[n].epn2 * (f_ux * f_ux + f_uy * f_uy + 0.0005 * f_uz * f_uz) / 2;
          engtmp_local += (1.0 - (3.0 * ac[n].u * ac[n].u - 2.0 * ac[n].u * ac[n].u * ac[n].u)) * MoleVolume * f_alpha + (3.0 * ac[n].u * ac[n].u - 2.0 * ac[n].u * ac[n].u * ac[n].u) * MoleVolume * f_delta + w * (ac[n].u * ac[n].u - 2.0 * ac[n].u * ac[n].u * ac[n].u + ac[n].u * ac[n].u * ac[n].u * ac[n].u);
        }
      }
    }
    engtmp_grad = engtmp_grad * hx * hy * hz;
    engtmp_local = engtmp_local * hx * hy * hz;
#if 0
    for (i = 0; i < NX; i++)
    {
      for (j = 0; j < NY; j++)
      {
        for (k = 0; k < NZ; k++)
        {
          ac[n].elas_field[k * NY * NX + j * NX + i]
           = ac[n].fieldE[(k + nghost) * ny * nx + (j + nghost) * nx + i + nghost];
        }
      }
    }
    fft_forward(ac[n].elas_field, ac[n].theta_re, ac[n].theta_im);
    for (i = 0; i < NX; i++)
    {
      for (j = 0; j < NY; j++)
      {
        for (k = 0; k < NZ; k++)
        {
	  ac[n].elas_field[k * NY * NX + j * NX + i] = ac[n].Bn[k * NY * NX + j * NX + i] * (8.0*NX*NY*NZ*ac[n].theta_re*8.0*NX*NY*NZ*ac[n].theta_re + 8.0*NX*NY*NZ*ac[n].theta_im*8.0*NX*NY*NZ*ac[n].theta_im);
        }
      }
    }
    for (i = 0; i < NX; i++)
    {
      for (j = 0; j < NY; j++)
      {
        for (k = 0; k < NZ; k++)
        {
	  engtmp_elastic += 0.5 * ElasticScale * ac[n].elas_field[k * NY * NX + j * NX + i];
        }
      }
    }
    engtmp_elast = engtmp_elast * hx * hy * hz;
#endif
    // communicate along all procs
    MPI_Reduce (&maxtmp, &mmax, 1, MPI_DOUBLE, MPI_MAX, prank, MPI_COMM_WORLD);
    MPI_Reduce (&mintmp, &mmin, 1, MPI_DOUBLE, MPI_MIN, prank, MPI_COMM_WORLD);
    MPI_Reduce (&voltmp, &vol, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);
    MPI_Reduce (&engtmp_grad, &eng_grad, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);
    MPI_Reduce (&engtmp_local, &eng_local, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);
    //MPI_Reduce (&engtmp_elast, &eng_elast, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);
	
    // write min and max and volume of field variable
    if (myrank == prank) {
      printf ("ac[%d].fieldE\n", n);
      printf ("max phi\t\t%+1.15lf\n", mmax);
      printf ("min phi\t\t%+1.15lf\n", mmin);
#if 1
      printf ("vol\t\t%+1.15lf\n", vol);
      printf ("eng_grad\t\t%+1.15lf\n", eng_grad);
      printf ("eng_local\t\t%+1.15lf\n", eng_local);
      printf ("eng_elast\t\t%+1.15lf\n", eng_elast);
      fprintf (fp, "%d,%+1.15lf,%+1.15lf,%+1.15lf,%+1.15lf,%+1.15lf,%+1.15lf\n", iter, mmax, mmin, vol, eng_grad, eng_local, eng_elast);
#endif
      fclose (fp); 
    }
  }
  // CH write to csv file
  for (n = 0; n < nch; n++) {
    if (myrank == prank) {
      sprintf (fname, "%sphi_ch_%d.csv", work_dir, n);
      fp = fopen (fname, "a");
      if (fp == NULL) {
        printf ("fopen error %s!\n", strerror(errno));
        exit (1);
      }
    }  
    maxtmp = -1.0e30;
    mintmp = 1.0e30;
    voltmp = 0.0;
    engtmp = 0.0;

    // check min and max of field variable and sum volume
    for (k = iz2; k < iz3; k++) {
      for (j = iy2; j < iy3; j++) {
        for (i = ix2; i < ix3; i++) {
          tmp = ch[n].fieldCI[k * nx * ny + j * nx + i];
          if (tmp > maxtmp) {
            maxtmp = tmp;
          }
          if (tmp < mintmp) {
            mintmp = tmp;
          }
          voltmp += tmp + 1;
        }
      }
    }
    voltmp *= hx * hy * hz;
    // calculate energy
    for (k = iz2; k < iz3 - 1; k++) {
      for (j = iy2; j < iy3 - 1; j++) {
        for (i = ix2; i < ix3 - 1; i++) {
          f_val[0] = ch[n].fieldCI[k * nx * ny + j * nx + i];
          f_val[1] = ch[n].fieldCI[k * nx * ny + j * nx + i + 1];
          f_val[2] = ch[n].fieldCI[k * nx * ny + (j + 1) * nx + i];
          f_val[3] = ch[n].fieldCI[k * nx * ny + (j + 1) * nx + i + 1];
          f_val[4] = ch[n].fieldCI[(k + 1) * nx * ny + j * nx + i];
          f_val[5] = ch[n].fieldCI[(k + 1) * nx * ny + j * nx + i + 1];
          f_val[6] = ch[n].fieldCI[(k + 1) * nx * ny + (j + 1) * nx + i];
          f_val[7] = ch[n].fieldCI[(k + 1) * nx * ny + (j + 1) * nx + i + 1];
          ch[n].c = (f_val[0] + f_val[1] + f_val[2] + f_val[3] + f_val[4] + f_val[5] + f_val[6] + f_val[7]) / 8;
          f_ux = ((f_val[1] - f_val[0]) + (f_val[3] - f_val[2]) + (f_val[5] - f_val[4]) + (f_val[7] - f_val[6])) / 4 / hx;
          f_uy = ((f_val[2] - f_val[0]) + (f_val[3] - f_val[1]) + (f_val[6] - f_val[4]) + (f_val[7] - f_val[5])) / 4 / hy;
          f_uz = ((f_val[4] - f_val[0]) + (f_val[5] - f_val[1]) + (f_val[6] - f_val[2]) + (f_val[7] - f_val[3])) / 4 / hz;
    		  engtmp += ch[n].epn2 * (f_ux * f_ux + f_uy * f_uy + f_uz * f_uz) / 2;
          engtmp += ((ch[n].c * ch[n].c - 1) * (ch[n].c * ch[n].c - 1) / 4);
        }
      }
    }

    engtmp = engtmp * hx * hy * hz;
    // communicate along all procs
    MPI_Reduce (&maxtmp, &mmax, 1, MPI_DOUBLE, MPI_MAX, prank, MPI_COMM_WORLD);
    MPI_Reduce (&mintmp, &mmin, 1, MPI_DOUBLE, MPI_MIN, prank, MPI_COMM_WORLD);
    MPI_Reduce (&voltmp, &vol, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);
    MPI_Reduce (&engtmp, &eng, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);

    if (myrank == prank) {
      printf ("ch[%d].fieldCI\n", n);
      printf ("max phi\t\t%+1.15lf\n", mmax);
      printf ("min phi\t\t%+1.15lf\n", mmin);
#if 0
      printf ("vol\t\t%+1.15lf\n", vol);
      printf ("eng\t\t%+1.15lf\n", eng);
      fprintf (fp, "%d,%+1.15lf,%+1.15lf,%+1.15lf,%+1.15lf\n", iter, mmax, mmin, vol, eng);
#endif
      fclose (fp); 
    }
  }

  // couple function write to csv file
  if (nac > 0 && nch > 0) {
    if (myrank == prank) {
      sprintf (fname, "%sphi_couple.csv", work_dir);
      fp = fopen (fname, "a");
      if (fp == NULL) {
        printf ("fopen error %s!\n", strerror(errno));
        exit (1);
      }
    }   
    // calculate energy
    for (k = iz2; k < iz3 - 1; k++) {
      for (j = iy2; j < iy3 - 1; j++) {
        for (i = ix2; i < ix3 - 1; i++) {
		      for (n = 0; n < nch; n++) {
            f_val[0] = ac[n].fieldE[k * nx * ny + j * nx + i];
            f_val[1] = ac[n].fieldE[k * nx * ny + j * nx + i + 1];
            f_val[2] = ac[n].fieldE[k * nx * ny + (j + 1) * nx + i];
            f_val[3] = ac[n].fieldE[k * nx * ny + (j + 1) * nx + i + 1];
            f_val[4] = ac[n].fieldE[(k + 1) * nx * ny + j * nx + i];
            f_val[5] = ac[n].fieldE[(k + 1) * nx * ny + j * nx + i + 1];
            f_val[6] = ac[n].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i];
            f_val[7] = ac[n].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i + 1];
            ac[n].u = (f_val[0] + f_val[1] + f_val[2] + f_val[3] + f_val[4] + f_val[5] + f_val[6] + f_val[7]) / 8;
            f_ux = ((f_val[1] - f_val[0]) + (f_val[3] - f_val[2]) + (f_val[5] - f_val[4]) + (f_val[7] - f_val[6])) / 4 / hx;
            f_uy = ((f_val[2] - f_val[0]) + (f_val[3] - f_val[1]) + (f_val[6] - f_val[4]) + (f_val[7] - f_val[5])) / 4 / hy;
            f_uz = ((f_val[4] - f_val[0]) + (f_val[5] - f_val[1]) + (f_val[6] - f_val[2]) + (f_val[7] - f_val[3])) / 4 / hz;
            engtmp +=  ac[n].epn2 * (f_ux * f_ux + f_uy * f_uy + f_uz * f_uz) / 2;
          }
		  for (n = 0; n < nch; n++) {
            f_val[0] = ch[n].fieldCI[k * nx * ny + j * nx + i];
            f_val[1] = ch[n].fieldCI[k * nx * ny + j * nx + i + 1];
            f_val[2] = ch[n].fieldCI[k * nx * ny + (j + 1) * nx + i];
            f_val[3] = ch[n].fieldCI[k * nx * ny + (j + 1) * nx + i + 1];
            f_val[4] = ch[n].fieldCI[(k + 1) * nx * ny + j * nx + i];
            f_val[5] = ch[n].fieldCI[(k + 1) * nx * ny + j * nx + i + 1];
            f_val[6] = ch[n].fieldCI[(k + 1) * nx * ny + (j + 1) * nx + i];
            f_val[7] = ch[n].fieldCI[(k + 1) * nx * ny + (j + 1) * nx + i + 1];
            ch[n].c = (f_val[0] + f_val[1] + f_val[2] + f_val[3] + f_val[4] + f_val[5] + f_val[6] + f_val[7]) / 8;
            f_ux = ((f_val[1] - f_val[0]) + (f_val[3] - f_val[2]) + (f_val[5] - f_val[4]) + (f_val[7] - f_val[6])) / 4 / hx;
            f_uy = ((f_val[2] - f_val[0]) + (f_val[3] - f_val[1]) + (f_val[6] - f_val[4]) + (f_val[7] - f_val[5])) / 4 / hy;
            f_uz = ((f_val[4] - f_val[0]) + (f_val[5] - f_val[1]) + (f_val[6] - f_val[2]) + (f_val[7] - f_val[3])) / 4 / hz;
            engtmp += ac[n].epn2 * (f_ux * f_ux + f_uy * f_uy + f_uz * f_uz) / 2;
          }
    		  engtmp += EF();
        }
      }
	  }
    //engtmp = engtmp * hx * hy * hz;
    MPI_Reduce (&engtmp, &eng, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);
	  if (myrank == prank) {
#if 0
      printf ("eng\t\t%+1.15lf\n", eng);
      fprintf (fp, "%d,%+1.15lf\n", iter, eng);
#endif
      fclose (fp); 
    }
  }

  // every 100 steps write one time section and output all data
  //if (restart == 1 && iter % 1 == 0) {
  if (iter % 1 == 0) {
    outputfield ();
//   for (n = 0; n < nac; n++) {
//     sprintf (fname, "%siter_%d_ac_%d_fieldE_%d%d%d.txt", "./output_x/", iter, n, cart_id[0], cart_id[1], cart_id[2]);
//     write_section_x (fname, ac[n].fieldE);
//     sprintf (fname, "%siter_%d_ac_%d_fieldE_%d%d%d.txt", "./output_z/", iter, n, cart_id[0], cart_id[1], cart_id[2]);
//     write_section_z (fname, ac[n].fieldE);
//     sprintf (fname, "%siter_%d_ac_%d_fieldE_%d%d%d.txt", "./output_y/", iter, n, cart_id[0], cart_id[1], cart_id[2]);
//     write_section_y (fname, ac[n].fieldE);
//   }
//   for (n = 0; n < nch; n++) {
//     sprintf (fname, "%siter_%d_ch_%d_fieldCI_%d%d%d.txt", "./output_x/", iter, n, cart_id[0], cart_id[1], cart_id[2]);
//     write_section_x (fname, ch[n].fieldCI);
//     sprintf (fname, "%siter_%d_ch_%d_fieldCI_%d%d%d.txt", "./output_z/", iter, n, cart_id[0], cart_id[1], cart_id[2]);
//     write_section_z (fname, ch[n].fieldCI);
//     sprintf (fname, "%siter_%d_ch_%d_fieldCI_%d%d%d.txt", "./output_y/", iter, n, cart_id[0], cart_id[1], cart_id[2]);
//     write_section_y (fname, ch[n].fieldCI);
//   }

  }
#if 0
  if (iter % 100 == 0) {
    for (n = 0; n < nac; n++) {
      sprintf (fname, "%sfield_iter%d_ac%d_%d%d%d.txt", restart_dir, iter, n, cart_id[0], cart_id[1], cart_id[2]);
      write_field (fname, ac[n].fieldE);
    }
    for (n = 0; n < nch; n++) {
      sprintf (fname, "%sfield_iter%d_ch%d_%d%d%d.txt", restart_dir, iter, n, cart_id[0], cart_id[1], cart_id[2]);
      write_field (fname, ch[n].fieldCI);
    }
  }
#endif
}

// init field variable
void
init_field() {
  int n;
  char fname[1024];
  if (restart == 1) {
    for (n = 0; n < nac; n++) {
      sprintf (fname, "%sfield_iter%d_ac%d_%d%d%d.txt", restart_dir, restart_iter, n, cart_id[0], cart_id[1], cart_id[2]);
      read_field (fname, ac[n].fieldE);
    }
    for (n = 0; n < nch; n++) {
      sprintf (fname, "%sfield_iter%d_ch%d_%d%d%d.txt", restart_dir, restart_iter, n, cart_id[0], cart_id[1], cart_id[2]);
      read_field (fname, ch[n].fieldCI);
    }
  } else {
    if (nch == 0) {
      for (n = 0; n < nac; n++) {
        init_field_sphere(ac[n].fieldE, sqrt(ch[n].epn2));
      }
    }
    if (nac == 0) {
      for (n = 0; n < nch; n++) {
        init_field_cube(ch[n].fieldCI);
      }
    }
    if (nac > 0 && nch > 0) {
      couple_init_field();
    }
  }
}

