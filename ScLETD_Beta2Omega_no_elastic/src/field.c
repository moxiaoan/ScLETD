#include "ScLETD.h"
#include<sys/types.h>
#include<unistd.h>
#include<limits.h>
#include<sys/stat.h>


/*
void
outputfield ()
{
  int i, j, k, n;
  char filename[1024];
  FILE *file;

for (n = 0; n < nac; n++) 
{
  //sprintf (filename, "%seta%d_%06d_%02d%02d%02d.dat", data_dir, n, ioutput, cart_id[0], cart_id[1], cart_id[2]);
  sprintf (filename, "%s%02d/eta%d_%06d_%02d%02d%02d.dat", data_dir, cart_id[2], n, ioutput, cart_id[0], cart_id[1], cart_id[2]);
  file = fopen (filename, "wb");
  //file = fopen (filename, "w");
  //if (file == NULL) {
   // printf ("fopen error %s!\n", strerror(errno));
   // exit (1);
  //}
  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        fwrite (&ac[n].fieldE[k * nx * ny + j * nx + i], sizeof (double), 1, file);
        //fprintf (file, "%+1.15lf\n", ac[n].fieldE[k * nx * ny + j * nx + i]);
      }
    }
  }
  fclose (file);
}
for (n = 0; n < nch; n++) 
{
  //sprintf (filename, "%sc%d_%06d_%02d%02d%02d.dat", data_dir, n, ioutput, cart_id[0], cart_id[1], cart_id[2]);
  sprintf (filename, "%s%02d/c%d_%06d_%02d%02d%02d.dat", data_dir, cart_id[2], n, ioutput, cart_id[0], cart_id[1], cart_id[2]);
 // file = fopen (filename, "wb");
  file = fopen (filename, "w");

  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
        //fwrite (&ch[n].fieldCI[k * nx * ny + j * nx + i], sizeof (double), 1, file);
        fprintf (file, "%+1.15lf\n", ch[n].fieldCI[k * nx * ny + j * nx + i]);
      }
    }
  }
  fclose (file);
}

  ioutput+=1;
}
*/

void
write_section (char *fname, double *field)
{
  FILE *fp;
  int i, j, k;
  if (cart_id[2] == (procs[2]/2)) {
    fp = fopen (fname, "w");
    if (fp == NULL) {
      printf ("fopen error %s!\n", strerror(errno));
      exit (1);
    }
    if (procs[2] == 1) {
      k = nz / 2;
    } else {
      k = 2;
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
  for (k = nghost; k < nz - nghost; k++) {
    for (j = nghost; j < ny - nghost; j++) {
      for (i = nghost; i < nx - nghost; i++) {
        fscanf (fp, "%lf", &field[k * nx * ny + j * nx + i]);
      }
    }
  }
  fclose (fp);
}

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
  for (k = nghost; k < nz-nghost; k++) {
    for (j = nghost; j < ny-nghost; j++) {
      for (i = nghost; i < nx-nghost; i++) {
        fprintf (fp, "%+1.15lf\n", field[k * nx * ny + j * nx + i]);
      }
    }
  }
  fclose (fp);
}
/*
void
init_field_check (void)
{
  int n;
  int l;
  int i, j, k;
  int rad1;
  int cnt1;
  int x, y, z;
  cnt1 = nx * procs[0] / 2;
  rad1 = nx * procs[0] / 6;

  for (n = 0; n < nac; n++) {
    for (l = 0; l < nx * ny * nz; l++) {
        ac[n].fieldE[l] = 1.0;
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
            ac[n].fieldE[k * nx * ny + j * nx + i] = -1.0;
          }
        }
      }
    }
  }
  for (n = 0; n < nch; n++) {
    for (l = 0; l < nx * ny * nz; l++) {
        ch[n].fieldCI[l] = -1.0;
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
            ch[n].fieldCI[k * nx * ny + j * nx + i] = 1.0;
          }
        }
      }
    }
  }
}


void 
couple_init_field_two_sphere ()
{
  int l;
  int i, j, k;
  double x, y, z;
  double c1[3], c2[3];
  double rad1, rad2;
  double d1, d2;
  // two sphere in y direction
  c1[0] = xmin + (xmax - xmin) / 2.0;
  c1[1] = ymin + (ymax - ymin) / 3.0;
  c1[2] = zmin + (zmax - zmin) / 2.0;
  c2[0] = xmin + (xmax - xmin) / 2.0;
  c2[1] = ymax - (ymax - ymin) / 3.0;
  c2[2] = zmin + (zmax - zmin) / 2.0;
  rad1 = rad2 = (xmax - xmin) / 5.0;
  for (l = 0; l < nx * ny * nz; l++) {
    ac[0].fieldE[l] = 1.4;
    ch[0].fieldCI[l] = 0.24;
  }

  for (i = ix1; i < ix4; i++) {
    for (j = iy1; j < iy4; j++) {
      for (k = iz1; k < iz4; k++) {
        x = xmin + fieldgx[k * ny * nx + j * nx + i] * hx;
        y = ymin + fieldgy[k * ny * nx + j * nx + i] * hy;
        z = zmin + fieldgz[k * ny * nx + j * nx + i] * hz;
        d1 = sqrt((x-c1[0]) * (x-c1[0]) + (y-c1[1]) * (y-c1[1]) + (z-c1[2])*(z-c1[2]));
        d2 = sqrt((x-c2[0]) * (x-c2[0]) + (y-c2[1]) * (y-c2[1]) + (z-c2[2])*(z-c2[2]));
        if ((d1 < rad1) || (d2 < rad2)) {
          ac[0].fieldE[k * nx * ny + j * nx + i] = 0.0;
          ch[0].fieldCI[k * nx * ny + j * nx + i] = 0.65;
        }
      }
    }
  }
}


void
couple_init_field_cube ()
{
  int n;
  int l;
  int i, j, k;
  int rad1;
  int cnt1;
  int x, y, z;
  cnt1 = nx * procs[0] / 2;
  rad1 = 8;
  for (n = 0; n < nac; n++) {
    for (l = 0; l < nx * ny * nz; l++) {
      ac[n].fieldE[l] = 0.0;
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
            ac[n].fieldE[k * nx * ny + j * nx + i] = 1.4;
          }
        }
      }
    }
  }
  for (n = 0; n < nch; n++) {
    for (l = 0; l < nx * ny * nz; l++) {
      ch[n].fieldCI[l] = 0.65;
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
            ch[n].fieldCI[k * nx * ny + j * nx + i] = 0.3;
          }
        }
      }
    }
  }
}

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
    field[l] = 0.0;
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

void
init_field_sphere (double *field)
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
        field[k * nx * ny + j * nx + i] = tanh ((0.4 - sqrt(x * x + y * y + z * z)) / sqrt (2.0) / sqrt(epn2));
      }
    }
  }
}
*/
void 
copy(void)
{	
	   int n, i, j, k;
	  if (iter > 0){
	   for (n = 0; n < nac; n++){
	    for (k = 0; k < nz; k++){
	      for (j = 0; j < ny; j++){
	  	for (i = 0; i < nx; i++){
		    ac[n].fieldE[k * nx * ny + j * nx + i] = ac[n].fielde[k * nx * ny + j * nx + i];
		}
              }
	    }
	   }
	  }


}

void print_info(void)
{
     int buf[1];
     MPI_Status status1;
     buf[0] = 1;
     if ( myrank>0) {
         MPI_Recv(buf, 1, MPI_INT, myrank-1, 0, MPI_COMM_WORLD, &status1);
     }
      
     printf("step%d, myrank = %d, (%d, %d, %d), on %s\n",iter,myrank,cart_id[0],cart_id[1],cart_id[2],processor_name);
 
     if ( myrank<nprocs-1) {
         MPI_Send(buf, 1, MPI_INT, myrank+1, 0, MPI_COMM_WORLD);
     }
     MPI_Barrier(MPI_COMM_WORLD);
}
void
check_soln_new (double time)
{
//  if (myrank == prank) {
//    printf ("--------------iter %d--------------\n", iter);
//  }
//  print_info();
  int n;
  int i, j, k;
  double tmp, maxtmp, mintmp, voltmp, engtmp, mmax, mmin, vol, eng;
  double f_val[32], f_ux, f_uy, f_uz;
  FILE *fp;
  char fname[1024];
  if (iter == 0) {
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
    for (n = 0; n < nac; n++) {
      if (myrank == prank) {
        sprintf (fname, "%sphi_ac_%d.csv", work_dir, n);
        fp = fopen (fname, "w");
        if (fp == NULL) {
          printf ("fopen error %s!\n", strerror(errno));
          exit (1);
        }
    //    fprintf (fp, "iter,maxphiE,minphiE,volE,engE\n");
        fprintf (fp, "iter,maxphiE,minphiE,volE\n");
        fclose (fp);
      }
    }
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
/*
    for (k = iz2; k < iz3 - 1; k++) {
      for (j = iy2; j < iy3 - 1; j++) {
        for (i = ix2; i < ix3 - 1; i++) {
          f_val[0] = ac[0].fieldE[k * nx * ny + j * nx + i];
          f_val[1] = ac[0].fieldE[k * nx * ny + j * nx + i + 1];
          f_val[2] = ac[0].fieldE[k * nx * ny + (j + 1) * nx + i];
          f_val[3] = ac[0].fieldE[k * nx * ny + (j + 1) * nx + i + 1];
          f_val[4] = ac[0].fieldE[(k + 1) * nx * ny + j * nx + i];
          f_val[5] = ac[0].fieldE[(k + 1) * nx * ny + j * nx + i + 1];
          f_val[6] = ac[0].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i];
          f_val[7] = ac[0].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i + 1];
	  f_val[8] = ac[1].fieldE[k * nx * ny + j * nx + i]; 
          f_val[9] = ac[1].fieldE[k * nx * ny + j * nx + i + 1]; 
          f_val[10] = ac[1].fieldE[k * nx * ny + (j + 1) * nx + i]; 
          f_val[11] = ac[1].fieldE[k * nx * ny + (j + 1) * nx + i + 1]; 
          f_val[12] = ac[1].fieldE[(k + 1) * nx * ny + j * nx + i]; 
          f_val[13] = ac[1].fieldE[(k + 1) * nx * ny + j * nx + i + 1]; 
          f_val[14] = ac[1].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i]; 
          f_val[15] = ac[1].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i + 1]; 
          f_val[16] = ac[2].fieldE[k * nx * ny + j * nx + i]; 
          f_val[17] = ac[2].fieldE[k * nx * ny + j * nx + i + 1]; 
          f_val[18] = ac[2].fieldE[k * nx * ny + (j + 1) * nx + i]; 
          f_val[19] = ac[2].fieldE[k * nx * ny + (j + 1) * nx + i + 1]; 
          f_val[20] = ac[2].fieldE[(k + 1) * nx * ny + j * nx + i]; 
          f_val[21] = ac[2].fieldE[(k + 1) * nx * ny + j * nx + i + 1]; 
          f_val[22] = ac[2].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i]; 
          f_val[23] = ac[2].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i + 1]; 
          f_val[24] = ac[3].fieldE[k * nx * ny + j * nx + i]; 
          f_val[25] = ac[3].fieldE[k * nx * ny + j * nx + i + 1]; 
          f_val[26] = ac[3].fieldE[k * nx * ny + (j + 1) * nx + i]; 
          f_val[27] = ac[3].fieldE[k * nx * ny + (j + 1) * nx + i + 1]; 
          f_val[28] = ac[3].fieldE[(k + 1) * nx * ny + j * nx + i]; 
          f_val[29] = ac[3].fieldE[(k + 1) * nx * ny + j * nx + i + 1]; 
          f_val[30] = ac[3].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i]; 
          f_val[31] = ac[3].fieldE[(k + 1) * nx * ny + (j + 1) * nx + i + 1];

          ac[0].u = (f_val[0] + f_val[1] + f_val[2] + f_val[3] + f_val[4] + f_val[5] + f_val[6] + f_val[7]) / 8;
          ac[1].u = (f_val[8] + f_val[9] + f_val[10] + f_val[11] + f_val[12] + f_val[13] + f_val[14] + f_val[15]) / 8;
          ac[2].u = (f_val[16] + f_val[17] + f_val[18] + f_val[19] + f_val[20] + f_val[21] + f_val[22] + f_val[23]) / 8;
          ac[3].u = (f_val[24] + f_val[25] + f_val[26] + f_val[27] + f_val[28] + f_val[29] + f_val[30] + f_val[31]) / 8;

//	  f_ux = ((f_val[1] - f_val[0]) + (f_val[3] - f_val[2]) + (f_val[5] - f_val[4]) + (f_val[7] - f_val[6])) / 4 / hx;
//          f_uy = ((f_val[2] - f_val[0]) + (f_val[3] - f_val[1]) + (f_val[6] - f_val[4]) + (f_val[7] - f_val[5])) / 4 / hy;
//          f_uz = ((f_val[4] - f_val[0]) + (f_val[5] - f_val[1]) + (f_val[6] - f_val[2]) + (f_val[7] - f_val[3])) / 4 / hz;

	  f_ux = ((f_val[n * 8 + 1] - f_val[n * 8 + 0]) + (f_val[n * 8 + 3] - f_val[n * 8 + 2]) + (f_val[n * 8 + 5] - f_val[n * 8 + 4]) + (f_val[n * 8 + 7] - f_val[n * 8 + 6])) / 4 / hx;
          f_uy = ((f_val[n * 8 + 2] - f_val[n * 8 + 0]) + (f_val[n * 8 + 3] - f_val[n * 8 + 1]) + (f_val[n * 8 + 6] - f_val[n * 8 + 4]) + (f_val[n * 8 + 7] - f_val[n * 8 + 5])) / 4 / hy;
          f_uz = ((f_val[n * 8 + 4] - f_val[n * 8 + 0]) + (f_val[n * 8 + 5] - f_val[n * 8 + 1]) + (f_val[n * 8 + 6] - f_val[n * 8 + 2]) + (f_val[n * 8 + 7] - f_val[n * 8 + 3])) / 4 / hz;
	  double ff_ux = f_ux * ac[n].lambda_check[0][0] + f_uy * ac[n].lambda_check[1][0] + f_uz * ac[n].lambda_check[2][0];
          double ff_uy = f_ux * ac[n].lambda_check[0][1] + f_uy * ac[n].lambda_check[1][1] + f_uz * ac[n].lambda_check[2][1];
      double ff_uz = f_ux * ac[n].lambda_check[0][2] + f_uy * ac[n].lambda_check[1][2] + f_uz * ac[n].lambda_check[2][2];
	  
	  engtmp += epn2 * (f_ux * f_ux + f_uy * f_uy + f_uz * f_uz) / 2;
          engtmp += (2376.0 / 2372.0 / 2.0) * ac[n].u * ac[n].u - ((7128.0 + 12.0 * 2372.0) / 2372.0 / 3.0) * ac[n].u * ac[n].u * ac[n].u + ((4752.0 + 12.0 * 2372.0) / 2372.0 / 4.0) * ac[n].u * ac[n].u * (ac[0].u * ac[0].u + ac[1].u * ac[1].u + ac[2].u * ac[2].u + ac[3].u * ac[3].u);

	}
      }
    }
    engtmp = engtmp * hx * hy * hz;*/
    MPI_Reduce (&maxtmp, &mmax, 1, MPI_DOUBLE, MPI_MAX, prank, MPI_COMM_WORLD);
    MPI_Reduce (&mintmp, &mmin, 1, MPI_DOUBLE, MPI_MIN, prank, MPI_COMM_WORLD);
    MPI_Reduce (&voltmp, &vol, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);
//    MPI_Reduce (&engtmp, &eng, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);
    if(maxtmp > 2.0){ 
        printf("step%dfieldE, myrank = %d, (%d, %d, %d), on %s, max %lf min %lf\n",iter,myrank,cart_id[0],cart_id[1],cart_id[2],processor_name,maxtmp,mintmp);
    }
//    printf("step%dfieldE, myrank = %d, (%d, %d, %d), on %s\n",iter,myrank,cart_id[0],cart_id[1],cart_id[2],processor_name);
    if (myrank == prank) {
      printf ("--------------iter %d, eta%d--------------\n", iter,n);
    }
    if (myrank == prank) {
      printf ("ac[%d].fieldE\n", n);
      printf ("max phi\t\t%+1.15lf\n", mmax);
      printf ("min phi\t\t%+1.15lf\n", mmin);
      printf ("vol\t\t%+1.15lf\n", vol);
      //printf("step%dfieldE, myrank = %d, on %s\n",iter,myrank,processor_name);
//      printf ("eng\t\t%+1.15lf\n", eng);
//      fprintf (fp, "%d,%+1.15lf,%+1.15lf,%+1.15lf,%+1.15lf\n", iter, mmax, mmin, vol, eng);
      fprintf (fp, "%d,%+1.15lf,%+1.15lf,%+1.15lf\n", iter, mmax, mmin, vol);
      fclose (fp); 
    }
  }

/*  for (n = 0; n < nch; n++) {
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
    		  engtmp += epn2 * (f_ux * f_ux + f_uy * f_uy + f_uz * f_uz) / 2;
          engtmp += ((ch[n].c * ch[n].c - 1) * (ch[n].c * ch[n].c - 1) / 4);
        }
      }
    }
    engtmp = engtmp * hx * hy * hz;
    MPI_Reduce (&maxtmp, &mmax, 1, MPI_DOUBLE, MPI_MAX, prank, MPI_COMM_WORLD);
    MPI_Reduce (&mintmp, &mmin, 1, MPI_DOUBLE, MPI_MIN, prank, MPI_COMM_WORLD);
    MPI_Reduce (&voltmp, &vol, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);
    MPI_Reduce (&engtmp, &eng, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);

    if (myrank == prank) {
      printf ("ch[%d].fieldCI\n", n);
      printf ("max phi\t\t%+1.15lf\n", mmax);
      printf ("min phi\t\t%+1.15lf\n", mmin);
      //printf ("vol\t\t%+1.15lf\n", vol);
      //printf ("eng\t\t%+1.15lf\n", eng);
      fprintf (fp, "%d,%+1.15lf,%+1.15lf,%+1.15lf,%+1.15lf\n", iter, mmax, mmin, vol, eng);
      fclose (fp); 
    }
  }


  if (nac > 0 && nch > 0) {
    if (myrank == prank) {
      sprintf (fname, "%sphi_couple.csv", work_dir);
      fp = fopen (fname, "a");
      if (fp == NULL) {
        printf ("fopen error %s!\n", strerror(errno));
        exit (1);
      }
    }   
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
            engtmp +=  epn2 * (f_ux * f_ux + f_uy * f_uy + f_uz * f_uz) / 2;
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
            engtmp += epn2 * (f_ux * f_ux + f_uy * f_uy + f_uz * f_uz) / 2;
          }
    		  engtmp += EF();
        }
      }
	  }
    engtmp = engtmp * hx * hy * hz;
    MPI_Reduce (&engtmp, &eng, 1, MPI_DOUBLE, MPI_SUM, prank, MPI_COMM_WORLD);
	  if (myrank == prank) {
      printf ("eng\t\t%+1.15lf\n", eng);
      fprintf (fp, "%d,%+1.15lf\n", iter, eng);
      fclose (fp); 
    }
  }
*/

  /*if (iter % 100 == 0) {
    for (n = 0; n < nac; n++) {
      sprintf (fname, "%siter_%d_ac_%d_fieldE_%d%d%d.txt", work_dir, iter, n, cart_id[0], cart_id[1], cart_id[2]);
      write_section (fname, ac[n].fieldE);
    }
    for (n = 0; n < nch; n++) {
      sprintf (fname, "%siter_%d_ch_%d_fieldCI_%d%d%d.txt", work_dir, iter, n, cart_id[0], cart_id[1], cart_id[2]);
      write_section (fname, ch[n].fieldCI);
    }
  }*/
//  if (iter != 0 && iter % 1000 == 0) {
  //  outputfield ();
  /*  for (n = 0; n < nac; n++) {
      sprintf (fname, "%sfield_iter%d_ac%d_%d%d%d.txt", restart_dir, iter, n, cart_id[0], cart_id[1], cart_id[2]);
      write_field (fname, ac[n].fieldE);
    }
    for (n = 0; n < nch; n++) {
      sprintf (fname, "%sfield_iter%d_ch%d_%d%d%d.txt", restart_dir, iter, n, cart_id[0], cart_id[1], cart_id[2]);
      write_field (fname, ch[n].fieldCI);
    }*/
 // }
}
/*
void
couple_init_field_sphere ()
{
  int n;
  int l;
  int i, j, k;
  double x, y, z;
  double cnt1 = (double)procs[0] * nx * hx / 2;
  double rad1 = 6 * hx;
    for (l = 0; l < nx * ny * nz; l++) {
      ac[0].fieldE[l] = 0;
      ch[0].fieldCI[l] = 0.65;
    }
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        for (k = 0; k < nz; k++) {
          x = (cart_id[0] * nx + i) * hx;
          y = (cart_id[1] * ny + j) * hy;
          z = (cart_id[2] * nz + k) * hz;
          if (pow((x - cnt1) * (x - cnt1) + (y - cnt1) * (y - cnt1) + (z - cnt1) * (z - cnt1), 0.5) < rad1){
            ac[0].fieldE[k * nx * ny + j * nx + i] = 1.4;
            ch[0].fieldCI[k * nx * ny + j * nx + i] = 0.3;
          }
        }
      }
    }
}
*/
void myball(int iter, double rad, double *field)
{
  double thr;
  int i, j, k, l;
  int x, y, z;
  int x1, y1, z1;
  if (checkpoint == 1){
    if (iter == 0){
    	for (l = 0; l < nx * ny * nz; l++)
    	{
    		field[l] = 0.0;
    	}
    }
  }
  thr = rand() / (double)(RAND_MAX);
  if (thr < percent)
  {

		x = (int)((rand() / (double)(RAND_MAX)) * (nx - nghost - rad));
		y = (int)((rand() / (double)(RAND_MAX)) * (ny - nghost - rad));
		z = (int)((rand() / (double)(RAND_MAX)) * (nz - nghost - rad));

		for (i = ix1; i < ix4; i++)
		{
			for (j = iy1; j < iy4; j++)
			{
				for (k = iz1; k < iz4; k++)
				{
					if (sqrt((i - x) * (i - x) + (j - y) * (j - y) + (k - z) * (k - z)) < rad)
					{
						field[k * nx * ny + j * nx + i] = 1.0;
					}				
				}
			}

		}

  }
}



void gaussian_serial(double *f)
{
	int i, j, k, l;
	double u1, u2, g1, g2, tp;
  double fc = 0.14;
	l = 0;
  double thr;
	while (l < (nx * ny * nz - 1))
	{
		thr = rand() / (double)(RAND_MAX);
		if (thr < 0.01){
		u1 = rand() / (double)(RAND_MAX);
		u2 = rand() / (double)(RAND_MAX);
//                u1 = 0.02323;            
//                u2 = 0.03434;
 //                u1 = 0.226584;
 //                u2 = 0.055392;
     //           printf("\n--www--u1=%lf, u2= %lf\n",u1,u2);
		tp = sqrt(-2.0 * log(u1));
		g1 = tp * cos(2.0 * PI * u2);
		g2 = tp * sin(2.0 * PI * u2);
                //printf("\n--www--g1=%lf, g2= %lf\n",g1,g2);
	
		f[l++] = g1 * fc;
		f[l++] = g2 * fc;
		}
	};
}


void 
init_field_zero(double *field) {
  int i, j, k;
  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      for (k = 0; k < nz; k++) {
        field[k * nx * ny + j * nx + i] = 0.0;
      }
    }
  }
}
/*
void
init_field() {
  int n;
  char fname[1024];
  if (restart == 1) {
    for (n = 0; n < nac; n++) {
      sprintf (fname, "%sfield_iter%d_ac%d_%d%d%d.txt", "./restart/", restart_iter, n, cart_id[0], cart_id[1], cart_id[2]);
      read_field (fname, ac[n].fieldE);
    }
    for (n = 0; n < nch; n++) {
      sprintf (fname, "%sfield_iter%d_ch%d_%d%d%d.txt", "./restart/", restart_iter, n, cart_id[0], cart_id[1], cart_id[2]);
      read_field (fname, ch[n].fieldCI);
    }
  } else {
    //init_field_check ();
    if (nch == 0) {
      time_t t;
      srand((unsigned)time(&t) * (myrank + 1));
      for (n = 0; n < nac; n++) {
        //init_field_sphere(ac[n].fieldE);
        //init_field_zero(ac[n].fieldE);
        //gaussian_serial(ac[n].f3);
//        myball(iter, ac[n].fieldE);
        //init_field_cube(ac[n].fieldE);
      }
    }
    if (nac == 0) {
      for (n = 0; n < nch; n++) {
        init_field_cube(ch[n].fieldCI);
      }
    }
    if (nac > 0 && nch > 0) {
      couple_init_field_cube();
      //couple_init_field_sphere ();
    }
  }
}
*/

