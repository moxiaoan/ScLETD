#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

int main()
{
	int nx = 132;
	int ny = 132;
	int nz = 132;
	int s1 = 0;
	int s2 = 5;
	double *fieldE;
	double *fieldE1;
	int i, j, k;
	fieldE = (double *)_mm_malloc(sizeof(double) * (nx-4) * (ny-4) * (nz-4) * 8, 256);
	fieldE1 = (double *)_mm_malloc(sizeof(double) * nx * ny * nz, 256);
	char filename[1024];
	FILE *file;

	sprintf (filename, "./data/eta_0000%02d_000000.dat", s2);
	file = fopen (filename, "r");
	for (k = 0; k < nz; k++) {
		for (j = 0; j < ny; j++) {
			for (i = 0; i < nx; i++) {
				fscanf (file, "%lf", &fieldE1[k * nx * ny + j * nx + i]);
        		}
        	}
  	}
  	fclose (file);
	for (k = 2; k < nz-2; k++) {
		for (j = 2; j < ny-2; j++) {
			for (i = 2; i < nx-2; i++) {
				fieldE[(k-2) * (nx-4)*2 * (ny-4)*2 + (j-2) * (nx-4)*2 + i-2] = fieldE1[k * nx * ny + j * nx + i];
        		}
        	}
  	}
	sprintf (filename, "./data/eta_0000%02d_010000.dat", s2);
	file = fopen (filename, "r");
	for (k = 0; k < nz; k++) {
		for (j = 0; j < ny; j++) {
			for (i = 0; i < nx; i++) {
				fscanf (file, "%lf", &fieldE1[k * nx * ny + j * nx + i]);
        		}
        	}
  	}
  	fclose (file);
	for (k = 2; k < nz-2; k++) {
		for (j = 2; j < ny-2; j++) {
			for (i = 2; i < nx-2; i++) {
				fieldE[(k-2) * (nx-4)*2 * (ny-4)*2 + (j-2) * (nx-4)*2 + i-2 + (nx-4)] = fieldE1[k * nx * ny + j * nx + i];
        		}
        	}
  	}
	sprintf (filename, "./data/eta_0000%02d_000100.dat", s2);
	file = fopen (filename, "r");
	for (k = 0; k < nz; k++) {
		for (j = 0; j < ny; j++) {
			for (i = 0; i < nx; i++) {
				fscanf (file, "%lf", &fieldE1[k * nx * ny + j * nx + i]);
        		}
        	}
  	}
  	fclose (file);
	for (k = 2; k < nz-2; k++) {
		for (j = 2; j < ny-2; j++) {
			for (i = 2; i < nx-2; i++) {
				fieldE[(k-2) * (nx-4)*2 * (ny-4)*2 + (j-2+ny-4) * (nx-4)*2 + (i-2)] = fieldE1[k * nx * ny + j * nx + i];
        		}
        	}
  	}
	sprintf (filename, "./data/eta_0000%02d_010100.dat", s2);
	file = fopen (filename, "r");
	for (k = 0; k < nz; k++) {
		for (j = 0; j < ny; j++) {
			for (i = 0; i < nx; i++) {
				fscanf (file, "%lf", &fieldE1[k * nx * ny + j * nx + i]);
        		}
        	}
  	}
  	fclose (file);
	for (k = 2; k < nz-2; k++) {
		for (j = 2; j < ny-2; j++) {
			for (i = 2; i < nx-2; i++) {
				fieldE[(k-2) * (nx-4)*2 * (ny-4)*2 + (j-2+ny-4) * (nx-4)*2 + (i-2)+ nx-4] = fieldE1[k * nx * ny + j * nx + i];
        		}
        	}
  	}

	sprintf (filename, "./data/eta_0000%02d_000001.dat", s2);
	file = fopen (filename, "r");
	for (k = 0; k < nz; k++) {
		for (j = 0; j < ny; j++) {
			for (i = 0; i < nx; i++) {
				fscanf (file, "%lf", &fieldE1[k * nx * ny + j * nx + i]);
        		}
        	}
  	}
  	fclose (file);
	for (k = 2; k < nz-2; k++) {
		for (j = 2; j < ny-2; j++) {
			for (i = 2; i < nx-2; i++) {
				fieldE[(k-2+nz-4) * (nx-4)*2 * (ny-4)*2 + (j-2) * (nx-4)*2 + i-2] = fieldE1[k * nx * ny + j * nx + i];
        		}
        	}
  	}
	sprintf (filename, "./data/eta_0000%02d_010001.dat", s2);
	file = fopen (filename, "r");
	for (k = 0; k < nz; k++) {
		for (j = 0; j < ny; j++) {
			for (i = 0; i < nx; i++) {
				fscanf (file, "%lf", &fieldE1[k * nx * ny + j * nx + i]);
        		}
        	}
  	}
  	fclose (file);
	for (k = 2; k < nz-2; k++) {
		for (j = 2; j < ny-2; j++) {
			for (i = 2; i < nx-2; i++) {
				fieldE[(k-2+nz-4) * (nx-4)*2 * (ny-4)*2 + (j-2) * (nx-4)*2 + i-2+nx-4] = fieldE1[k * nx * ny + j * nx + i];
        		}
        	}
  	}
	sprintf (filename, "./data/eta_0000%02d_000101.dat", s2);
	file = fopen (filename, "r");
	for (k = 0; k < nz; k++) {
		for (j = 0; j < ny; j++) {
			for (i = 0; i < nx; i++) {
				fscanf (file, "%lf", &fieldE1[k * nx * ny + j * nx + i]);
        		}
        	}
  	}
  	fclose (file);
	for (k = 2; k < nz-2; k++) {
		for (j = 2; j < ny-2; j++) {
			for (i = 2; i < nx-2; i++) {
				fieldE[(k-2+nz-4) * (nx-4)*2 * (ny-4)*2 + (j-2+ny-4) * (nx-4)*2 + i-2] = fieldE1[k * nx * ny + j * nx + i];
        		}
        	}
  	}
	sprintf (filename, "./data/eta_0000%02d_010101.dat", s2);
	file = fopen (filename, "r");
	for (k = 0; k < nz; k++) {
		for (j = 0; j < ny; j++) {
			for (i = 0; i < nx; i++) {
				fscanf (file, "%lf", &fieldE1[k * nx * ny + j * nx + i]);
        		}
        	}
  	}
  	fclose (file);
	for (k = 2; k < nz-2; k++) {
		for (j = 2; j < ny-2; j++) {
			for (i = 2; i < nx-2; i++) {
				fieldE[(k-2+nz-4) * (nx-4)*2 * (ny-4)*2 + (j-2+ny-4) * (nx-4)*2 + i-2+nx-4] = fieldE1[k * nx * ny + j * nx + i];
        		}
        	}
  	}

	sprintf (filename, "./data/change/eta_0000%02d.dat", s2);
	file = fopen (filename, "w");
	for (k = 0; k < nz*2-8; k++) {
		for (j = 0; j < ny*2-8; j++) {
			for (i = 0; i < nx*2-8; i++) {
				fprintf (file, "%lf\n", fieldE[k * (nx-4)*2 * (ny-4)*2 + j * (nx-4)*2 + i]);
        		}
        	}
  	}
  	fclose (file);
	_mm_free(fieldE);
	_mm_free(fieldE1);
	return 0;
}
