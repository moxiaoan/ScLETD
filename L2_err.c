#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

int main()
{
        int nx = 68;
        int ny = 68;
        int nz = 68;
        double *fieldE;
        double *fieldE1;
        int i, j, k;
        int s1, s2, s3;
        double err = 0.0;
        double err1 = 0.0;
        fieldE = (double *)_mm_malloc(sizeof(double) * nx * ny * nz, 256);
        fieldE1 = (double *)_mm_malloc(sizeof(double) * nx * ny * nz, 256);
        char filename[1024];
        FILE *file;
        double x, max;
        max = -1000.0;

        for (s1 = 0; s1 < 2; s1++) {
                for (s2 = 0; s2 < 2; s2++) {
                        for (s3 = 0; s3 < 2; s3++) {
                                sprintf (filename, "/public/home/cnicgyq/cov_elastic_applications/kks_cov/data/c_000002_%02d%02d%02d.dat", s3, s2, s1);
                                //sprintf (filename, "./Bn/eta0_000001_0%d0%d0%d", s3, s2, s1);
                                file = fopen (filename, "r");
                                for (k = 0; k < nz; k++) {
                                        for (j = 0; j < ny; j++) {
                                                for (i = 0; i < nx; i++) {
                                                        fscanf (file, "%lf", &fieldE[k * nx * ny + j * nx + i]);
                                                }
                                        }
                                }
                                fclose (file);
                                sprintf (filename, "/public/home/cnicgyq/cov_elastic_applications/kks/data/c_000002_%02d%02d%02d.dat", s3, s2, s1);

                                file = fopen (filename, "r");
                                for (k = 0; k < nz; k++) {
                                        for (j = 0; j < ny; j++) {
                                                for (i = 0; i < nx; i++) {
                                                        fscanf (file, "%lf", &fieldE1[k * nx * ny + j * nx + i]);
                                                }
                                        }
                                }
                                fclose (file);

                                for (k = 0; k < nz; k++) {
                                        for (j = 0; j < ny; j++) {
                                                for (i = 0; i < nx; i++) {
                                                        double f1, f2;
                                                        f1 = fieldE[k * nx * ny + j * nx + i];
                                                        f2 = fieldE1[k * nx * ny + j * nx + i];
                                                        err += (f1 - f2) * (f1 - f2);
                                                        err1 += f1 * f1;
                                                        x = fabs(f1-f2);
                                                        if(x > max)
                                                        {
                                                                max = x;
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
        printf ("relative err is %+1.15lf\n", sqrt(err)/sqrt(err1));
        printf ("max err is %+1.15lf\n", max);
        //printf ("err is %+1.15lf\n", sqrt(err));

        _mm_free(fieldE);
        _mm_free(fieldE1);
        return 0;
}
