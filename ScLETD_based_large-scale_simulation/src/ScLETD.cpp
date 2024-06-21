#include "ScLETD.h"

int main(int argc, char **argv)
{
  int n;
  const int detect_iter = 100000;
  double tt;
  char filename[1024];
  FILE *file;
  start_mpi(argc, argv);
  sprintf(filename, "input%s.ini", argv[1]);
  file = fopen(filename, "r");
  fscanf(file, "%d,%d,%d", &nx, &ny, &nz);
  fscanf(file, "%d,%d,%d", &procs[0], &procs[1], &procs[2]);
  fscanf(file, "%d", &nghost);
  fscanf(file, "%s", work_dir);
  fscanf(file, "%s", data_dir);
  fscanf(file, "%lf,%lf", &dt, &t_total);
  //fscanf(file, "%d", &restart);
  fscanf(file, "%d", &periodic);
  fscanf(file, "%d,   %d", &nac, &nch);
  fscanf(file, "%d", &ELASTIC);
  fscanf(file, "%d", &ANISOTROPIC);
  fscanf(file, "%lf,%lf,%lf", &xmin, &ymin, &zmin);
  fscanf(file, "%lf,%lf,%lf", &xmax, &ymax, &zmax);
  fscanf(file, "%lf,%lf,%lf", &kkx, &kky, &kkz);
  fscanf(file, "%d,%d,%d", &checkpoint_, &nchk, &chk);
  fscanf(file, "%d", &nout);
  fscanf(file, "%d", &Approx);
  fclose(file);

  if (myrank == prank)
  {
    printf("running   on\t%d   processors\n", nprocs);
    printf("restart\t\t%d\n", restart);
    printf("nx,ny,nz\t%d\t%d\t%d\n", nx, ny, nz);
    printf("px,py,pz\t%d\t%d\t%d\n", procs[0], procs[1], procs[2]);
    printf("nghost\t\t%d\n", nghost);
    printf("output   dir\t%s\n", work_dir);
    printf("data   dir\t%s\n", data_dir);
    printf("dt,t_total\t%lf,%lf\n", dt, t_total);
    printf("ELASTIC\t\t%d\n", ELASTIC);
    printf("ANISOTROPIC\t\t%d\n", ANISOTROPIC);
    printf("nac,nch\t\t%d,%d\n", nac, nch);
    printf("xmin,ymin,zmin\t%lf\t%lf\t%lf\n", xmin, ymin, zmin);
    printf("xmax,ymax,zmax\t%lf\t%lf\t%lf\n", xmax, ymax, zmax);
    printf("checkpoint_,nchk,chk\t%d,%d,%d\n", checkpoint_, nchk, chk);
    printf("nout\t\t%d\n", nout);
    printf("Approx\t\t%d\n", Approx);
    printf("detect_iter\t\t%d\n", detect_iter);
  }
  if (nac == 0 && nch == 0)
  {
    printf("nac   =   0   and   nch   =   0\n");
    exit(1);
  }
  if (ELASTIC == 1)
  {
    elastic_input();
  }

  alloc_vars();
  init_vars();

  int rank;

  define_mpi_type();
  sw_cart_creat();
  init_para();
  read_matrices();
  if (ELASTIC == 1)
  {
    elastic_init();
  }
  if (checkpoint_ == 1)
  {
    irun = 0;
    chk = 0;
    init_field();
  }
  else if (checkpoint_ == 2)
  {
    read_chk();
    //irun = 4;
    ioutput = 7;
    chk = (chk + 1) % 2;
  }

  if (ANISOTROPIC == 1)
  {
    anisotropic_input();
  }
  iter = 0;
  tt = 0.0;
  ot1 = 0;
  ot2 = 0;
  out_iter = 0;
  out_tt = 0;
  count_gyq = 0;
  double dt_const = dt;
  while (out_tt < (floor(t_total / dt) / 1))
  {
    out_tt += 1;
    ac_calc_F1(iter, detect_iter);
    out_iter++;
  }
  dealloc_vars();
  close_mpi();

  return 0;
}
