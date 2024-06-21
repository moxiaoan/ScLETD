#include "ScLETD.h"
#include "add.h"
//#include "CNN.hpp"

int main(int argc, char **argv)
{
  int n;
  double tt;
  int detect_iter = 100;
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
  fscanf(file, "%s", detect_result_dir);
  fscanf(file, "%lf,%lf", &dt, &t_total);
  fscanf(file, "%lf", &epn2);
  fscanf(file, "%d", &periodic);
  fscanf(file, "%d,   %d", &nac, &nch);
  fscanf(file, "%lf,%lf,%lf", &xmin, &ymin, &zmin);
  fscanf(file, "%lf,%lf,%lf", &xmax, &ymax, &zmax);
  fscanf(file, "%d", &ANISOTROPIC);
  fscanf(file, "%lf,%lf,%lf", &kkx, &kky, &kkz);
  fscanf(file, "%lf", &rad);
  fscanf(file, "%lf", &percent);
  fscanf(file, "%d", &rotation);
  fscanf(file, "%d,%d,%d", &simulate_checkpoint, &nchk, &chk);
  fscanf(file, "%d", &nout);
  fscanf(file, "%d", &counts);
  fscanf(file, "%d", &Approx);
  fclose(file);

  if (myrank == prank)
  {
    printf("running   on\t%d   processors\n", nprocs);
    printf("nx,ny,nz\t%d\t%d\t%d\n", nx, ny, nz);
    printf("px,py,pz\t%d\t%d\t%d\n", procs[0], procs[1], procs[2]);
    printf("nghost\t\t%d\n", nghost);
    printf("output   dir\t%s\n", work_dir);
    printf("data dir\t%s\n", data_dir);
    printf("detect result dir\t%s\n", detect_result_dir);
    printf("epn2\t\t%lf\n", epn2);
    printf("dt,t_total\t%lf,%lf\n", dt, t_total);
    printf("nac,nch\t\t%d,%d\n", nac, nch);
    printf("xmin,ymin,zmin\t%lf\t%lf\t%lf\n", xmin, ymin, zmin);
    printf("xmax,ymax,zmax\t%lf\t%lf\t%lf\n", xmax, ymax, zmax);
    printf("ANISOTROPIC\t\t%d\n", ANISOTROPIC);
    printf("rad\t\t%lf\n", rad);
    printf("percent\t\t%lf\n", percent);
    printf("rotation\t\t%d\n", rotation);
    printf("checkpoint,nchk,chk\t%d,%d,%d\n", simulate_checkpoint, nchk, chk);
    printf("nout\t\t%d\n", nout);
    printf("counts\t\t%d\n", counts);
    printf("Approx\t\t%d\n", Approx);
  }
  if (nac == 0 && nch == 0)
  {
    printf("nac   =   0   and   nch   =   0\n");
    exit(1);
  }
  elastic_input();
  alloc_vars();

  init_vars();

  time_t t;
  srand((unsigned)time(&t) * (myrank + 1));

  define_mpi_type();
  sw_cart_creat();
  init_para();
  read_matrices();
  elastic_init();
  if (simulate_checkpoint == 1)
  {
    irun = 0;
    chk = 0;
  }
  else if (simulate_checkpoint == 2)
  {
    read_chk();
    chk = (chk + 1) % 2;
  }
  //printf("step%dfieldE, myrank = %d, (%d, %d, %d), on %s\n",iter,myrank,cart_id[0],cart_id[1],cart_id[2],processor_name);
  if (ANISOTROPIC == 1)
  {
    anisotropic_input();
  }

  ot2 = 0;
  ot1 = 0;
  ioutput = 1;
  iter = 0;
  tt = 0;
  out_iter = 0;
  out_tt = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  walltime = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);

  while (out_tt < (floor(t_total / dt) / 1))
  // while (out_tt < (floor(t_total / dt) / 1)) //这里是为了跑一步就检测特意写成这样的
  {
    out_tt += 1;
    ac_calc_F1(field2_all, iter, detect_iter);
    out_iter++;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  walltime = MPI_Wtime() - walltime;
  MPI_Reduce(&walltime, &runtime, 1, MPI_DOUBLE, MPI_MAX, prank, MPI_COMM_WORLD);
  if (myrank == prank)
  {
    printf("time\t\t%lf\n", tt);
    printf("wall time\t%lf\n", runtime);
  }
//  check_soln_new(tt);
  dealloc_vars();
  close_mpi();

  return 0;
}
