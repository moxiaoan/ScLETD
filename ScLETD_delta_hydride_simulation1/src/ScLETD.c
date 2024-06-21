#include <stdio.h>
#include "mpi.h"
#include "ScLETD.h"
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include "anisotropic_hip.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

int
main (int argc, char **argv)
{
  int n;
  double tt;
  char filename[1024];
  FILE *file;

  // MPI procs start
  start_mpi (argc, argv);
  // read input parameters from input%numProcs.ini
  sprintf (filename, "input%s.ini", argv[1]);
  file = fopen (filename, "r");
  // nx, ny, nz of each proc, including nghost
  fscanf (file, "%d,%d,%d", &nx, &ny, &nz);
  // number of procs in each dimension in mpi
  fscanf (file, "%d,%d,%d", &procs[0], &procs[1], &procs[2]);
  // number of ghosts of subdomain
  fscanf (file, "%d", &nghost);
  // output directory
  fscanf (file, "%s", work_dir);
  // time step and total time 
  fscanf (file, "%lf,%lf", &dt, &t_total);
  // if function is anisotropic, 0 for no and 1 for yes
  fscanf(file, "%d", &ANISOTROPIC);
  // anisotropic auxiliary parameter
  fscanf (file, "%lf,%lf,%lf", &kkx, &kky, &kkz);
  // boundary is periodic or not
  fscanf (file, "%d", &periodic);
  // number of Allen-Cahn functions and Cahn-Hilliard functions
  fscanf (file, "%d, %d", &nac, &nch);
  // if restart, restart from which step, read from checkpoint directory
  fscanf (file, "%d, %d, %s", &restart, &restart_iter, restart_dir);
  // range of simulation area
  fscanf (file, "%lf,%lf,%lf", &xmin, &ymin, &zmin);
  fscanf (file, "%lf,%lf,%lf", &xmax, &ymax, &zmax);
  // 1 order(ETD2=0) of 2 order ETD (ETD2=1)
  fscanf (file, "%d", &ETD2);
  // ELASTIC = 0 means no elastic, ELASTIC = 1 means with elastic
  fscanf(file, "%d", &ELASTIC);
  fscanf(file, "%d", &Approx);
  fclose (file);

  // print parameters to check
  if (myrank == prank) {
    printf ("running on\t%d processors\n", nprocs);
    printf ("nx,ny,nz\t%d\t%d\t%d\n", nx, ny, nz);
    printf ("px,py,pz\t%d\t%d\t%d\n", procs[0], procs[1], procs[2]);
    printf ("nghost\t\t%d\n", nghost);
    printf ("ini file\t%s\n", filename);
    printf ("output dir\t%s\n", work_dir);
    printf ("periodic\t\t%d\n", periodic);
    printf ("dt\t\t%lf\n", dt);
    printf ("nac,nch\t\t%d,%d\n", nac,nch);
    printf ("restart, iter, dir\t%d,%d,%s\n", restart, restart_iter, restart_dir);
    printf ("ETD2\t\t%d\n", ETD2);
    printf ("ELASTIC\t\t%d\n", ELASTIC);
    printf ("Approx\t\t%d\n", Approx);
  }

  // if no function, no work
  if (nac == 0 && nch == 0) {
    printf ("nac = 0 and nch = 0\n");
    exit(1);
  }

  // allocate memory to variables
  alloc_vars ();
  // initialize variables
  init_vars ();
  // read Cijkl and epsilon for elastic
  if (ELASTIC == 1) {
    elastic_read_input();
  }
  // read anisotrpic tensor from file
  if (ANISOTROPIC == 1)
  {
    anisotropic_input();
  }
  // define data of ghost for mpi communication
  define_mpi_type ();
  // create mpi 3d topology
  sw_cart_creat ();
  // initialize parameters
  init_para ();
  // read matrix of fourier and laplace operator
  read_matrices ();
  // initialize tensor of field variable
  init_field ();
  // number of steps
  iter = 0;

  // initialize elastic
  if (ELASTIC == 1) {
  //  fft_setup();
    conv_init_transfer();
    elastic_calculate_BN();
  //  elastic_copyin();
  //  elastic_calculate_ElasDri();
  //  elastic_copyout();
    elastic_calculate();
    elastic_transfer();
    hipMemcpy (Elas, ac[0].felas, sizeof (Dtype) * offset, hipMemcpyHostToDevice);
  }

  // time
  tt = iter*dt;
  // step of write field variable to file
  ioutput = 0;
  while (tt < t_total) {
    // check value of field variable and volume
    check_soln_new (tt);
    // step increase
    iter++;
    // time increase
    tt += dt;

    // transfer data of ghost
    transfer ();

    // used for 2 order laplace
    calc_mu();

    // calculate elastic
    if (ELASTIC == 1) 
    {
   //   elastic_copyin();
   //   elastic_calculate_ElasDri();
   //   elastic_copyout();
      elastic_calculate();
      elastic_transfer();
      hipMemcpy (Elas, ac[0].felas, sizeof (Dtype) * offset, hipMemcpyHostToDevice);
    }

    // calculate c_alpha and c_delta for KKS model
    calc_c_alpha_delta();

    // calculate nonlinear term of function
    for (n = 0; n < nac; n++) {
      ac_calc_FU (n, ac[n].fieldE1);
    }
    for (n = 0; n < nch; n++) {
      ch_calc_FU (n, ch[n].fieldCI1);
    }

    // calculate linear term and 1 order update eta
    for (n = 0; n < nac; n++) {
      stage = 0;
      PUX (MPXI, ac[n].fieldE, ac[n].fieldEt, ac[n].fieldEp);
      PUY (MPYI, ac[n].fieldEt, ac[n].fieldE, ac[n].fieldEp);
      PUZ (MPZI, ac[n].fieldE, ac[n].fieldEt, ac[n].fieldEp, ac[n].fieldEt);

      stage = 1;
      PUX (MPXI, ac[n].fieldE1, ac[n].fieldE, ac[n].fieldEp);
      PUY (MPYI, ac[n].fieldE, ac[n].fieldE1p, ac[n].fieldEp);
      PUZ (MPZI, ac[n].fieldE1p, ac[n].fieldE, ac[n].fieldEp, ac[n].fieldEt);
      ac_updateU_new (n, ac[n].fieldEt, ac[n].fieldEp);
      zxy_xyz (ac[n].fieldEt, ac[n].fieldE);
      stage = 2;
      PUX (MPX, ac[n].fieldE, ac[n].fieldE1p, ac[n].fieldEp);
      PUY (MPY, ac[n].fieldE1p, ac[n].fieldEt, ac[n].fieldEp);
      PUZ (MPZ, ac[n].fieldEt, ac[n].fieldE, ac[n].fieldEp, ac[n].fieldEt);
    }

    // calculate linear term and 1 order update c
    for (n = 0; n < nch; n++) {
      stage = 0;
      PUX (MPXI, ch[n].fieldCI, ch[n].fieldCIt, ch[n].fieldCIp);
      PUY (MPYI, ch[n].fieldCIt, ch[n].fieldCI, ch[n].fieldCIp);
      PUZ (MPZI, ch[n].fieldCI, ch[n].fieldCIt, ch[n].fieldCIp, ch[n].fieldCIt);
      stage = 1;
      PUX (MPXI, ch[n].fieldCI1, ch[n].fieldCI, ch[n].fieldCIp);
      PUY (MPYI, ch[n].fieldCI, ch[n].fieldCI1p, ch[n].fieldCIp);
      PUZ (MPZI, ch[n].fieldCI1p, ch[n].fieldCI, ch[n].fieldCIp, ch[n].fieldCIt);
      ch_updateU_new (n, ch[n].fieldCIt, ch[n].fieldCIp);
      zxy_xyz (ch[n].fieldCIt, ch[n].fieldCI);
      stage = 2;
      PUX (MPX, ch[n].fieldCI, ch[n].fieldCI1p, ch[n].fieldCIp);
      PUY (MPY, ch[n].fieldCI1p, ch[n].fieldCIt, ch[n].fieldCIp);
      PUZ (MPZ, ch[n].fieldCIt, ch[n].fieldCI, ch[n].fieldCIp, ch[n].fieldCIt);
    }
    // if ETD2, repeat
    if (ETD2 == 1) {

      // transfer data of ghost
      transfer ();

      // used for 2 order laplace
      calc_mu();

      // calculate elastic
      if (ELASTIC == 1) 
      {
//       elastic_copyin();
//       elastic_calculate_ElasDri();
//       elastic_copyout();
        elastic_calculate();
        elastic_transfer();
	hipMemcpy (Elas, ac[0].felas, sizeof (Dtype) * offset, hipMemcpyHostToDevice);
      }

      // calculate c_alpha and c_delta for KKS model
      calc_c_alpha_delta();


      // calculate nonlinear term of function
      for (n = 0; n < nac; n++) {
        ac_calc_FU (n, ac[n].fieldE2);
      }
      // calculate nonlinear term of function
      for (n = 0; n < nch; n++) {
        ch_calc_FU (n, ch[n].fieldCI2);
      }
      // calculate linear term and 2 order update eta
      for (n = 0; n < nac; n++) {
        prepare_U1_new (ac[n].fieldE1, ac[n].fieldE2);
        stage = 0;
        PUX (MPXI, ac[n].fieldE2, ac[n].fieldEt, ac[n].fieldEp);
        PUY (MPYI, ac[n].fieldEt, ac[n].fieldE1p, ac[n].fieldEp);
        PUZ (MPZI, ac[n].fieldE1p, ac[n].fieldE2, ac[n].fieldEp, ac[n].fieldEt);
        prepare_U2_new (ac[n].phiE2, ac[n].fieldE1, ac[n].fieldE2);
        zxy_xyz (ac[n].fieldE2, ac[n].fieldE1);
        stage = 2;
        PUX (MPX, ac[n].fieldE1, ac[n].fieldE1p, ac[n].fieldEp);
        PUY (MPY, ac[n].fieldE1p, ac[n].fieldEt, ac[n].fieldEp);
        PUZ (MPZ, ac[n].fieldEt, ac[n].fieldE1, ac[n].fieldEp, ac[n].fieldEt);
        correct_U_new (ac[n].fieldE, ac[n].fieldE1);
      }
      // calculate linear term and 2 order update c
      for (n = 0; n < nch; n++) {
        prepare_U1_new (ch[n].fieldCI1, ch[n].fieldCI2);
        stage = 0;
        PUX (MPXI, ch[n].fieldCI2, ch[n].fieldCIt, ch[n].fieldCIp);
        PUY (MPYI, ch[n].fieldCIt, ch[n].fieldCI1p, ch[n].fieldCIp);
        PUZ (MPZI, ch[n].fieldCI1p, ch[n].fieldCI2, ch[n].fieldCIp, ch[n].fieldCIt);
        prepare_U2_new (ch[n].phiCI2, ch[n].fieldCI1, ch[n].fieldCI2);
        zxy_xyz (ch[n].fieldCI2, ch[n].fieldCI1);
        stage = 2;
        PUX (MPX, ch[n].fieldCI1, ch[n].fieldCI1p, ch[n].fieldCIp);
        PUY (MPY, ch[n].fieldCI1p, ch[n].fieldCIt, ch[n].fieldCIp);
        PUZ (MPZ, ch[n].fieldCIt, ch[n].fieldCI1, ch[n].fieldCIp, ch[n].fieldCIt);
        correct_U_new (ch[n].fieldCI, ch[n].fieldCI1);
      }
    }
    // should barrier afeter each time step
    MPI_Barrier (MPI_COMM_WORLD);
  }

  // free fft memory after all compute
  if (ELASTIC == 1) {
//    fft_finish();
  }
  // free all memory 
  dealloc_vars ();
  // close mpi procs
  close_mpi ();

  return 0;
}
