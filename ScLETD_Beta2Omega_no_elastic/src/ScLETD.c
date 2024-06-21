#include   "ScLETD.h"

int   main(int   argc,   char   **argv)
{
      int   n;
      double   tt;
      char   filename[1024];
      FILE   *file;
      start_mpi(argc,   argv);
      sprintf(filename,   "input%s.ini",   argv[1]);
      file   =   fopen(filename,   "r");
      fscanf(file,   "%d,%d,%d",   &nx,   &ny,   &nz);
      fscanf(file,   "%d,%d,%d",   &procs[0],   &procs[1],   &procs[2]);
      fscanf(file,   "%d",   &nghost);
      fscanf(file,   "%s",   work_dir);
      fscanf(file,   "%s",   data_dir);
      fscanf(file,   "%lf,%lf",   &dt,   &t_total);
      fscanf(file,   "%lf",   &epn2);
      fscanf(file,   "%d",   &periodic);
      fscanf(file,   "%d,   %d",   &nac,   &nch);
      fscanf(file,   "%lf,%lf,%lf",   &xmin,   &ymin,   &zmin);
      fscanf(file,   "%lf,%lf,%lf",   &xmax,   &ymax,   &zmax);
      fscanf(file,   "%d",   &ANISOTROPIC);
      fscanf(file,   "%d",   &ELASTIC);
      fscanf(file,   "%lf,%lf,%lf",   &kkx,   &kky,   &kkz);
      fscanf(file,   "%lf",   &rad);
      fscanf(file,   "%lf",   &percent);
      fscanf(file,   "%d",   &rotation);
      fscanf(file,   "%d,%d,%d",  &checkpoint, &nchk, &chk);
      fscanf(file,   "%d",  &nout);
      fscanf(file,   "%d",  &counts);
      fscanf(file,   "%d,%d,%d",   &elas_x,   &elas_y,   &elas_z);
      fclose(file);


      if   (myrank   ==   prank)
      {
            printf("running   on\t%d   processors\n",   nprocs);
            printf("nx,ny,nz\t%d\t%d\t%d\n",   nx,   ny,   nz);
            printf("px,py,pz\t%d\t%d\t%d\n",   procs[0],   procs[1],   procs[2]);
            printf("nghost\t\t%d\n",   nghost);
            printf("output   dir\t%s\n",   work_dir);
            printf("data   dir\t%s\n",   data_dir);
            printf("epn2\t\t%lf\n",   epn2);
            printf("dt,t_total\t%lf,%lf\n",   dt,  t_total);
            printf("nac,nch\t\t%d,%d\n",   nac,   nch);
            printf("xmin,ymin,zmin\t%lf\t%lf\t%lf\n",   xmin,   ymin,   zmin);
            printf("xmax,ymax,zmax\t%lf\t%lf\t%lf\n",   xmax,   ymax,   zmax);
            printf("ANISOTROPIC\t\t%d\n",   ANISOTROPIC);
            printf("ELASTIC\t\t%d\n",   ELASTIC);
            printf("rad\t\t%lf\n",   rad);
            printf("percent\t\t%lf\n",   percent);
            printf("rotation\t\t%d\n",   rotation);
            printf("checkpoint,nchk,chk\t%d,%d,%d\n",   checkpoint,  nchk,  chk);
            printf("nout\t\t%d\n",   nout);
            printf("counts\t\t%d\n",   counts);
      }
      if   (nac   ==   0   &&   nch   ==   0)
      {
            printf("nac   =   0   and   nch   =   0\n");
            exit(1);
      }
      if   (ELASTIC == 1)
      {
	   elastic_input();
      }         
      alloc_vars();
//      elastic_malloc();
      init_vars();
      time_t t;
      srand((unsigned)time(&t) * (myrank + 1));
      
      define_mpi_type();
      sw_cart_creat();
      init_para();
      read_matrices();
      if   (ELASTIC == 1)
      {
           elastic_init();
      }
      if (checkpoint == 1) {
        irun = 0;
        chk = 0;
      }
      else if (checkpoint == 2) {
        read_chk ();
        chk = (chk + 1) % 2;
      }
      if   (ANISOTROPIC   ==   1)
      {
            anisotropic_input();
      }
      
      ac_calc_F1();
      if   (ELASTIC == 1)
      {
           //elastic_finish(); 
           fft_finish(); 
      }
//           elastic_finish(); 
      dealloc_vars();
      close_mpi();

      return   0;
}
