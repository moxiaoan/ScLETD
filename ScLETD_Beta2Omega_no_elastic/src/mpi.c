#include "ScLETD.h"


extern  void hip_init();
void
start_mpi (int argc, char **argv)
{
  MPI_Init (&argc, &argv);

  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
  MPI_Get_processor_name (processor_name, &namelen);

//  hipSetDevice(myrank%2);
  hip_init();
}


void
close_mpi (void)
{
  MPI_Type_free (&left_right);
  MPI_Type_free (&top_bottom);
  MPI_Type_free (&front_back);
  MPI_Finalize ();
}


void
define_mpi_type (void)
{
  MPI_Type_contiguous ((nghost + 2) * (ny + (nghost + 2) * 2) * (nz + (nghost + 2) * 2), MPI_DOUBLE, &left_right);
  MPI_Type_contiguous (nx * (nghost + 2) * (nz + (nghost + 2) * 2), MPI_DOUBLE, &top_bottom);
  MPI_Type_contiguous (nx * ny * (nghost + 2), MPI_DOUBLE, &front_back);

  MPI_Type_commit (&left_right);
  MPI_Type_commit (&top_bottom);
  MPI_Type_commit (&front_back);
}


void
sw_cart_creat (void)
{
  int n, i;
  
  int reorder, periods[3];
  int color, key, coords[3], rank;


  reorder = 0;
  periods[0] = periodic;
  periods[1] = periodic;
  periods[2] = periodic;

  MPI_Cart_create (MPI_COMM_WORLD, 3, procs, periods, reorder, &XYZ_COMM);
  MPI_Cart_coords (XYZ_COMM, myrank, 3, cart_id);
  MPI_Cart_shift (XYZ_COMM, 0, 1, &left, &right);
  MPI_Cart_shift (XYZ_COMM, 1, 1, &top, &bottom);
  MPI_Cart_shift (XYZ_COMM, 2, 1, &front, &back);

  MPI_Barrier (MPI_COMM_WORLD);

//  printf ("myrank = %d, (%d,%d,%d),%d,%d,%d,%d,%d,%d\n", myrank, cart_id[0], cart_id[1], cart_id[2], left, right, top, bottom, front, back);

  color = cart_id[0];
  key = (cart_id[0] * procs[1] + cart_id[1]) * procs[2] + cart_id[2];
  MPI_Comm_split (XYZ_COMM, color, myrank, &YZ_COMM);
  
  coords[0] = 0;
  coords[1] = 0;
  coords[2] = 0;
  MPI_Cart_rank (XYZ_COMM, coords, &prank);
  color1 = myrank / counts;
  key1 = myrank % counts;
  MPI_Comm_split (MPI_COMM_WORLD, color1, key1, &R_COMM); 

  for (n = 0; n < nac; n++) {
    MPI_Send_init (&ac[n].fieldEs_left[0], 1, left_right, left, 9, MPI_COMM_WORLD, &ac[n].ireq_left_right_fieldE[0]);
    MPI_Recv_init (&ac[n].fieldEr_right[0], 1, left_right, right, 9, MPI_COMM_WORLD, &ac[n].ireq_left_right_fieldE[1]);
    MPI_Send_init (&ac[n].fieldEs_right[0], 1, left_right, right, 9, MPI_COMM_WORLD, &ac[n].ireq_left_right_fieldE[2]);
    MPI_Recv_init (&ac[n].fieldEr_left[0], 1, left_right, left, 9, MPI_COMM_WORLD, &ac[n].ireq_left_right_fieldE[3]);
    MPI_Send_init (&ac[n].fieldEs_top[0], 1, top_bottom, top, 9, MPI_COMM_WORLD, &ac[n].ireq_top_bottom_fieldE[0]);
    MPI_Recv_init (&ac[n].fieldEr_bottom[0], 1, top_bottom, bottom, 9, MPI_COMM_WORLD, &ac[n].ireq_top_bottom_fieldE[1]);
    MPI_Send_init (&ac[n].fieldEs_bottom[0], 1, top_bottom, bottom, 9, MPI_COMM_WORLD, &ac[n].ireq_top_bottom_fieldE[2]);
    MPI_Recv_init (&ac[n].fieldEr_top[0], 1, top_bottom, top, 9, MPI_COMM_WORLD, &ac[n].ireq_top_bottom_fieldE[3]);
    //MPI_Send_init (&ac[n].fieldE[nx * ny * nghost], 1, front_back, front, 9, MPI_COMM_WORLD, &ac[n].ireq_front_back_fieldE[0]);
    MPI_Send_init (&ac[n].fieldEs_front[0], 1, front_back, front, 9, MPI_COMM_WORLD, &ac[n].ireq_front_back_fieldE[0]);
    MPI_Recv_init (&ac[n].fieldEr_back[0], 1, front_back, back, 9, MPI_COMM_WORLD, &ac[n].ireq_front_back_fieldE[1]);
    MPI_Send_init (&ac[n].fieldEs_back[0], 1, front_back, back, 9, MPI_COMM_WORLD, &ac[n].ireq_front_back_fieldE[2]);
    MPI_Recv_init (&ac[n].fieldEr_front[0], 1, front_back, front, 9, MPI_COMM_WORLD, &ac[n].ireq_front_back_fieldE[3]);
  }
  MPI_Send_init(&s_left[0], 1, left_right, left, 9, MPI_COMM_WORLD, &ireq_left_right[0]);
  MPI_Recv_init(&r_right[0], 1, left_right, right, 9, MPI_COMM_WORLD, &ireq_left_right[1]);
  MPI_Send_init(&s_right[0], 1, left_right, right, 9, MPI_COMM_WORLD, &ireq_left_right[2]);
  MPI_Recv_init(&r_left[0], 1, left_right, left, 9, MPI_COMM_WORLD, &ireq_left_right[3]);
  MPI_Send_init(&s_top[0], 1, top_bottom, top, 9, MPI_COMM_WORLD, &ireq_top_bottom[0]);
  MPI_Recv_init(&r_bottom[0], 1, top_bottom, bottom, 9, MPI_COMM_WORLD, &ireq_top_bottom[1]);
  MPI_Send_init(&s_bottom[0], 1, top_bottom, bottom, 9, MPI_COMM_WORLD, &ireq_top_bottom[2]);
  MPI_Recv_init(&r_top[0], 1, top_bottom, top, 9, MPI_COMM_WORLD, &ireq_top_bottom[3]);
  MPI_Send_init(&s_front[0], 1, front_back, front, 9, MPI_COMM_WORLD, &ireq_front_back[0]);
  MPI_Recv_init(&r_back[0], 1, front_back, back, 9, MPI_COMM_WORLD, &ireq_front_back[1]);
  MPI_Send_init(&s_back[0], 1, front_back, back, 9, MPI_COMM_WORLD, &ireq_front_back[2]);
  MPI_Recv_init(&r_front[0], 1, front_back, front, 9, MPI_COMM_WORLD, &ireq_front_back[3]);

/*  for (n = 0; n < nch; n++) {
    MPI_Send_init (&ch[n].fieldCIs_left[0], 1, left_right, left, 9, MPI_COMM_WORLD, &ch[n].ireq_left_right_fieldCI[0]);
    MPI_Recv_init (&ch[n].fieldCIr_right[0], 1, left_right, right, 9, MPI_COMM_WORLD, &ch[n].ireq_left_right_fieldCI[1]);
    MPI_Send_init (&ch[n].fieldCIs_right[0], 1, left_right, right, 9, MPI_COMM_WORLD, &ch[n].ireq_left_right_fieldCI[2]);
    MPI_Recv_init (&ch[n].fieldCIr_left[0], 1, left_right, left, 9, MPI_COMM_WORLD, &ch[n].ireq_left_right_fieldCI[3]);
    MPI_Send_init (&ch[n].fieldCIs_top[0], 1, top_bottom, top, 9, MPI_COMM_WORLD, &ch[n].ireq_top_bottom_fieldCI[0]);
    MPI_Recv_init (&ch[n].fieldCIr_bottom[0], 1, top_bottom, bottom, 9, MPI_COMM_WORLD, &ch[n].ireq_top_bottom_fieldCI[1]);
    MPI_Send_init (&ch[n].fieldCIs_bottom[0], 1, top_bottom, bottom, 9, MPI_COMM_WORLD, &ch[n].ireq_top_bottom_fieldCI[2]);
    MPI_Recv_init (&ch[n].fieldCIr_top[0], 1, top_bottom, top, 9, MPI_COMM_WORLD, &ch[n].ireq_top_bottom_fieldCI[3]);
    MPI_Send_init (&ch[n].fieldCI[nx * ny * nghost], 1, front_back, front, 9, MPI_COMM_WORLD, &ch[n].ireq_front_back_fieldCI[0]);
    MPI_Recv_init (&ch[n].fieldCIr_back[0], 1, front_back, back, 9, MPI_COMM_WORLD, &ch[n].ireq_front_back_fieldCI[1]);
    MPI_Send_init (&ch[n].fieldCI[nx * ny * (nz - nghost - (nghost + 2))], 1, front_back, back, 9, MPI_COMM_WORLD, &ch[n].ireq_front_back_fieldCI[2]);
    MPI_Recv_init (&ch[n].fieldCIr_front[0], 1, front_back, front, 9, MPI_COMM_WORLD, &ch[n].ireq_front_back_fieldCI[3]);
  }*/
}
