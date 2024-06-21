#ifndef  ADD_H
#define  ADD_H
 extern void start_mpi (int argc, char **argv);
// extern double CH_f(int n);
 extern void elastic_input();
 extern void elastic_malloc();
 extern void alloc_vars();
 extern void init_vars();
 extern void init_KL();
 extern void anisotropic_input();
 extern void define_mpi_type();
 extern void sw_cart_creat();
 extern void init_para();
 extern void read_matrices();
// extern void init_field();
 extern void rotation_matrix();
 extern void init_KL(void);
 extern void myball(int iter, double rad, double *field);
 extern void fft_setup();
 extern void elastic_init();
 extern void copy();
 extern void print_info(void);
 extern void check_soln_new (double time);
 extern void transfer(void);
 extern void elastic_transfer(void);
 extern void elastic_calculate();
 //extern void calc_mu();
 extern void ac_calc_FU(int n, double *f);
 extern void ac_calc_F1(void);
 extern void fft_forward(double *in, double *out_re, double *out_im);
 extern void fft_forward_A(double *in, double *out_re, double *out_im);
 extern void fft_forward_B(double *in, double *out_re, double *out_im);
 extern void fft_forward_C(double *in, double *out_re, double *out_im);
 extern void fft_forward_D(double *in, double *out_re, double *out_im);
 extern void fft_backward(double *in_re, double *in_im, double *out);
 //extern void dgemm1(int n);
 //extern void dgemm2(int n);
 //extern void update(int n);
 //extern void ac_calc_FU(int n);
// extern void ch_calc_FU_VariableMobility(int n, double *f);
 //extern void PUX(double *A, double *B, double *C, double *D);
 //extern void PUY(double *A, double *B, double *C, double *D);
 //extern void PUZ(double *A, double *B, double *C, double *D, double *E);
 //extern void ac_updateU_new(int n, double *field, double *field1);
 //extern void zxy_xyz(double *f, double *ft);
 //extern void xyz_yzx(double *f, double *ft);
 //extern void ch_calc_FU(int n, double *f);
 //extern void ac_updateU_new(int n, double *field, double *field1);
// extern void ch_updateU_new(int n, double *field, double *field1);
 //extern void prepare_U1_new(double *field1, double *field2);
 //extern void prepare_U2_new(double *phi, double *field1, double *field2);
 //extern void correct_U_new(double *field, double *field1);
// extern void ch_calc_FU_ConstantMobility(int n, double *fieldci1);
// extern void ch_mu(int n, double LCI, double *fieldmu_left, double *fieldmu_right, double *fieldmu_top, double *fieldmu_bottom, double *fieldmu_front, double *fieldmu_back,\
           double *fielde_left, double *fielde_right, double *fielde_top, double *fielde_bottom, double *fielde_front, double *fielde_back,\
           double *fieldu_left, double *fieldu_right, double *fieldu_top, double *fieldu_bottom, double *fieldu_front, double *fieldu_back);
 extern void fft_finish();
 extern void elastic_init_transfer();
 extern void close_mpi();
 extern void elastic_finish();
 extern void dealloc_vars();


// cblas_dgemm 
// fft3d_create 
// fft3d_set 
// fft3d_setup 
// fft3d_compute 
// fft3d_destroy 



#endif
