#ifndef ADD_H
#define ADD_H
extern void start_mpi(int argc, char **argv);
extern void alloc_vars();
extern void init_vars();
extern void define_mpi_type();
extern void sw_cart_creat();
extern void init_para();
extern void read_matrices();
extern void init_field_origin();
extern void write_small(double *f);
extern void read_ori(void);
extern void check_max(double *v);
extern void write_n(double *n);
extern void write_Bn_re(double *f, double *d, int p, int q);
extern void write_Bn_im(double *f, int p);
extern void cal_occupy_im_re(double *im, double *re);
extern void initial_ac(double *fieldep, double *fieldeq);
extern void initial_ch(double *fielde1, double *fielde2, double *fielde3, double *fielde4, double *fielde5, double *fielde6);
extern void init_field();
extern void print_info(void);
extern void ac_calc_F1(int iter, int detect_iter);
extern void compress_save_output(char *output_path, float *input, unsigned char *output, int size);
extern void check_soln_new(double time);
extern void outputfield(void);
extern void transfer(void);
extern void close_mpi();
extern void dealloc_vars();
extern void ch_mu(int n, double L, double *Umu_left, double *Umu_right, double *Umu_top, double *Umu_bottom, double *Umu_front, double *Umu_back,
                  double *Ue_left, double *Ue_right, double *Ue_top, double *Ue_bottom, double *Ue_front, double *Ue_back);
extern void elastic_input();
extern void anisotropic_input();
extern void elastic_init();
extern void elastic_transfer();
extern void elastic_calculate_forward(int n);
extern void elastic_calculate_backward(int n);
extern void elastic_init_transfer();
extern void elastic_finish();
extern void conv_init_transfer();
extern void write_section (char *fname, double *field);
#endif
