#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "rocblas.h"

hipEvent_t st, ed;
hipEvent_t st2, ed2;
hipEvent_t st3, ed3;
hipEvent_t st4, ed4;
hipDeviceProp_t props;

rocblas_handle handle;
rocblas_int x_m, x_n, x_k;
rocblas_int y_m, y_n, y_k;
rocblas_int z_m, z_n, z_k;
rocblas_int Gx_m, Gx_n, Gx_k;
rocblas_int Gy_m, Gy_n, Gy_k;
rocblas_int Gz_m, Gz_n, Gz_k;

rocblas_int conv_x_m, conv_x_n, conv_x_k, conv_y_m, conv_y_n, conv_y_k, conv_z_m, conv_z_n, conv_z_k;
rocblas_int conv_big_x_m, conv_big_x_n, conv_big_x_k, conv_big_y_m, conv_big_y_n, conv_big_y_k, conv_big_z_m, conv_big_z_n, conv_big_z_k;
 
