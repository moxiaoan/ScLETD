#include <stdio.h>
#include "mpi.h"
#include "ScLETD.h"
#include <string.h>
#include <errno.h>
#include <stdlib.h>
void
init_KL () {
  // grid size
  hx = 1e-9;
  hy = 1e-9;
  hz = 1e-9;
  // 6 * lambda * sigma / alpha = 6 * 3 * 0.28 / 2.2 = 2.29
  // or : 6 * 3 * 0.065 / 2.2 = 0.532
  ac[0].epn2 = 2.29e-9; 
  ch[0].epn2 = 1.0;
  // ac and ch [0,160]
  ac[0].LE = 5e-4;
  ac[0].KE = 0; //27.5;
  ch[0].LCI = 1.2302e-10;
  ch[0].KCI = 0; //1e-19; //31.5;
}

double
EF(){
//  return ;
}
// Allen-Cahn function nonlinear items
double AC_f(int n){
  double f;
  switch(n) {
    case 0 : {
      break;
    }
    default: {
      printf ("error:please choose correct AC_FIELD function.\n");
      exit(1);
    }
  }
  return f;
}

// Cahn-Hilliard function nonlinear items
// ch_field1
double CH_f(int n){
  double f;
  switch(n) {
    case 0 : {
      break;
    }
    default: {
      printf ("error:please choose correct CH_FIELD function.\n");
      exit(1);
    }
  }
  return f;
}


