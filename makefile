ROCBLAS_DIR = rocblas path

ROCBLAS_INC = $(ROCBLAS_DIR)/include

ROCBLAS_LIB = $(ROCBLAS_DIR)/lib

HIP_INC     = hip include path

HIP_LIB     = hip lib path

DIR_SRC = ./src

DIR_OBJ = ./obj

DIR_BIN = ./bin

SOURCES = $(wildcard ${DIR_SRC}/*.c)  

OBJECTS = $(patsubst %.c,${DIR_OBJ}/%.o,$(notdir ${SOURCES})) 

PROGS   = ${DIR_BIN}/ScLETD

CC      = hipcc mpic++ 

CFLAGS  = -O3 -lm -lmkl_core -lmkl_intel_lp64 -lmkl_sequential -I$(ROCBLAS_INC) -I$(HIP_INC) -I/mkl path 
LDFLAGS = -O3 -lm -lmkl_core -lmkl_intel_lp64 -lmkl_sequential -L$(ROCBLAS_LIB) -lrocblas -L$(HIP_LIB) -lhip_hcc -L/mkl path

${PROGS} : ${OBJECTS}
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

${DIR_OBJ}/%.o : ${DIR_SRC}/%.c
	$(CC) $(CFLAGS) -c $< -o $@ -w

run:
	mpirun -n 64 ./bin/ScLETD 64

.PHONY:clean
clean:
	rm -f $(OBJECTS) $(PROGS)
