ROCBLAS_DIR = rocblas path

ROCBLAS_INC = $(ROCBLAS_DIR)/include

ROCBLAS_LIB = $(ROCBLAS_DIR)/lib

HIP_INC     = hip include path

HIP_LIB     = hip lib path

DIR_SRC     = ./src

DIR_OBJ     = ./obj

DIR_BIN     = ./bin

SOURCES_H   = $(wildcard ${DIR_SRC}/*.c)

SOURCES     = $(wildcard ${DIR_SRC}/*.c)

OBJECTS     = $(patsubst %.c,${DIR_OBJ}/%.o,$(notdir ${SOURCES}))

PROGS       = ${DIR_BIN}/ScLETD

CC_H        = hipcc 

LD          = mpicxx

CFLAGS_H    = -O3 -I$(ROCBLAS_INC) -I$(HIP_INC)

CFLAGS_D    = -O3 -I$(ROCBLAS_INC) -I$(HIP_INC) 

LDFLAGS     = -lm -L$(ROCBLAS_LIB) -lrocblas  -L$(HIP_LIB) -lhip_hcc 

${PROGS} : ${OBJECTS} ./obj/ScLETD.o
	$(CC_H) $(OBJECTS) ./obj/ScLETD.o -o $@ $(LDFLAGS) -w	

${DIR_OBJ}/%.o : ${DIR_SRC}/%.c
	$(CC_H) $(CFLAGS_H) -c  $< -o $@ -w

./obj/ScLETD.o : ./src/ScLETD.cpp ./src/ScLETD.h
	$(LD) -c ./src/ScLETD.cpp -o ./obj/ScLETD.o

run:
	mpirun -n 1 ./bin/ScLETD 64


.PHONY:clean
clean:
	rm -f $(OBJECTS) $(PROGS)
