DIR_SRC = ./src

DIR_OBJ = ./obj

DIR_BIN = ./bin

SOURCES = $(wildcard ${DIR_SRC}/*.c)  

OBJECTS = $(patsubst %.c,${DIR_OBJ}/%.o,$(notdir ${SOURCES})) 

PROGS   = ${DIR_BIN}/ScLETD

CC      = mpicc

CFLAGS  = -O3 -lm -lmkl_core -lmkl_intel_lp64 -lmkl_sequential  -I/home/scwss/local/fftmpi -DFFT_DOUBLE

LDFLAGS = -O3 -lm -lmkl_core -lmkl_intel_lp64 -lmkl_sequential -L/home/scwss/local/fftmpi -lfft3dmpi

${PROGS} : ${OBJECTS}
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

${DIR_OBJ}/%.o : ${DIR_SRC}/%.c
	$(CC) $(CFLAGS) -c $< -o $@

run:
	mpirun -n 8 ./bin/ScLETD 64

.PHONY:clean
clean:
	rm -f $(OBJECTS) $(PROGS)
