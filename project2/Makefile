CFLAGS = -std=c++11 # -g -O0
CC = nvcc
LDLIBS = -lcublas -lcudnn
INCLUDE_FOLDER  = ../../../../../../usr/local/cuda/include/
HEADERS = -I $(INCLUDE_FOLDER) $(INCLUDE_FOLDER)/cudnn.h # -I /usr/local/cuda/samples/common/inc -I /usr/local/cuda/include/cudnn.h

convolution: convolution.o
	$(CC) $(CFLAGS) -o convolution convolution.o $(LDLIBS)

convolution.o: convolution.cu ../cuda_common.h
	$(CC) $(CFLAGS) -c convolution.cu

conv224:
	python gen_convolution.py -Nx=224 -Ny=224 -Kx=3 -Ky=3 -Ni=64 -Nn=64 -blockXDim=222 -blockYDim=1 -blockZDim=4

conv14:
	python gen_convolution.py -Nx=14 -Ny=14 -Kx=3 -Ky=3 -Ni=512 -Nn=512 -blockXDim=12 -blockYDim=12 -blockZDim=4

clean:
	rm -f *.o convolution convolution.o convolution.cu