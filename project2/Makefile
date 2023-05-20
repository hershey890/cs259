CFLAGS = -std=c++11 # -g -O0
CC = nvcc
LDLIBS = -lcublas -lcudnn
INCLUDE_FOLDER  = ../../../../../../usr/local/cuda/include/
HEADERS = -I $(INCLUDE_FOLDER) $(INCLUDE_FOLDER)/cudnn.h # -I /usr/local/cuda/samples/common/inc -I /usr/local/cuda/include/cudnn.h

convolution: convolution.o
	$(CC) $(CFLAGS) -o convolution convolution.o $(LDLIBS)

convolution.o: convolution.cu ../cuda_common.h
	$(CC) $(CFLAGS) -c convolution.cu

clean:
	rm -f *.o classifier convolution