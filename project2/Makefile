CFLAGS = -std=c++11 # -g -O0
CC = nvcc
LDLIBS = -lcublas -lcudnn
INCLUDE_FOLDER  = ../../../../../../usr/local/cuda/include/
HEADERS = -I $(INCLUDE_FOLDER) $(INCLUDE_FOLDER)/cudnn.h # -I /usr/local/cuda/samples/common/inc -I /usr/local/cuda/include/cudnn.h

classifier: classifier.o
	$(CC) $(CFLAGS) -o classifier classifier.o $(LDLIBS)

classifier.o: classifier.cu ../cuda_common.h
	$(CC) $(CFLAGS) -c classifier.cu

convolution: convolution.o
	$(CC) $(CFLAGS) -o convolution convolution.o $(LDLIBS)

convolution.o: convolution.cu ../cuda_common.h
	$(CC) $(CFLAGS) -c convolution.cu

conv: conv1.o
	$(CC) $(CFLAGS) -o conv conv1.o $(LDLIBS)

conv1.o: conv1.cu ../cuda_common.h
	$(CC) $(CFLAGS) -c conv1.cu

clean:
	rm -f *.o classifier