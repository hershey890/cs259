CFLAGS = -std=c++11 # -g -O0
CC = nvcc
LDLIBS = -lcublas -lcusolver -lcudart
# INCLUDE_FOLDER  = ../../../../../../usr/local/cuda/include/ # 259 server folder
INCLUDE_FOLDER = ~/../../usr/local/cuda/include/ # hersh folder
HEADERS = -I $(INCLUDE_FOLDER) 
# -I ~/projects/ucla/cs259/
# -I /usr/local/cuda/samples/common/inc -I /usr/local/cuda/include/cudnn.h

ransac_img: ransac_img.o
	$(CC) $(CFLAGS) -o ransac_img ransac_img.o $(LDLIBS)

ransac_img.o: ransac_img.cu
	$(CC) $(CFLAGS) -c ransac_img.cu $(HEADERS)

test: test.o
	$(CC) $(CFLAGS) -o test test.o $(LDLIBS)

test.o: test.cu cusolver_utils.h
	$(CC) $(CFLAGS) -c test.cu  $(HEADERS)
#$(HEADERS)

clean:
	rm -f *.o test ransac_img