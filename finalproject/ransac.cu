#include <iostream>


float** readPtsFile(std::string filename)
{
/*
        with open('./test.bin', 'rb') as f:
            n_bytes = int.from_bytes(f.read(4), byteorder='little')
            
            src_pts_bytes = f.read(n_bytes)
            dst_pts_bytes = f.read(n_bytes)

            src_pts_test = np.frombuffer(src_pts_bytes, dtype=np.float32).reshape(-1, 2)
            dst_pts_test = np.frombuffer(dst_pts_bytes, dtype=np.float32).reshape(-1, 2)
*/
}

// make 2 modes, 1 for a file and another for stdin
float** read_stdin()
{
}

int main()
{
    std::string lineInput;
    while (std::cin >> lineInput) {
        std::cout << lineInput;
    }
}
