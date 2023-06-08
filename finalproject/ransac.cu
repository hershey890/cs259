#include <iostream>
#include <fstream>

struct Points {
public:
    float* source; // n/2 rows x 2 cols
    float* destination; // n/2 rows x 2 cols
};

// Read src_dst_pts.bin -
Points* readPtsFile(std::string filename) {
    std::ifstream stream(filename, std::ios::in | std::ios::binary);

    uint32_t n_bytes;
    stream.read(reinterpret_cast<char *>(&n_bytes), sizeof(uint32_t));

    Points *points = new(std::nothrow) Points;
    points->source = new(std::nothrow) float[n_bytes / sizeof(float)];
    points->destination = new(std::nothrow) float[n_bytes / sizeof(float)];
    stream.read(reinterpret_cast<char *>(points->source), n_bytes);
    stream.read(reinterpret_cast<char *>(points->destination), n_bytes);

    stream.close();

    return points;
}



// make 2 modes, 1 for a file and another for stdin
float** read_stdin()
{
    return nullptr;
}

int main()
{
    readPtsFile("./data/src_dst_pts.bin");
}
