#include <iostream>
#include <fstream>

struct SourceDestinationPointsBytes {
public:
    float* src_pts_bytes; // n/2 rows x 2 cols
    float* dst_pts_bytes; // n/2 rows x 2 cols
};

// Read src_dst_pts.bin -
SourceDestinationPointsBytes* readPtsFile(std::string filename)
{
    std::ifstream stream (filename, std::ios::in | std::ios::binary);

    uint32_t n_bytes;
    stream.read(reinterpret_cast<char*>(&n_bytes), sizeof(uint32_t));

    SourceDestinationPointsBytes* bytes = new (std::nothrow) SourceDestinationPointsBytes;
    bytes->src_pts_bytes = new (std::nothrow) float[n_bytes / sizeof(float)];
    bytes->dst_pts_bytes = new (std::nothrow) float[n_bytes / sizeof(float)];
    stream.read(reinterpret_cast<char*>(bytes->src_pts_bytes), n_bytes);
    stream.read(reinterpret_cast<char*>(bytes->dst_pts_bytes), n_bytes);

    stream.close();

    return bytes;
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
