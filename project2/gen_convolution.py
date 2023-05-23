import argparse
import math
import re
import string
import subprocess

OPS_IN_TERA_OPS = 1e-12
KIB_IN_BYTES = 1024
DATATYPE_BYTES = 4
FFMA_CYCLES = 4
BASE_CLOCK_HZ = 1200 * 1e6
L1_CACHE_SIZE_KB = 96
L2_CACHE_SIZE_MB = 4.5
L2_BANDWIDTH_GBPS = 2155
L1_MISS_CYCLES = 193
L2_MISS_CYCLES = 1029
WARP_SIZE = 32
DRAM_BANDWIDTH_GBPS = 651
NUM_STREAMING_MULTIPROCESSORS = 80
SINGLE_PRECISION_THROUGHPUT = 14.03
SINGLE_PRECISION_THROUGHPUT_PER_SM = SINGLE_PRECISION_THROUGHPUT / NUM_STREAMING_MULTIPROCESSORS


def get_execution_time(kx, ky, nx, ny, ni, nn, block_x_dim, block_y_dim, block_z_dim):
    blocks = ((nx - kx + 1) / block_x_dim) * ((ny - ky + 1) / block_y_dim) * (nn / block_z_dim)
    macs_per_sm = kx * ky * block_x_dim * block_y_dim * ni * block_z_dim
    macs_per_thread = macs_per_sm / WARP_SIZE
    cycles_per_thread = macs_per_thread * FFMA_CYCLES
    blocks_per_sm = math.ceil(blocks / NUM_STREAMING_MULTIPROCESSORS)
    execution_time = blocks_per_sm * cycles_per_thread / BASE_CLOCK_HZ

    weights = kx * ky * ni * nn * DATATYPE_BYTES
    inputs = nx * ny * ni * DATATYPE_BYTES
    outputs = (nx - kx + 1) * (ny - ky + 1) * nn * DATATYPE_BYTES
    total_size_bytes = weights + inputs + outputs
    l1_miss_per_sm = (total_size_bytes / NUM_STREAMING_MULTIPROCESSORS) / (L1_CACHE_SIZE_KB * KIB_IN_BYTES)
    l2_misses = total_size_bytes / (L2_CACHE_SIZE_MB * KIB_IN_BYTES**2)
    l1_miss_per_sm_cycles = l1_miss_per_sm * L1_MISS_CYCLES
    l2_miss_cycles = l2_misses * L2_MISS_CYCLES
    l1_miss_time = l1_miss_per_sm_cycles / BASE_CLOCK_HZ
    l2_miss_time = l2_miss_cycles / BASE_CLOCK_HZ
    dram_load_time = total_size_bytes / (DRAM_BANDWIDTH_GBPS * (KIB_IN_BYTES**3))
    memory_load_time = l1_miss_time + l2_miss_time + dram_load_time

    return execution_time + memory_load_time


def get_memory_load_time(kx, ky, nx, ny, ni, nn):
    weights = kx * ky * ni * nn * DATATYPE_BYTES
    inputs = nx * ny * ni * DATATYPE_BYTES
    outputs = (nx - kx + 1) * (ny - ky + 1) * nn * DATATYPE_BYTES
    total_size_bytes = weights + inputs + outputs
    l1_miss_per_sm = (total_size_bytes / NUM_STREAMING_MULTIPROCESSORS) / (L1_CACHE_SIZE_KB * KIB_IN_BYTES)
    l2_misses = total_size_bytes / (4.5 * KIB_IN_BYTES**2)
    l1_miss_per_sm_cycles = l1_miss_per_sm * L1_MISS_CYCLES
    l2_miss_cycles = l2_misses * L2_MISS_CYCLES
    l1_miss_time = l1_miss_per_sm_cycles / BASE_CLOCK_HZ
    l2_miss_time = l2_miss_cycles / BASE_CLOCK_HZ
    dram_load_time = total_size_bytes / (DRAM_BANDWIDTH_GBPS * (KIB_IN_BYTES**3))

    memory_load_time = l1_miss_time + l2_miss_time + dram_load_time
    return memory_load_time


def get_operation_intensity(kx, ky, nx, ny, ni, nn, mem_size_bytes):
    weights = kx * ky * ni * nn * DATATYPE_BYTES
    inputs = nx * ny * ni * DATATYPE_BYTES
    outputs = (nx - kx + 1) * (ny - ky + 1) * nn * DATATYPE_BYTES
    total_size_bytes = weights + inputs + outputs
    if total_size_bytes > mem_size_bytes:
        return 0.0, False
    computations = kx * ky * nx * ny * ni * nn
    operation_intensity = computations / mem_size_bytes
    return operation_intensity, True


def create_and_run_convolution(nx, ny, kx, ky, ni, nn, block_x_dim, block_y_dim, block_z_dim):
    with open('_convolution.cu', 'r') as base_conv_fo:
        base_conv_file_str = base_conv_fo.read()
        interpolated_conv_file_str = string.Template(base_conv_file_str).substitute(
            Nx=nx,
            Ny=ny,
            Kx=kx,
            Ky=ky,
            Ni=ni,
            Nn=nn,
            blockXDim=block_x_dim,
            blockYDim=block_y_dim,
            blockZDim=block_z_dim,
        )
        with open('convolution.cu', 'w') as interpolated_conv_fo:
            interpolated_conv_fo.write(interpolated_conv_file_str)

    subprocess.call(["make"])
    nvprof_out = subprocess.check_output(["nvprof", "./convolution"], stderr=subprocess.STDOUT, universal_newlines=True)
    subprocess.call(["make", "clean"])
    runtime_match = re.findall(
        '\\d+[.]?\\d+%\\s+\\d+[.]?\\d+(?: ns|us|ms)\\s+\\d+\\s+\\d+[.]?\\d+(?:ns|us|ms)\\s+\\d+[.]?\\d+(?:ns|us|ms)\\s+'
        '(\\d+[.]?\\d+)(ns|us|ms)\\s+Conv2dGpu[(]float[*], float[*], float[*][)]', nvprof_out)
    if len(runtime_match) != 1:
        print(nvprof_out)
        raise 'unknown match discovered'
    runtime_decimal, runtime_unit = runtime_match[0]
    runtime = float(runtime_decimal)
    if runtime_unit == 'ns':
        runtime *= 1e-9
    elif runtime_unit == 'us':
        runtime *= 1e-6
    else:
        runtime *= 1e-3
    # print('Runtime in seconds: {}'.format(runtime))
    # print('TeraOPs/s: {}'.format((computations / 1e12) / runtime))


def main():
    arg_parser = argparse.ArgumentParser(description='script that creates a convolution file with the specified '
                                                     'parameters')
    arg_parser.add_argument('-Nx', type=int, required=True)
    arg_parser.add_argument('-Ny', type=int, required=True)
    arg_parser.add_argument('-Kx', type=int, required=True)
    arg_parser.add_argument('-Ky', type=int, required=True)
    arg_parser.add_argument('-Ni', type=int, required=True)
    arg_parser.add_argument('-Nn', type=int, required=True)
    arg_parser.add_argument('-blockXDim', type=int, required=True)
    arg_parser.add_argument('-blockYDim', type=int, required=True)
    arg_parser.add_argument('-blockZDim', type=int, required=True)
    args = arg_parser.parse_args()

    nx = args.Nx
    ny = args.Ny
    kx = args.Kx
    ky = args.Ky
    ni = args.Ni
    nn = args.Nn
    block_x_dim = args.blockXDim
    block_y_dim = args.blockYDim
    block_z_dim = args.blockZDim
    print('{} us'.format(get_execution_time(kx, ky, nx, ny, ni, nn, block_x_dim, block_y_dim, block_z_dim) * 1e6))
    # print('{} us'.format(get_memory_load_time(kx, ky, nx, ny, ni, nn)))

    # weights = args.Kx * args.Ky * args.Ni * args.Nn * DATATYPE_BYTES
    # inputs = args.Nx * args.Ny * args.Ni * DATATYPE_BYTES
    # outputs = (args.Nx - args.Kx + 1) * (args.Ny - args.Ky + 1) * args.Nn * DATATYPE_BYTES
    # total_size_bytes = weights + inputs + outputs
    # decimal_l1 = total_size_bytes / (L1_CACHE_SIZE_KB * KIB_IN_BYTES)
    # computations = args.Kx * args.Ky * args.Nx * args.Ny * args.Ni * args.Nn
    # l1_operation_intensity = computations / total_size_bytes
    # l2_tflops = l1_operation_intensity * OPS_IN_TERA_OPS * L2_BANDWIDTH_GBPS * KIB_IN_BYTES ** 3
    # dram_tflops = l1_operation_intensity * OPS_IN_TERA_OPS * DRAM_BANDWIDTH_GBPS * KIB_IN_BYTES ** 3
    #
    # print('Weight space occupied in bytes: {}'.format(weights))
    # print('Inputs space occupied in bytes: {}'.format(inputs))
    # print('Outputs space occupied in bytes: {}'.format(outputs))
    # print('Total size space occupied in bytes: {}'.format(total_size_bytes))
    # print('Percent L1: {0:.2f}'.format(decimal_l1 * 100))
    # print('Operation intensity: {0:.2f}'.format(l1_operation_intensity))
    # print('L2 TFlops: {0:.2f}'.format(l2_tflops))
    # print('DRAM TFlops: {0:.2f}'.format(dram_tflops))


if __name__ == '__main__':
    main()
