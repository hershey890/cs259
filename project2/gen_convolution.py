import argparse
import math
import re
import string
import subprocess
import csv

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
SM_PER_FP_32_CORE = 64
DRAM_BANDWIDTH_GBPS = 651
NUM_STREAMING_MULTIPROCESSORS = 80
SINGLE_PRECISION_THROUGHPUT = 14.03
SINGLE_PRECISION_THROUGHPUT_PER_SM = SINGLE_PRECISION_THROUGHPUT / NUM_STREAMING_MULTIPROCESSORS
THREADS_PER_BLOCK = 1024
OPS_PER_MAC = 2
GPU_MEMORY_BYTES = 12884901888


def get_operation_intensity(kx, ky, nx, ny, ni, nn, mem_size_bytes):
    weights = kx * ky * ni * nn * DATATYPE_BYTES
    inputs = nx * ny * ni * DATATYPE_BYTES
    outputs = (nx - kx + 1) * (ny - ky + 1) * nn * DATATYPE_BYTES
    total_size_bytes = weights + inputs + outputs
    computations = kx * ky * nx * ny * ni * nn * OPS_PER_MAC
    operation_intensity = computations / total_size_bytes
    return operation_intensity


def get_execution_time(kx, ky, nx, ny, ni, nn, block_x_dim, block_y_dim, block_z_dim):
    blocks = ((nx - kx + 1) / block_x_dim) * ((ny - ky + 1) / block_y_dim) * (nn / block_z_dim)
    macs_for_block = kx * ky * ni * block_x_dim * block_y_dim * block_z_dim
    macs_per_core = math.ceil(macs_for_block / SM_PER_FP_32_CORE)
    cycles_per_core = macs_per_core * FFMA_CYCLES
    blocks_per_sm = math.ceil(blocks / NUM_STREAMING_MULTIPROCESSORS)
    execution_time = blocks_per_sm * cycles_per_core / BASE_CLOCK_HZ

    return execution_time


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

    subprocess.call(["make"], stdout=subprocess.DEVNULL)
    nvprof_runtime_out = subprocess.check_output(["nvprof", "./convolution"], stderr=subprocess.STDOUT, universal_newlines=True)
    nvprof_operations_out = subprocess.check_output(["nvprof", "--kernels", "Conv2dGpu", "--metrics", "flop_count_sp", "./convolution"], stderr=subprocess.STDOUT, universal_newlines=True)
    subprocess.call(["make", "clean"], stdout=subprocess.DEVNULL)
    runtime_match = re.findall(
        '\\d+[.]?\\d+%\\s+\\d+[.]?\\d+(?: ns|us|ms)\\s+\\d+\\s+\\d+[.]?\\d+(?:ns|us|ms)\\s+\\d+[.]?\\d+(?:ns|us|ms)\\s+'
        '(\\d+[.]?\\d+)(ns|us|ms)\\s+Conv2dGpu[(]float[*], float[*], float[*][)]', nvprof_runtime_out)
    operations_match = re.findall(
        '\\d+\\s+flop_count_sp\\s+Floating Point Operations[(]Single Precision[)]\\s+(\\d+)', nvprof_operations_out
    )
    if 'CUDA Runtime Error' in nvprof_runtime_out or 'CUDA Runtime Error' in nvprof_operations_out:
        raise ValueError("CUDA runtime error")
    if len(runtime_match) != 1:
        # print(nvprof_runtime_out)
        raise ValueError(nvprof_operations_out)
    if len(operations_match) != 1:
        raise ValueError(nvprof_operations_out)
    runtime_decimal, runtime_unit = runtime_match[0]
    operations = operations_match[0]
    runtime = float(runtime_decimal)
    print(nvprof_operations_out, operations)
    if runtime_unit == 'ns':
        runtime *= 1e-9
    elif runtime_unit == 'us':
        runtime *= 1e-6
    else:
        runtime *= 1e-3
    return runtime, int(operations)


def compare_model(nx, ny, kx, ky, ni, nn, block_x_dim, block_y_dim, block_z_dim):
    weights = kx * ky * ni * nn * DATATYPE_BYTES
    inputs = kx * ny * ni * DATATYPE_BYTES
    outputs = (nx - kx + 1) * (ny - ky + 1) * nn * DATATYPE_BYTES
    total_size_bytes = weights + inputs + outputs

    # nvprof results
    runtime_seconds_nvprof, operations = create_and_run_convolution(
        nx, ny, kx, ky, ni, nn, block_x_dim, block_y_dim, block_z_dim)
    operational_intensity_nvprof = operations / total_size_bytes / OPS_PER_MAC
    teraops_nvprof = (operations * OPS_IN_TERA_OPS) / runtime_seconds_nvprof

    # model results
    operation_intensity_model_l1 = get_operation_intensity(
        kx, ky, block_x_dim, block_y_dim, ni, block_z_dim, L1_CACHE_SIZE_KB * KIB_IN_BYTES)
    operation_intensity_model_vram = get_operation_intensity(kx, ky, nx, ny, ni, nn, GPU_MEMORY_BYTES)
    execution_time = get_execution_time(kx, ky, nx, ny, ni, nn, block_x_dim, block_y_dim, block_z_dim)
    memory_load_time = get_memory_load_time(kx, ky, nx, ny, ni, nn)
    runtime_seconds_model = max(execution_time, memory_load_time)
    computations = kx * ky * nx * ny * ni * nn
    teraops_model = computations * OPS_IN_TERA_OPS / runtime_seconds_model

    print("L1 Operational intensity nv, model:   ", operational_intensity_nvprof, operation_intensity_model_l1)
    print("VRAM Operational intensity nv, model: ", operational_intensity_nvprof, operation_intensity_model_vram)
    print("TOP/s nv, model:                      ", teraops_nvprof, teraops_model)
    print("Runtime (us) nv, model:               ", runtime_seconds_nvprof*1e6, runtime_seconds_model*1e6)

    operational_intensity_accuracy = abs(operation_intensity_model_vram - operational_intensity_nvprof) / operational_intensity_nvprof
    teraops_accuracy = abs(teraops_model - teraops_nvprof) / teraops_nvprof

    return [
        operational_intensity_nvprof,
        operation_intensity_model_l1,
        operation_intensity_model_vram,
        teraops_nvprof,
        teraops_model,
        runtime_seconds_nvprof,
        runtime_seconds_model,
        block_x_dim * block_y_dim * block_z_dim <= 1024,
        operational_intensity_accuracy,
        teraops_accuracy
    ]


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

    compare_model(nx, ny, kx, ky, ni, nn, block_x_dim, block_y_dim, block_z_dim)


if __name__ == '__main__':
    # main()

    test_values = [
        [224, 224, 3, 3, 64, 64, 222, 1, 4],
        [224, 224, 3, 3, 64, 64, 111, 1, 8],
        [224, 224, 3, 3, 64, 64, 222, 2, 2],
        [224, 224, 3, 3, 64, 64, 111, 1, 4],
        [224, 224, 3, 3, 64, 64, 111, 1, 2],
        [224, 224, 3, 3, 64, 64, 111, 2, 1],
        [224, 224, 3, 3, 64, 64, 111, 1, 1],
        [14, 14, 3, 3, 512, 512, 12, 12, 4], # Nx - Kx + 1
        [14, 14, 3, 3, 512, 512, 3, 3, 32],
        [14, 14, 3, 3, 512, 512, 6, 6, 16],
        [14, 14, 3, 3, 512, 512, 12, 6, 8],
        [14, 14, 3, 3, 512, 512, 6, 12, 8],
        [14, 14, 3, 3, 512, 512, 12, 3, 16],
        [14, 14, 3, 3, 512, 512, 3, 12, 16],
    ]
    results = []
    for val in test_values:
        try:
            ret = compare_model(*val)
        except ValueError:
            print("CUDA runtime error for: ", val)
        else:
            block_dims = str(val[-3]) + " " + str(val[-2]) + " " + str(val[-1])
            ret = [block_dims] + ret
            results.append(ret)
    
    with open('results.csv', 'w') as results_fo:
        csv_writer = csv.writer(results_fo)
        csv_writer.writerow([
            'Tile Size',
            'Operational intensity (nvprof)',
            'L1 Operational intensity (model)',
            'VRAM Operational intensity (model)',
            'TOP/s (nvprof)', 'TOP/s (model)',
            'Runtime (us) (nvprof)',
            'Runtime (us) (model)',
            'Can fit in L1',
            'Operational Intensity Absolute Difference Error',
            'Teraops Absolute Difference Error'
        ])
        for result in results:
            csv_writer.writerow(result)
        