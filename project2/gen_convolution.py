import argparse
import string


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

    weights = args.Kx * args.Ky * args.Ni * args.Nn * 4
    inputs = args.Nx * args.Ny * args.Ni * 4
    outputs = (args.Nx - args.Kx + 1) * (args.Ny - args.Ky + 1) * args.Nn * 4
    total_size = weights + inputs + outputs
    decimal_l1 = total_size / (65536/1.5)
    computations = args.Kx * args.Ky * args.Nx * args.Ny * args.Ni * args.Nn
    operation_intensity = computations / total_size

    print('Weight space occupied in bytes: {}'.format(weights))
    print('Inputs space occupied in bytes: {}'.format(inputs))
    print('Outputs space occupied in bytes: {}'.format(outputs))
    print('Total size space occupied in bytes: {}'.format(total_size))
    print('Percent L1: {0:.2f}'.format(decimal_l1 * 100))
    print('Operation intensity: {0:.2f}'.format(operation_intensity))
    print('Cycle latency (assuming FFMA per Operation): {}'.format(computations * 6))

    with open('_convolution.cu', 'r') as base_conv_fo:
        base_conv_file_str = base_conv_fo.read()
        interpolated_conv_file_str = string.Template(base_conv_file_str).substitute(
            Nx=args.Nx,
            Ny=args.Ny,
            Kx=args.Kx,
            Ky=args.Ky,
            Ni=args.Ni,
            Nn=args.Nn,
            blockXDim=args.blockXDim,
            blockYDim=args.blockYDim,
            blockZDim=args.blockZDim,
        )
        with open('convolution.cu', 'w') as interpolated_conv_fo:
            interpolated_conv_fo.write(interpolated_conv_file_str)


if __name__ == '__main__':
    main()
