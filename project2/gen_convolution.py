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
