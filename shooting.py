# import os

# os.environ["OPENBLAS_NUM_THREADS"] = "64"
# os.environ["MKL_NUM_THREADS"] = "64"

from pinhole import *
from argparse import ArgumentParser, RawTextHelpFormatter
import json
from pprint import pprint


def get_option():
    argparser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    argparser.add_argument('-f', '--file', type=str,
                           default=None, help='Configuration file. (default=None)')
    argparser.add_argument('-s', '--save_option', type=str, default="",
                           help='Save Option. \n'
                                '"t" (original trans matrix only), \n'
                                '"b" (blur matrix only), \n'
                                '"f" (Fourier Bessel image), \n'
                                '"" (Do nothing) <- default, \n')
    return argparser.parse_args()


def probably_matrix(kw):
    optsys = OpticalSystem(**kw)
    # optsys.save_transmission_matrix()


if __name__ == '__main__':
    args = get_option()

    # arguments
    # sim_name=None, mode="pinhole", auto=False, tm=False, save_option="", thread_num=-1,
    # h_xy=None, hole_z=948, f=14.3, aperture_depth=58, aperture_phi=20,
    # screen_size=(17.0, 17.0), hole_size=0.5, image_size=(170, 170), n=10,
    # shape=None, x_range=None, y_range=None, z_range=None, parameters_range=None

    arguments = {"sim_name": "Test_test", "auto": False, "tm": True, "save_option": "",
                 "h_xy": None, "hole_z": 948, "f": 14.3, "aperture_depth": 58, "aperture_phi": 21,
                 "screen_size": (17.0, 17.0), "hole_size": 0.5, "image_size": (128, 128), "n": 2, "shape": (50, 50, 50),
                 "x_range": None, "y_range": None, "z_range": None, "parameters_range": [1, 2, 2, 1]}

    if args.file:
        with open(args.file) as f:
            config_dic = json.load(f)
        for k, v in config_dic.items():
            arguments[k] = v

    if not arguments["save_option"]:
        arguments["save_option"] = args.save_option if args.save_option else input("save_option:")
    breakpoint()
    while True:
        pprint(arguments)
        print("Is it OK?", end="")
        ok = input(" y/n: ")
        if ok == "y":
            break
        elif ok == "n":
            print("Quit.")
            quit()

    probably_matrix(arguments)
