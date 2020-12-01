# import os

# os.environ["OPENBLAS_NUM_THREADS"] = "64"
# os.environ["MKL_NUM_THREADS"] = "64"

from IP_sim import *
from argparse import ArgumentParser, RawTextHelpFormatter
import json
from pprint import pprint


def get_option():
    argparser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    argparser.add_argument('-f', '--file', type=str,
                           default=None, help='Configuration file. (default=None)')
    argparser.add_argument('-s', '--save_option', type=str, default="",
                           help='Save Option. \n'
                                '"bo" (blur matrix only), \n'
                                '"oo" (original trans matrix only), \n'
                                '"fb" (Fourier Bessel image), \n'
                                '"" (blurred trans matrix) <- default')
    return argparser.parse_args()


def probably_matrix(kw):
    optsys = OpticalSystem(**kw)
    # optsys.save_transmission_matrix()


if __name__ == '__main__':
    args = get_option()

    # arguments
    # sim_name=None, mode="pinhole", auto=False, tm=False, save_option="",
    # hole_list=None, hole_z=948, f=14.3, aperture_z=58, aperture_phi=21,
    # screen_size=(17.0, 17.0), hole_size=0.5, image_size=(170, 170), n=10,
    # shape=None, xyz_range=None, start_xyz=None, parameter_max=None

    arguments = {"sim_name": "Test_test",
                 "mode": "pinhole",
                 "auto": False,
                 "tm": True,
                 "save_option": "",
                 # "hole_list": [[0.0,5.0],[2.0,-2.0]],
                 "hole_z": 948,
                 "f": 14.3,
                 "aperture_z": 58,
                 "aperture_phi": 21,
                 "screen_size": (17.0, 17.0),
                 "hole_size": 0.5,
                 "image_size": (170, 170),
                 "n": 2,
                 "shape": (70, 70, 70),
                 "xyz_range": None,
                 "start_xyz": None,
                 "parameter_max": [2, 2, 3, 3],
                 }

    if args.file:
        with open(args.file) as f:
            config_dic = json.load(f)
        for k, v in config_dic.items():
            arguments[k] = v

    arguments["save_option"] = args.save_option if args.save_option else "oo"

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
