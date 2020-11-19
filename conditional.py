import os

os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"

from IP_sim import *
from argparse import ArgumentParser, RawTextHelpFormatter
import json


def get_option():
    argparser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    argparser.add_argument('-f', '--file', type=str,
                           default=None, help='Configuration file. (default=None)')
    argparser.add_argument('-s', '--save_option', type=str, default="",
                           help='Save Option. \n'
                                '"bo" (blur matrix only), \n'
                                '"oo" (original trans matrix only), \n'
                                '"" (blurred trans matrix) <- default')
    return argparser.parse_args()


def probably_matrix(kw):
    OpticalSystem(**kw)


def get_fb_mat(kw):
    OpticalSystem(**kw).plasma_data.mode_matrix()


if __name__ == '__main__':
    args = get_option()

    # arguments
    # sim_name=None, mode="pinhole", auto=False, tm=False,
    # hole_list=None, f=14.3, screen_size=(17.0, 17.0), hole_size=0.5, aperture_z=58, aperture_phi=21,
    # shape=(10, 10, 10), xyz_range=(100, 100, 100), o_xyz=(0, 0, 300), image_size=(170, 170), n=10

    if args.file:
        with open(args.file) as f:
            config_dic = json.load(f)
    else:
        config_dic = {"sim_name": "Test",
                      "mode": "pinhole",
                      "auto": True,
                      "tm": True,
                      "save_option": "",
                      "hole_list": None,
                      "f": 14.3,
                      "screen_size": (17.0, 17.0),
                      "hole_size": 0.5,
                      "aperture_z": 58,
                      "aperture_phi": 21,
                      "shape": (10, 10, 10),
                      "xyz_range": (100, 100, 100),
                      "start_xyz": (0, 0, 300),
                      "image_size": (170, 170),
                      "n": 2}

    config_dic["save_option"] = args.save_option
    # config_dic["save_option"] = "fb"
    while True:
        print(f"{config_dic}\nIs it OK?", end="")
        ok = input(" y/n: ")
        if ok == "y":
            break
        elif ok == "n":
            print("Quit.")
            quit()

    probably_matrix(config_dic)
