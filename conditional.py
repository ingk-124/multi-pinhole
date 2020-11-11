from IP_sim import *
from argparse import ArgumentParser
import json


def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-f', '--file', type=str,
                           default=None,
                           help='Configure file (type=str) default=None')
    return argparser.parse_args()


def probably_matrix(kw):
    OpticalSystem(**kw)


# def probably_matrix(kw, save=True, parent="./npz/"):
#     W = OpticalSystem(**kw,auto=True)
#     path = dir_rename(parent + W.sim_name)
#     I = W.plasma_data.voxel.shape[1]
#     J = W.return_image_size[0] * W.return_image_size[1]
#     P = sparse.csr_matrix((0, J))
#     for i in tqdm(range(I)):
#         W.plasma_data.voxel[-1, i] = 1
#         H = W.simulate(fast_mode=True)
#         # print(f"P.shape={P.shape},H.shape={H.shape}")
#         P = sparse.vstack([P, sparse.csr_matrix(H)])
#         W.plasma_data.voxel[-1, i] = 0
#
#     if save:
#         sparse.save_npz(path, P)
#
#     return P


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
        config_dic = {"sim_name": None,
                      "mode": "pinhole",
                      "auto": True,
                      "tm": True,
                      "hole_list": None,
                      "f": 14.3,
                      "screen_size": (17.0, 17.0),
                      "hole_size": 0.5,
                      "aperture_z": 58,
                      "aperture_phi": 21,
                      "shape": (10, 10, 10),
                      "xyz_range": (100, 100, 100),
                      "o_xyz": (0, 0, 300),
                      "image_size": (170, 170),
                      "n": 2}

    while True:
        print(f"{config_dic}\nIs it OK?", end="")
        ok = input(" y/n: ")
        if ok == "y":
            break
        elif ok == "n":
            for k, v in config_dic.items():
                print(f"{k}:{v}", end="")
                v_ = input("->")
                config_dic[k] = v_ if v_ else v

    probably_matrix(config_dic)
