import os

# os.environ["OPENBLAS_NUM_THREADS"] = "64"
# os.environ["MKL_NUM_THREADS"] = "64"

from IP_sim import *
from argparse import ArgumentParser, RawTextHelpFormatter
import json
from pprint import pprint
import functools
import logging
import struct
import sys

logger = logging.getLogger()


def patch_mp_connection_bpo_17560():
    """Apply PR-10305 / bpo-17560 connection send/receive max size update

    See the original issue at https://bugs.python.org/issue17560 and
    https://github.com/python/cpython/pull/10305 for the pull request.

    This only supports Python versions 3.3 - 3.7, this function
    does nothing for Python versions outside of that range.

    """
    patchname = "Multiprocessing connection patch for bpo-17560"
    if not (3, 3) < sys.version_info < (3, 8):
        logger.info(
            patchname + " not applied, not an applicable Python version: %s",
            sys.version
        )
        return

    from multiprocessing.connection import Connection

    orig_send_bytes = Connection._send_bytes
    orig_recv_bytes = Connection._recv_bytes
    if (
            orig_send_bytes.__code__.co_filename == __file__
            and orig_recv_bytes.__code__.co_filename == __file__
    ):
        logger.info(patchname + " already applied, skipping")
        return

    @functools.wraps(orig_send_bytes)
    def send_bytes(self, buf):
        n = len(buf)
        if n > 0x7fffffff:
            pre_header = struct.pack("!i", -1)
            header = struct.pack("!Q", n)
            self._send(pre_header)
            self._send(header)
            self._send(buf)
        else:
            orig_send_bytes(self, buf)

    @functools.wraps(orig_recv_bytes)
    def recv_bytes(self, maxsize=None):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        if size == -1:
            buf = self._recv(8)
            size, = struct.unpack("!Q", buf.getvalue())
        if maxsize is not None and size > maxsize:
            return None
        return self._recv(size)

    Connection._send_bytes = send_bytes
    Connection._recv_bytes = recv_bytes

    logger.info(patchname + " applied")


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
    argparser.add_argument('-t', '--trans_mat', type=str,
                           default="", help='Transmission matrix file. (default="")')
    return argparser.parse_args()


def probably_matrix(kw):
    os = OpticalSystem(**kw)
    os.print_member()
    os.save_transmission_matrix()


if __name__ == '__main__':
    args = get_option()

    # arguments
    # sim_name=None, mode="pinhole", auto=False, tm=False, save_option="",
    # hole_list=None, hole_z=948, f=14.3, aperture_z=58, aperture_phi=21,
    # screen_size=(17.0, 17.0), hole_size=0.5, image_size=(170, 170), n=10,
    # shape=None, xyz_range=None, start_xyz=None, parameter_max=None

    arguments = {"sim_name": "Test",
                 "mode": "pinhole",
                 "auto": False,
                 "tm": False,
                 "save_option": "",
                 "hole_list": None,
                 "hole_z": 948,
                 "f": 14.3,
                 "aperture_z": 58,
                 "aperture_phi": 21,
                 "screen_size": (17.0, 17.0),
                 "hole_size": 0.5,
                 "image_size": (170, 170),
                 "n": 2,
                 "shape": (100, 100, 100),
                 "xyz_range": None,
                 "start_xyz": None,
                 "parameter_max": None,
                 "tm_file": ""}

    if args.file:
        with open(args.file) as f:
            config_dic = json.load(f)
        for k, v in config_dic.items():
            arguments[k] = v

    arguments["save_option"] = args.save_option if args.save_option else ""
    arguments["tm_file"] = args.trans_mat if args.trans_mat else ""

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
