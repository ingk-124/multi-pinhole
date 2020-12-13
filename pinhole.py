import numpy as np
from PIL import Image
import datetime
from pathlib import Path
import re
from scipy import ndimage
from scipy import sparse
from scipy.special import jv
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import itertools
import multiprocessing as multi
from multiprocessing import Pool
from joblib import Parallel, delayed


def rename(name, over_write=False):
    path = Path(name)
    parent = path.parent
    if path.exists():
        if over_write:
            return path
        else:
            n = len([p for p in parent.glob("*") if re.search(path.stem + r"(\Z|\(\d*\))", p.stem)])
            new_path = parent / f"{path.stem}({n}){path.suffix}"
            return new_path
    else:
        parent.mkdir(parents=True, exist_ok=True)
        return path


def trans_mat2d(a, b, n):
    ab = a * b
    abnn = ab * (n ** 2)
    k_ = range(ab)
    z = np.arange(abnn).reshape((a * n, b * n))
    idx = np.array([z[k // b * n:(k // b + 1) * n, (k % b) * n:(k % b + 1) * n].ravel() for k in k_])
    k_axis = np.tile(k_, (n ** 2, 1)).T
    M = sparse.coo_matrix((np.ones(abnn), (k_axis.ravel(), idx.ravel())), shape=(ab, abnn))
    return M


def trans_mat3d(a, b, c, n):
    abc = a * b * c
    bc = b * c
    abcnnn = abc * (n ** 3)
    k_ = range(abc)
    z = np.arange(abcnnn).reshape((a * n, b * n, c * n))
    idx = np.r_[[z[(k // bc) * n:(k // bc + 1) * n, ((k // c) % b) * n:((k // c) % b + 1) * n,
                 (k % c) * n:(k % c + 1) * n].ravel() for k in k_]]
    k_axis = np.tile(k_, (n ** 3, 1)).T
    M = sparse.coo_matrix((np.ones(abcnnn) / (n ** 3), (k_axis.ravel(), idx.ravel())), shape=(abc, abcnnn))
    return M


def j_mk(m_max, k_max):
    _ = 4 * m_max + 2 * k_max + 5
    x = np.linspace(0, _, _ * 128 + 1)
    j_mat = []
    for m in range(m_max):
        y = jv(m, x)
        dy = y[:-1] * y[1:]
        j_mat.append(x[1:][dy < 0][:k_max])
    return np.vstack(j_mat)


class MyError(Exception):
    pass


class Plasma:  # 仮想プラズマ
    """
    Voxel data of plasma.
    """

    def __init__(self, x_range: (float, float) = None, y_range: (float, float) = None, z_range: (float, float) = None,
                 shape: (int, int, int) = None, R=508, a=250):
        """
        Parameters
        ----------
        shape : list of int
        Shape of voxel
        x_range : list of int
        Range of x-axis
        y_range : list of int
        Range of y-axis
        z_range : list of int
        Range of z-axis
        R: int = 508
        a: int = 250
        """
        super().__init__()
        if x_range is None:
            x_range = [-250.0, 250.0]
        if y_range is None:
            y_range = [0.0, 758.0]
        if z_range is None:
            z_range = [-250.0, 250.0]
        if shape is None:
            shape = [10, 10, 10]

        self.shape = np.array(shape)
        self.R = R
        self.a = a
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        xyz = []
        steps = []

        for [start, end], num in zip([x_range, y_range, z_range], shape):
            arr, step = np.linspace(start, end, num, retstep=True)
            xyz.append(arr)
            steps.append(step)

        self.d_xyz = np.array(steps)  # size of voxel

        self.J = int(np.prod(shape))
        self.j_mat = None
        self.parameters = None

        xxx, yyy, zzz = np.meshgrid(*xyz, indexing="ij")

        self.voxel = np.r_[
            "1,2,0", xxx.ravel(), yyy.ravel(), zzz.ravel(), np.ones(self.J), np.zeros(self.J)].T

        self.r, self.theta, self.phi = self.rtp()

    def rtp(self):
        """
        Calculate r, theta and phi for each point
        """
        r_xy = np.linalg.norm(self.voxel[:2], axis=0) - self.R
        r = np.linalg.norm([r_xy, self.voxel[2]], axis=0) / self.a
        theta = np.arctan2(self.voxel[2], r_xy)
        phi = np.arctan2(*self.voxel[:2])
        return r, theta, phi

    def fb(self, para):
        m, k, (tor, pol), cs = para
        iota = pol / tor
        psy = 2 * np.pi * iota

        if cs == "cos":
            luminosity = sum(
                [jv(m, self.j_mat[m, k - 1] * self.r) * np.cos(m * (self.theta + iota * self.phi + psy * _)) for _ in
                 range(tor)])
            return sparse.csr_matrix(np.where(self.r > 1, 0, luminosity))
        elif cs == "sin":
            luminosity = sum(
                [jv(m, self.j_mat[m, k - 1] * self.r) * np.sin(m * (self.theta + iota * self.phi + psy * _)) for _ in
                 range(tor)])
            return sparse.csr_matrix(np.where(self.r > 1, 0, luminosity))
        else:
            raise MyError("argument cs is must be 'cos' or 'sin'.")

    def parameter_list(self, m=5, k=5, tor=3, pol=8):
        iota = [_ for _ in itertools.product(range(1, tor + 1), range(-pol, pol + 1)) if np.gcd(*_) == 1]
        self.parameters = list(itertools.product([0], range(1, k + 1), [(1, 0)], ["cos"])) + list(
            itertools.product(range(1, m + 1), range(1, k + 1), iota, ["cos", "sin"]))
        self.j_mat = j_mk(m + 1, k + 1)

    def mode_matrix(self):

        print(multi.cpu_count())
        with Pool(multi.cpu_count()) as p:
            pmap = p.imap(self.fb, self.parameters)
            result = sparse.vstack(list(tqdm(pmap, total=len(self.parameters))))

        return result


class OpticalSystem:

    def sim_dir(self):
        main_path = Path("./data/" + self.sim_name)
        if main_path.exists():
            print(main_path, "exists.")
            _ = list(map(print, [p for p in main_path.parent.glob("*") if
                                 re.search(main_path.stem + r"(\Z|\(\d*\))", p.stem)]))

            while True:
                c = input("Add(a)/Overwrite(w)/New(n)/Quit(q): ")
                if c == "q":
                    quit()
                elif c == "w":
                    self.over_write = True
                    return main_path
                elif c == "a":
                    return main_path
                elif c == "n":
                    new_path = rename(main_path)
                    return new_path
                else:
                    print("You can input only a/w/n/q.")

        else:
            return main_path

    def mk_mask(self):
        # maskの中心点を計算 単位はpixel
        center_points = self.mat_A.dot([0, 0, self.aperture_depth, 1]).reshape(self.hole_num, 2) / self.aperture_depth

        # 半径を計算
        mask_r = (self.f * (self.aperture_phi - self.hole_size) / self.aperture_depth + self.hole_size) * self.k_uv / 2

        # グリッドの生成 ピクセルの中心の座標を計算するために+0.5 ex) (0,0)と(1,1)を対角とするピクセルの中心座標は(0.5,0.5)
        im_0 = np.array([[v, u] for u in range(self.sim_image_size[0]) for v in range(self.sim_image_size[1])]) + 0.5

        # 円の中心との距離<mask_rか判定
        mask_list = [(((im_0 - c) / mask_r) ** 2).sum(axis=1) <= 1 for c in center_points]

        # アパーチャの影がFalse, それ以外がTrueになるarray (shape=self.sim_image_size)
        effective_area = np.sum(mask_list, axis=0, dtype=bool)

        return mask_list, effective_area

    def mk_kernel(self):
        # r_0はピンホールの半径(単位はピクセル)
        r_0 = (self.hole_size / 2) * self.k_uv
        # kernel_width
        kernel_width = np.ceil(r_0)
        # 中心からの距離を算出してwidthから引く
        r_array = np.array(
            np.meshgrid(*[np.arange(s, e) for s, e in zip(-kernel_width - 1, kernel_width + 1 + 1)], indexing="ij"))
        # 円の内部は1外部は0
        kernel = np.where(((r_array / r_0.reshape(-1, 1, 1)) ** 2).sum(axis=0) <= 1, 1.0, 0.0)

        # 正規化
        return kernel / kernel.sum()

    def light_vector(self, tm=False):
        if tm:
            voxel = np.r_["0,2,1", self.plasma_data.voxel[:-1], np.where(self.plasma_data.r <= 1, 1, 0)]
            active_voxel = self.mat_Rt.dot(voxel[:, voxel[-1, :] != 0])
        else:
            active_voxel = self.mat_Rt.dot(self.plasma_data.voxel[:, self.plasma_data.voxel[-1, :] != 0])

        if active_voxel.size:
            uv = (np.dot(self.mat_A, active_voxel[:4]) / active_voxel[2]).reshape(self.hole_num, 2, -1)
            r2_list = np.sum((active_voxel[None, :2] - self.h_xy[..., None]) ** 2, axis=1) + active_voxel[2] ** 2
            luminosity = (active_voxel[-1] / r2_list).reshape(self.hole_num, 1, -1)
            # breakpoint()
            return np.append(uv, luminosity, axis=1)
        else:
            print('There is no light points!')

    def mk_image_vec(self):
        image_vec = self.trans_mat_org(tm=False).sum(axis=1)
        return image_vec

    def trans_mat_org(self, tm=True):
        light_vector = self.light_vector(tm=tm)
        # light_vector -> ピクセル(整数値)に変換 これがrow
        row_list = np.dot([1, self.sim_image_size[1]], np.floor(light_vector[:, :2, :])).astype("i4")

        # columnのインデックス(hole_num分)
        columns = np.tile(np.arange(row_list.shape[-1]).astype("i4"), (self.hole_num, 1))
        # holeごとに luminosity/row/column の順にまとめる
        index_list = [index[:, (index[1] >= 0) & (index[1] < self.image_vec_size)] for index in
                      np.stack([light_vector[:, -1, :], row_list, columns], axis=1)]

        # trans_mat_orgのshape
        shape = (self.image_vec_size, self.plasma_data.J)
        # holeごとに処理
        mat_list = [sparse.csc_matrix((data, (row, col)), shape=shape) for data, row, col in index_list]
        # maskを適応(要素積)
        return np.sum([image_mat.multiply(mask[:, None]) for image_mat, mask in zip(mat_list, self.mask_list)], axis=0)
        # breakpoint()

    def blur_mat(self):
        E = sparse.identity(self.image_vec_size, dtype='i2', format='csr')
        E.data = self.effective_area.astype(float)
        print(multi.cpu_count())
        # breakpoint()
        with Pool(multi.cpu_count()) as p:
            pmap = p.imap(self.pinhole_blur, range(self.image_vec_size))
            M = sparse.vstack(list(tqdm(pmap, total=self.image_vec_size))).T

        return self.image_trans_mat * M

    def pinhole_blur(self, i):
        m = np.zeros(self.image_vec_size)
        m[i] = 1.0
        result = ndimage.convolve(m.reshape(self.sim_image_size), self.kernel, mode='constant', cval=0).ravel()
        return sparse.coo_matrix(result)

    def fb_modes(self):
        self.plasma_data.parameter_list(*self.parameters_range)
        self.fb_path.mkdir()
        print(multi.cpu_count())
        mode_arr = np.array(self.plasma_data.parameters, dtype='O')
        self.plasma_data.rtp()

        Parallel(n_jobs=self.thread_num, verbose=10, prefer='threads')(
            [delayed(self.fb_save)(i, para) for i, para in enumerate(self.plasma_data.parameters)])

        np.save(self.path / "mode_array.npy", mode_arr)

        with open(self.path / "mode_list.txt", "w") as f:
            n = int(np.log10(len(self.plasma_data.parameters)) + 1)
            header = f"{'i'.center(n)}|{'m'.center(5)}{'k'.center(5)}{'iota'.center(7)}{'tri'.center(5)}|" * 4
            f.write(f"{header}\n{'-' * len(header)}\n")
            for i, para in enumerate(self.plasma_data.parameters):
                m, k, (tor, pol), tri = para
                text = f"{str(i).center(n)}|{str(m).center(5)}{str(k).center(5)}{f'{pol}/{tor} '.rjust(7)}{tri.center(5)}|"
                f.write(text)
                if (i + 1) % 4 == 0:
                    f.write("\n")

    def fb_save(self, i, para):
        result = self.plasma_data.fb(para)
        sparse.save_npz(self.fb_path / f"mode_No{i}.npz", result)
        del result
        return 0

    def __init__(self, sim_name=None, auto=False, tm=False, save_option="", thread_num=-1,
                 h_xy=None, hole_z=948, f=14.3, aperture_depth=58, aperture_phi=20,
                 screen_size=(17.0, 17.0), hole_size=0.5, image_size=(170, 170), n=10,
                 shape=None, x_range=None, y_range=None, z_range=None, parameters_range=None):
        """

        Parameters
        ----------
        sim_name
        auto
        tm
        save_option
        thread_num
        h_xy
        hole_z
        f
        aperture_depth
        aperture_phi
        screen_size
        hole_size
        image_size
        n
        shape
        x_range
        y_range
        z_range
        parameters_range
        """

        self.sim_name = sim_name if sim_name else str(datetime.date.today())
        self.over_write = False
        self.save_option = save_option
        self.path = self.sim_dir()
        self.fb_path = rename(self.path / "fb_mode/")

        #  -*- tree-view -*-
        # ./
        # ┣ pinhole.py
        # ┣ shooting.py
        # ┣ calc.py
        # ┣ data/
        # ┃   ┣ sim_name/
        # ┃   ┃   ┣ trans_mat_org.npz
        # ┃   ┃   ┣ blur_mat.npz
        # ┃   ┃   ┣ fb_matrix.npz
        # ┃   ┃   ┣ fb_modes/
        # ┃   ┃   ┃   ┣ mode_No0.npz
        # ┃   ┃   ┃   ┣ mode_No1.npz
        #

        self.thread_num = thread_num
        if h_xy is None:
            h_xy = [[0.00, 0.00],
                    [-2.50, 4.33],
                    [2.50, 4.33],
                    [5.00, 0.00],
                    [2.50, -4.33],
                    [-2.50, -4.33],
                    [-5.00, 0.00]]
        if parameters_range is None:
            parameters_range = [5, 5, 3, 8]

        # pinhole
        self.f = f
        self.hole_num = len(h_xy)
        self.h_xy = np.array(h_xy)
        self.hole_size = hole_size
        self.screen_size = np.asarray(screen_size)
        self.aperture_depth = aperture_depth
        self.aperture_phi = aperture_phi
        self.offset = self.screen_size / 2

        # image
        self.sim_image_size = np.array([image_size[0] * n, image_size[1] * n])
        self.return_image_size = image_size
        self.image_trans_mat = trans_mat2d(*image_size, n)
        self.k_uv = self.sim_image_size / screen_size
        self.image_vec_size = int(np.prod(self.sim_image_size))

        # FB
        self.parameters_range = parameters_range

        # matrix
        self.mat_Rt = np.array([[-1, 0, 0, 0, 0],  # X=-x -> u
                                [0, 0, -1, 0, 0],  # Y=-z -> v
                                [0, -1, 0, hole_z, 0],  # Z=-y+hole_z
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1]])
        self.mat_k = np.diag(np.tile(self.k_uv, self.hole_num))
        c_XY = (self.h_xy + self.offset).ravel()
        r_XY = (f * self.h_xy).ravel()
        self.mat_A = self.mat_k @ np.r_["1,2,0",
                                        [-f, 0] * self.hole_num,
                                        [0, -f] * self.hole_num,
                                        c_XY, r_XY]

        # mask
        self.mask_list, self.effective_area = self.mk_mask()

        # plasma
        self.plasma_data = Plasma(shape=shape, x_range=x_range, y_range=y_range, z_range=z_range)

        # check voxel size
        d_i = abs((self.mat_Rt[:3, :4] @ self.plasma_data.voxel[:4])[2].min() * n / (self.f * self.k_uv))
        v_d = abs(self.mat_Rt[:2, :3] @ self.plasma_data.d_xyz)
        print(f"v_d={np.round(v_d, decimals=2)}, d={np.round(d_i, decimals=2)}")
        if np.any(v_d > d_i):
            print("Warning: Voxel size is inadequate. ",
                  f"(Voxel size={np.round(v_d, decimals=2)}, d={np.round(d_i, decimals=2)})")
            while True:
                c = "y" if auto else input("continue? (y)/n: ")
                if c == "n":
                    quit()
                elif c == "y":
                    break
                else:
                    pass

        # kernel
        self.kernel = self.mk_kernel()

        if tm and save_option:
            self.save_transmission_matrix()

    def simulate(self, fast_mode=False, image_save=True, return_image=True, show=False):
        org_sim_im = self.mk_image_vec().reshape(self.sim_image_size)

        # 畳み込み
        blur_sim_im = ndimage.convolve(org_sim_im, self.kernel, mode='constant', cval=0)
        # print(blur_sim_im.nonzero())
        # breakpoint()

        org_im = self.image_trans_mat.dot(org_sim_im.ravel().T).reshape(self.return_image_size)
        blur_im = self.image_trans_mat.dot(blur_sim_im.ravel().T).reshape(self.return_image_size)

        if show:
            plt.imshow(blur_sim_im)
            plt.show()
            plt.imshow(blur_im)
            plt.show()
        if fast_mode:
            return blur_im.ravel()
        else:
            org_pil = Image.fromarray((org_im * 255 / org_im.max()).astype("uint8"))
            blur_pil = Image.fromarray((blur_im * 255 / blur_im.max()).astype("uint8"))

            if image_save:
                im_path = rename(self.path + "/simulate")
                org_pil.save(im_path / "org_image.png")
                Image.fromarray((org_sim_im * 255 / org_sim_im.max()).astype("uint8")).save(im_path / "org_big.png")
                blur_pil.save(im_path / "blur_image.png")
                Image.fromarray((blur_sim_im * 255 / org_sim_im.max()).astype("uint8")).save(im_path / "blur_big.png")

            if return_image:
                return org_im.ravel(), blur_im.ravel(), org_pil, blur_pil
            else:
                return org_im.ravel(), blur_im.ravel()

    def save_transmission_matrix(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if "t" in self.save_option:
            sparse.save_npz(rename(self.path / "trans_mat_org.npz", self.over_write), self.trans_mat_org())
            print("trans_mat_org.npz: saved!")
        if "b" in self.save_option:
            sparse.save_npz(rename(self.path / "blur_mat.npz", self.over_write), self.blur_mat())
            print("blur_mat.npz: saved!")
        if "f" in self.save_option:
            self.fb_modes()
            print("Fourier-Bessel modes: saved!")


if __name__ == '__main__':
    dic = {"sim_name": "Test", "image_size": (128, 128), "shape": (333, 511, 333),
           "auto": False, "n": 10, "aperture_phi": 21, "f": 14.3, "parameters_range": [1, 2, 2, 1],
           # "h_xy": [[0, 0], ],
           # "save_option": "fb"
           }
    time_set = time.time()
    opti = OpticalSystem(**dic)
    j = np.zeros(dic["shape"])
    # j = np.ones(dic["shape"])
    # lena = np.asarray(Image.open("lena.jpg").convert("L").resize((dic["shape"][0], dic["shape"][2]))).astype(float)
    # j[:, 0, :] = lena
    # j[::3, 10, ::10] = 1
    j[80:120, 200:250, :] = 1
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.imshow(j[:, 10, :].T[::-1, ::-1])
    # plt.show()

    opti.plasma_data.voxel[-1] = j.ravel()
    opti.simulate(fast_mode=False, image_save=False, return_image=False, show=True)
