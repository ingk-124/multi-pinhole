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


def dir_rename(dir_name):
    dir_path = Path(dir_name)
    parent = dir_path.parent
    if dir_path.exists():
        n = len([p for p in parent.glob("*") if re.search(dir_path.stem + r"(\Z|\(\d*\))", p.stem)])
        new_path = parent / "{}({})".format(dir_path.stem, n)
        new_path.mkdir(parents=True)
        return new_path
    else:
        dir_path.mkdir(parents=True)
        return dir_path


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


class Plasma:  # 仮想プラズマ

    def __init__(self, shape=None, xyz_range=None, start_xyz=None, R=508, a=250):
        # shape: ボクセルのマス xyz_range: プラズマの範囲 o_xyz: 中心←いる？
        super().__init__()
        if shape is None:
            shape = [10, 10, 10]
        if xyz_range is None:
            xyz_range = [500, 758, 500]
        if start_xyz is None:
            start_xyz = [0, xyz_range[1], 0]

        self.shape = np.array(shape)
        self.xyz_range = np.array(xyz_range)
        self.start_xyz = start_xyz
        self.R = R
        self.a = a

        self.d_xyz = self.xyz_range / shape  # size of voxel
        self.offset = start_xyz - np.array([0, xyz_range[1], 0]) / 2

        self.J = int(np.prod(shape))
        self.j_mat = None
        self.parameters = None

        start = self.xyz_range * (1 - self.shape) / (2 * self.shape)
        end = -start

        x, y, z = [np.linspace(a, b, s) + o for a, b, s, o in zip(start, end, self.shape, self.offset)]

        # voxel_start = o_xyz - self.xyz_range / 2 + self.d_xyz / 2
        # voxel_end = o_xyz + self.xyz_range / 2 + self.d_xyz / 2
        #
        # x, y, z = [np.linspace(start, end, step, endpoint=False) for start, end, step in
        #            zip(voxel_start, voxel_end, shape)]  # x -> y ->zの順

        xxx, yyy, zzz = np.meshgrid(x, y, z, indexing="ij")
        # zzz, yyy, xxx = np.meshgrid(z, y, x, indexing="ij")

        self.voxel = np.r_[
            "1,2,0", xxx.ravel(), yyy.ravel(), zzz.ravel(), np.ones(self.J), np.zeros(self.J)].T

        # self.voxel = np.array([np.append(self.d_xyz * (np.array([i, j, k]) + 0.5) + voxel_start, [1, 0])
        #                        for k in range(shape[2])
        #                        for j in range(shape[1])
        #                        for i in range(shape[0])]).T  # voxel_size * voxel_num -> x,y,z

        # j_mat=np.vstack([j_mk(m) for m in range(25)])

        # self.r, self.theta, self.phi = self.rtp()

    def rtp(self):
        r_xy = np.linalg.norm(self.voxel[:2], axis=0) - self.R
        r = np.linalg.norm([r_xy, self.voxel[2]], axis=0) / self.a
        theta = np.arctan2(self.voxel[2], r_xy)
        phi = np.arctan2(*self.voxel[:2])

        return r, theta, phi

    def fb(self, para, r, theta, phi):
        m, k, (tor, pol), cs = para
        t = pol / tor
        psy = 2 * np.pi * t

        if cs == "c":
            luminosity = sum(
                [jv(m, self.j_mat[m, k - 1] * r) * np.cos(m * (theta + t * phi + psy * _)) for _ in
                 range(tor)])
            return sparse.csr_matrix(np.where(r > 1, 0, luminosity))
        elif cs == "s":
            luminosity = sum(
                [jv(m, self.j_mat[m, k - 1] * r) * np.sin(m * (theta + t * phi + psy * _)) for _ in
                 range(tor)])
            return sparse.csr_matrix(np.where(r > 1, 0, luminosity))

    def parameter_list(self, m_max=5, k_max=5, tor_max=3, pol_max=8):
        t = [_ for _ in itertools.product(range(1, tor_max + 1), range(0, pol_max + 1)) if np.gcd(*_) == 1]
        self.parameters = list(itertools.product([0], range(1, k_max + 1), [(1, 0)], ["c"])) + list(
            itertools.product(range(1, m_max + 1), range(1, k_max + 1), t, ["c", "s"]))
        self.j_mat = j_mk(m_max + 1, k_max + 1)

    def mode_matrix(self):

        print(multi.cpu_count())
        with Pool(multi.cpu_count()) as p:
            pmap = p.imap(self.fb, self.parameters)
            result = sparse.vstack(list(tqdm(pmap, total=len(self.parameters))))

        return result


class OpticalSystem:

    def mk_mask(self):
        # maskの中心点を計算
        self.center_points = np.dot([0, 0, self.aperture_z, 1], self.mat_P.T).reshape(self.hole_num,
                                                                                      2) * self.ppmm / self.aperture_z
        # 半径を計算
        # mask_r = (self.f * self.aperture_phi * self.ppmm) / (2 * self.aperture_z)
        mask_r = ((self.f * (self.aperture_phi - self.hole_size)) / self.aperture_z + self.hole_size) * self.ppmm / 2

        # グリッドの生成 ピクセルの中心の座標を計算するために+0.5 ex) (0,0)と(1,1)を対角とするピクセルの中心座標は(0.5,0.5)
        im_0 = np.array([[i, j] for j in range(self.sim_image_size[1]) for i in range(self.sim_image_size[0])]) + 0.5

        # 円の中心との距離<mask_rか判定
        mask_list = [(np.linalg.norm(im_0 - c, axis=1) <= mask_r) for c in self.center_points]

        # self.effective_area = sum(self.mask_list).astype(bool)
        # アパーチャの影がFalse, それ以外がTrueになるarray (shape=self.sim_image_size)
        effective_area = np.sum(mask_list, axis=0, dtype=bool)

        return mask_list, effective_area

    def light_vector(self, tm=False):

        if tm:
            active_voxel = self.mat_t.dot(np.r_["0,2,1", self.plasma_data.voxel[:-1], np.ones(self.plasma_data.J)])
        else:
            active_voxel = self.mat_t.dot(self.plasma_data.voxel[:, self.plasma_data.voxel[-1, :] != 0])

        if active_voxel.size:
            uv = (np.dot(self.mat_P, active_voxel[:4]) / active_voxel[2]).reshape(self.hole_num, 2, -1)
            # r_2_list = [np.sum((active_voxel[:3].T - h) ** 2, axis=1) for h in self.hole_xyz]

            if self.mode == "lens":
                luminosity = np.ones_like(uv[:, 0, :])[:, np.newaxis, :]
            else:
                r_list = np.linalg.norm(active_voxel[:3] - self.hole_xyz[:, :, np.newaxis], axis=-2)
                luminosity = (active_voxel[-1] / r_list ** 2)[:, np.newaxis, :]
            return np.append(uv, luminosity, axis=1)
        else:
            print('There is no light points!')

    def mk_image_vec(self):
        self.trans_mat_org(tm=False)
        image_vec = self.tm_org.sum(axis=1)
        return image_vec

    def trans_mat_org(self, tm=True):
        if not self.mask_list:
            self.mk_mask()
        light_vector = self.light_vector(tm=tm)
        # light_vector -> ピクセル(整数値)に変換 これがrow
        row_list = np.dot([self.sim_image_size[0], 1], np.floor(light_vector[:, :2, :] * self.ppmm)).astype("i4")

        # columnのインデックス(hole_num分)
        if tm:
            columns = np.tile(np.arange(self.plasma_data.J).astype("i4"), (self.hole_num, 1))
        else:
            columns = np.tile(np.arange(light_vector.shape[-1]).astype("i4"), (self.hole_num, 1))
        # holeごとに luminosity/row/column の順にまとめる
        index_list = [index[:, (index[1] >= 0) & (index[1] < self.I)] for index in
                      np.stack([light_vector[:, -1, :], row_list, columns], axis=1)]

        # tm_orgのshape
        shape = (self.I, self.plasma_data.J)
        # holeごとに処理
        mat_list = [sparse.csc_matrix((data, (row, col)), shape=shape) for data, row, col in index_list]
        # maskを適応(要素積)
        self.tm_org = np.sum(
            [image_mat.multiply(mask[:, None]) for image_mat, mask in zip(mat_list, self.mask_list)], axis=0)

    def blur_mat(self):
        E = sparse.identity(self.I, dtype='i2', format='csr')
        E.data = self.effective_area.astype(float)
        print(multi.cpu_count())
        # breakpoint()
        with Pool(multi.cpu_count()) as p:
            pmap = p.imap(self.pinhole_blur, range(self.I))
            M = sparse.vstack(list(tqdm(pmap, total=self.I))).T

        self.bm = self.image_trans_mat * M

    def fb_modes(self):
        self.plasma_data.parameter_list(*self.parameter_max)
        self.path = dir_rename("./fb_mode/" + self.sim_name)
        print(multi.cpu_count())
        # with Pool(multi.cpu_count()) as pool:
        #     pmap = pool.imap(self.fb_image, self.plasma_data.parameters)
        #     _ = list(tqdm(pmap, total=len(self.plasma_data.parameters)))
        mode_arr = np.array(self.plasma_data.parameters, dtype='O')
        # breakpoint()
        r, t, p = self.plasma_data.rtp()
        Parallel(n_jobs=10, verbose=10, prefer='threads')(
            [delayed(self.fb_image)(para, r, t, p) for para in self.plasma_data.parameters])
        # Parallel(n_jobs=-1, verbose=10)([delayed(self.fb_image)(p) for p in self.plasma_data.parameters])
        np.save(self.path / "mode_array.npy", mode_arr)
        n = int(np.log10(len(self.plasma_data.parameters)) + 1)
        with open(self.path / "mode_list.txt", "w") as f:
            for i, m in enumerate(self.plasma_data.parameters):
                f.write(f"{i:>{n}}: {tuple(m)}  ")
                if (i + 1) % 4 == 0:
                    f.write("\n")

    def fb_image(self, p, r, theta, phi):
        # directory = dir_rename(self.path / f"{p}")
        result = self.plasma_data.fb(p, r, theta, phi)
        # print(sum([result.indptr.nbytes, result.indices.nbytes, result.data.nbytes]))
        # np.save(directory / "data.npy", result.data)
        # np.save(directory / "indices.npy", result.indices)
        # np.save(directory / "indptr.npy", result.indptr)
        sparse.save_npz(self.path / f"{p}.npz", result)
        del result
        return 0

    def mk_kernel(self):
        # widthはピンホールの半径(単位はピクセル)
        kernel_width = int((self.hole_size / 2) * self.ppmm)
        # 中心からの距離を算出してwidthから引く
        r_array = kernel_width - np.linalg.norm(np.meshgrid(np.arange(-kernel_width - 1, kernel_width + 1 + 1),
                                                            np.arange(-kernel_width - 1, kernel_width + 1 + 1)), axis=0)
        # 円の内部は1外部は0
        kernel = np.where(r_array >= 0, 1.0, 0.0)
        # 正規化
        return kernel / kernel.sum()

    def __init__(self, sim_name=None, mode="pinhole", auto=False, tm=False, save_option="",
                 hole_list=None, hole_z=948, f=14.3, aperture_z=58, aperture_phi=21,
                 screen_size=(17.0, 17.0), hole_size=0.5, image_size=(170, 170), n=10,
                 shape=None, xyz_range=None, start_xyz=None,
                 parameter_max=None):

        self.mode = mode
        self.sim_name = sim_name if sim_name else str(datetime.date.today())

        if hole_list is None:
            hole_list = [[0.00, 0.00],
                         [-2.50, 4.33],
                         [2.50, 4.33],
                         [5.00, 0.00],
                         [2.50, -4.33],
                         [-2.50, -4.33],
                         [-5.00, 0.00]]
        if parameter_max is None:
            parameter_max = [5, 5, 3, 8]

        self.f = f
        self.hole_xyz = np.array([[*h[::-1], 0.0] for h in hole_list])
        self.hole_num = len(hole_list)
        self.hole_size = hole_size
        self.screen_size = np.asarray(screen_size)

        self.sim_image_size = (image_size[0] * n, image_size[1] * n)
        self.return_image_size = image_size
        self.image_trans_mat = trans_mat2d(*image_size, n)
        self.ppmm = self.sim_image_size[0] / screen_size[0]
        self.offset = self.screen_size / 2
        self.plasma_data = Plasma(shape, xyz_range, start_xyz)
        self.I = int(np.prod(self.sim_image_size))
        self.parameter_max = parameter_max

        self.__stop__ = False

        d = (hole_z - self.plasma_data.start_xyz[1]) * n / (self.f * self.ppmm)

        # print(self.plasma_data.d_xyz[:2], d)

        if np.any(self.plasma_data.d_xyz[[0, -1]] > d):
            print("Warning: Voxel size is inadequate. ",
                  f"(Voxel size={np.round(self.plasma_data.d_xyz[[0, -1]], decimals=2)}, d={d:.2})")
            print(f"Ideal shape: {(self.plasma_data.xyz_range[[0, -1]] / d).astype(int) + 1}")
            while True:
                c = "y" if auto else input("continue? (y)/n: ")
                if c == "n":
                    self.__stop__ = True
                    break
                elif c == "y":
                    break
                else:
                    pass

        self.aperture_z = aperture_z
        self.aperture_phi = aperture_phi

        self.mat_t = np.array([[0, 0, 1, 0, 0],
                               [-1, 0, 0, 0, 0],
                               [0, -1, 0, hole_z, 0],
                               [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1]])

        self.mat_P = np.r_["1,2,0",
                           [f, 0] * self.hole_num,
                           [0, f] * self.hole_num,
                           (self.hole_xyz[:, :2] + self.offset).ravel(),
                           (f * self.hole_xyz[:, :2]).ravel()]
        self.center_points = None
        self.mask_list, self.effective_area = self.mk_mask()

        self.kernel = self.mk_kernel()
        self.tm_org = None
        self.bm = None
        self.save_option = save_option
        self.path = None
        if tm:
            self.save_transmission_matrix()

    def pinhole_blur(self, i):
        m = np.zeros(self.I)
        m[i] = 1.0
        result = ndimage.convolve(m.reshape(self.sim_image_size), self.kernel, mode='constant', cval=0).ravel()
        return sparse.coo_matrix(result)

    def simulate(self, fast_mode=False, image_save=True, return_image=True, show=False):
        if self.__stop__:
            return 0

        if not self.mask_list:
            self.mk_mask()
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
                sim_path = dir_rename("./data/" + self.sim_name + "/simulate")
                org_pil.save(sim_path / "org_image.png")
                Image.fromarray((org_sim_im * 255 / org_sim_im.max()).astype("uint8")).save(sim_path / "org_big.png")
                blur_pil.save(sim_path / "blur_image.png")
                Image.fromarray((blur_sim_im * 255 / org_sim_im.max()).astype("uint8")).save(sim_path / "blur_big.png")

            if return_image:
                return org_im.ravel(), blur_im.ravel(), org_pil, blur_pil
            else:
                return org_im.ravel(), blur_im.ravel()

    def save_transmission_matrix(self):
        if self.__stop__:
            return 0

        if self.save_option == "bo":
            self.path = dir_rename("./npz/" + self.sim_name)
            self.blur_mat()
            sparse.save_npz(self.path / "blur_mat.npz", self.bm)
            print("blur_mat.npz: saved!")
        elif self.save_option == "oo":
            self.path = dir_rename("./npz/" + self.sim_name)
            self.trans_mat_org()
            sparse.save_npz(self.path / "trans_mat_org.npz", self.tm_org)
            print("trans_mat_org.npz: saved!")
        elif self.save_option == "fb":
            self.fb_modes()
            print("FourierBessel_mat.npz: saved!")
        else:
            self.blur_mat()
            path = dir_rename("./npz/" + self.sim_name)
            sparse.save_npz(path / "blur_mat.npz", self.bm)
            self.trans_mat_org()
            sparse.save_npz(path / "trans_mat_org.npz", self.bm)
            mat_A = self.bm * self.tm_org
            sparse.save_npz(path / "mat_A.npz", mat_A)
            print("mat_A.npz: saved!")


if __name__ == '__main__':
    v = 10
    dic = {"sim_name": "Test", "mode": "lens", "image_size": (128, 128), "shape": (10, 10, 10),
           "xyz_range": (498, 765, 498), "start_xyz": (0, 700, 0), "auto": False, "n": 1,
           "hole_list": [[5.0, 0], [-5.0, 0], [0, -5.0], [0, 5.0]],
           "parameter_max": [1, 1, 1, 1], "save_option": "fb"}
    time_set = time.time()
    os = OpticalSystem(**dic)
    # o.plasma_data.voxel[-1, :] = 0.1
    # os.plasma_data.voxel[-1, np.linalg.norm(os.plasma_data.voxel[:3] - [[0], [300], [100]], axis=0) < 50] = 100
    # os.plasma_data.voxel[-1, np.linalg.norm(os.plasma_data.voxel[:3] - [[-100], [300], [0]], axis=0) < 50] = 100
    # o.plasma_data.voxel[-1, 555] = 10
    # breakpoint()
    print("set: ", time.time() - time_set)
    time_set = time.time()
    # P = o.trans_mat_org()
    # print("trans mat: ", time.time() - t)
    # t = time.time()
    # B = os.blur_mat()
    # print("blur mat: ", time.time() - t)

    os.save_transmission_matrix()
    # o.trans_mat_org()
    # print("saved: ", time.time() - t)

    # print(o.plasma_data.voxel[:, o.plasma_data.voxel[-1, :] != 0])
    #
    # o_im, b_im = os.simulate(show=True, image_save=False, return_image=False)
    # print(b_im.nonzero())
    # breakpoint()
    # print(o.light_vector.shape)

    # pl = Plasma()
    # mode_ = pl.fb_modes()
    # mat = pl.mode_matrix()
    # print(mat.shape)
