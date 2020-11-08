import numpy as np
from PIL import Image
import datetime
from pathlib import Path
import re
from scipy import ndimage
from scipy import sparse
import matplotlib.pyplot as plt


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


class Plasma:  # 仮想プラズマ

    def __init__(self, shape=None, xyz_range=None, o_xyz=None):
        # shape: ボクセルのマス xyz_range: プラズマの範囲 o_xyz: 中心←いる？
        super().__init__()
        if shape is None:
            shape = [10, 10, 10]
        if xyz_range is None:
            xyz_range = [100, 100, 100]
        if o_xyz is None:
            o_xyz = [0, 0, 300]

        self.xyz_range = np.array(xyz_range)
        self.d_xyz = self.xyz_range / shape  # size of voxel
        self.o_xyz = o_xyz

        voxel_num = np.prod(shape)

        # voxel_start = o_xyz - np.array(xyz_range) / 2

        voxel_start = o_xyz - self.xyz_range / 2 + self.d_xyz / 2
        voxel_end = o_xyz + self.xyz_range / 2 + self.d_xyz / 2

        x, y, z = [np.linspace(start, end, step, endpoint=False) for start, end, step in
                   zip(voxel_start, voxel_end, shape)]

        # xxx, yyy, zzz = np.meshgrid(x,y,z)
        zzz, yyy, xxx = np.meshgrid(z, y, x, indexing="ij")
        self.voxel = np.r_["1,2,0", xxx.ravel(), yyy.ravel(), zzz.ravel(), np.ones(voxel_num), np.zeros(voxel_num)].T
        # self.voxel = np.array([np.append(self.d_xyz * (np.array([i, j, k]) + 0.5) + voxel_start, [1, 0])
        #                        for k in range(shape[2])
        #                        for j in range(shape[1])
        #                        for i in range(shape[0])]).T  # voxel_size * voxel_num -> x,y,z


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
        self.mask_list = [(np.linalg.norm(im_0 - c, axis=1) <= mask_r) for c in self.center_points]

        # self.effective_area = sum(self.mask_list).astype(bool)
        # アパーチャの影がFalse, それ以外がTrueになるarray (shape=self.sim_image_size)
        self.effective_area = np.sum(self.mask_list, axis=0, dtype=bool)

    def mk_light_vector(self):  # 透視投影なのでxyの向きはそのまま、実際のピンホール画像は上下左右逆なので注意

        if self.plasma_data:
            active_voxel = self.plasma_data.voxel[:, self.plasma_data.voxel[-1, :] != 0]

            if active_voxel.size:
                uv = (np.dot(self.mat_P, active_voxel[:4]) / active_voxel[2]).reshape(self.hole_num, 2, -1)
                # r_2_list = [np.sum((active_voxel[:3].T - h) ** 2, axis=1) for h in self.hole_xyz]
                r_list = np.linalg.norm(active_voxel[:3] - self.hole_xyz[:, :, np.newaxis], axis=-2)
                luminosity = (active_voxel[-1] / r_list ** 2)[:, np.newaxis, :]

                # if self.mode == "lens":
                #     uv_ = uv - o.hole_xyz[:, :2, None]
                #     uv = (np.dot(self.mat_P, active_voxel[:4]) /
                #     np.linalg.norm(self.hole_xyz.reshape(-1,1,3)-active_voxel[:3])).reshape(self.hole_num, 2, -1)
                #     # r_2_list = [np.sum((active_voxel[:3].T - h) ** 2, axis=1) for h in self.hole_xyz]
                #     r_2_list = np.linalg.norm(active_voxel[:3] - self.hole_xyz[:, :, np.newaxis], axis=-2)
                #     luminosity = (active_voxel[-1] / r_2_list**2)[:, np.newaxis, :]

                self.light_vector = np.append(uv, luminosity, axis=1)
            else:
                print('There is no light points!')
        else:
            print('No plasma data. Please make plasma data.')

    def mk_image_vec(self):
        vec_list = []
        for datum, mask in zip(self.light_vector, self.mask_list):
            vec_list.append(np.zeros_like(self.effective_area, dtype=float))
            index = np.dot([1, self.sim_image_size[0]], np.floor(datum[:2] * self.ppmm)).astype(int)
            index_set = np.unique(index)
            luminosity = np.array(list(map(sum, [datum[-1, c] for c in [index == i for i in index_set]])))
            # vec_list[-1][index[index<=self.effective_area.size]] = datum[-1,index<=self.effective_area.size]
            flag = index_set <= self.effective_area.size
            # flag = np.all([index_set <= self.effective_area.size, index_set >= 0], axis=0)
            vec_list[-1][index_set[flag]] = luminosity[flag]
            vec_list[-1] = vec_list[-1] * mask

        image_vec = sum(vec_list)
        return image_vec

    def __init__(self, sim_name=None, mode="pinhole", auto=False,
                 hole_list=None, f=14.3, screen_size=(17.0, 17.0), hole_size=0.5, aperture_z=58, aperture_phi=21,
                 shape=(10, 10, 10), xyz_range=(100, 100, 100), o_xyz=(0, 0, 300), image_size=(170, 170), n=10):

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

        self.f = f
        self.hole_xyz = np.array([[*h, 0.0] for h in hole_list])
        self.hole_num = len(hole_list)
        self.hole_size = hole_size
        self.screen_size = np.asarray(screen_size)

        self.sim_image_size = (image_size[0] * n, image_size[1] * n)
        # self.sim_image_size = image_size
        self.return_image_size = image_size
        self.image_trans_mat = trans_mat2d(*image_size, n)
        self.ppmm = self.sim_image_size[0] / screen_size[0]
        self.offset = self.screen_size / 2
        self.plasma_data = Plasma(shape, xyz_range, o_xyz)

        self.__stop__ = False

        d = (o_xyz[-1] - xyz_range[-1] / 2) / (self.f * self.ppmm) * n

        # print(self.plasma_data.d_xyz[:2], d)

        if np.any(self.plasma_data.d_xyz[:2] > d):
            print("Warning: Voxel size is inadequate. ",
                  f"(Voxel size={np.round(self.plasma_data.d_xyz[:2], decimals=2)}, d={d:.2})")
            print(f"Ideal shape: {(self.plasma_data.xyz_range[:2] / d).astype(int) + 1}")
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

        self.mat_P = np.r_["1,2,0", [-f, 0] * self.hole_num,
                           [0, -f] * self.hole_num,
                           (self.hole_xyz[:, :2] + self.offset).ravel(),
                           (f * self.hole_xyz[:, :2]).ravel()]

        self.center_points = None
        self.mask_list = None
        self.effective_area = None
        self.light_vector = np.empty(shape=(self.hole_num, 3, 0), dtype=int)

    def simulate(self, fast_mode=False, image_save=True, return_image=True, show=False):
        if self.__stop__:
            return 0

        if not self.mask_list:
            self.mk_mask()
        self.mk_light_vector()
        org_sim_im = self.mk_image_vec().reshape(self.sim_image_size)
        # nはピンホールの半径(単位はピクセル)
        n = int((self.hole_size / 2) * self.ppmm)
        # 中心からの距離を算出してnから引く
        r_array = n - np.linalg.norm(np.meshgrid(np.arange(-n - 1, n + 1 + 1), np.arange(-n - 1, n + 1 + 1)), axis=0)
        # 円の内部は1外部は0
        kernel = np.where(r_array >= 0, 1.0, 0.0)
        # 正規化
        kernel = kernel / kernel.sum()
        # print(kernel.shape)
        # 畳み込み
        blur_sim_im = ndimage.convolve(org_sim_im, kernel, mode='constant', cval=0)
        # print(blur_sim_im.nonzero())

        org_im = self.image_trans_mat.dot(org_sim_im.ravel()).reshape(self.return_image_size)
        blur_im = self.image_trans_mat.dot(blur_sim_im.ravel()).reshape(self.return_image_size)

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


if __name__ == '__main__':
    dic = {"sim_name": None, "mode": "pinhole", "image_size": (128, 128), "shape": (50, 50, 100),
           "xyz_range": (100, 100, 100), "o_xyz": (0, 0, 300)}
    o = OpticalSystem(**dic)
    o.plasma_data.voxel[-1, np.linalg.norm(o.plasma_data.voxel[:3] - [[30], [20], [300]], axis=0) < 10] = 10
    # o.plasma_data.voxel[-1, 100] = 10
    # print(o.plasma_data.voxel[:, o.plasma_data.voxel[-1, :] != 0])

    o.simulate(show=True, image_save=True)
