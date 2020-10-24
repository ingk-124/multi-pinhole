import numpy as np
from PIL import Image
import datetime
from pathlib import Path
import re
from scipy import ndimage


def file_rename(original_name):
    file_path = Path(original_name)  # ディレクトリに分割 -> list
    parent = file_path.parent
    if parent.exists():
        pass
    else:
        parent.mkdir(parents=True)
    # n = len(list(parent.glob(file_path.stem + "*" + file_path.suffix)))
    n = len([p for p in parent.glob("*") if re.search(file_path.stem + r"(\Z|\(\d*\))", p.stem)])

    return parent / "{}({}){}".format(file_path.stem, n, file_path.suffix)  # nameと拡張子を結合

    # if n:
    #     filename = parent / "{}({}){}".format(file_path.stem, n, file_path.suffix)  # nameと拡張子を結合
    #     return filename
    # else:  # n=0のときoriginal_nameを返す
    #     return original_name


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

        self.d_xyz = np.array(xyz_range) / shape  # size of voxel
        voxel_num = np.prod(shape)

        # voxel_start = o_xyz - np.array(xyz_range) / 2

        voxel_start = o_xyz - np.array(xyz_range) / 2 + self.d_xyz / 2
        voxel_end = o_xyz + np.array(xyz_range) / 2 + self.d_xyz / 2

        x, y, z = [np.linspace(start, end, step, endpoint=False) for start, end, step in
                   zip(voxel_start, voxel_end, shape)]

        xxx, yyy, zzz = np.meshgrid(x, y, z)
        self.voxel = np.r_["1,2,0", xxx.ravel(), yyy.ravel(), zzz.ravel(), np.ones(voxel_num), np.zeros(voxel_num)].T
        # self.voxel = np.array([np.append(self.d_xyz * (np.array([i, j, k]) + 0.5) + voxel_start, [1, 0])
        #                        for k in range(shape[2])
        #                        for j in range(shape[1])
        #                        for i in range(shape[0])]).T  # voxel_size * voxel_num -> x,y,z


class OpticalSystem:

    def mk_mask(self):

        self.center_points = np.dot([0, 0, self.aperture_z, 1], self.mat_P.T).reshape(self.hole_num,
                                                                                      2) * self.ppmm / self.aperture_z

        mask_r = (self.f * self.aperture_phi * self.ppmm) / (2 * self.aperture_z)
        im_0 = np.array([[i, j] for j in range(self.image_size[1]) for i in range(self.image_size[0])]) + 0.5

        self.mask_list = [(np.linalg.norm(im_0 - c, axis=1) <= mask_r) for c in self.center_points]
        self.effective_area = sum(self.mask_list).astype(bool)

    def mk_light_vector(self):

        if self.plasma_data:
            active_voxel = self.plasma_data.voxel[:, self.plasma_data.voxel[-1, :] != 0]

            if active_voxel.size:
                uv = (np.dot(self.mat_P, active_voxel[:4]) / active_voxel[2]).reshape(self.hole_num, 2, -1)
                r_2_list = [np.sum((active_voxel[:3].T - h) ** 2, axis=1) for h in self.hole_xyz]
                luminosity = (active_voxel[-1] / r_2_list)[:, np.newaxis, :]
                self.light_vector = np.append(uv, luminosity, axis=1)
            else:
                print('There is no light points!')
        else:
            print('No plasma data. Please make plasma data.')

    def mk_image_vec(self):
        vec_list = []
        for datum, mask in zip(self.light_vector, self.mask_list):
            vec_list.append(np.zeros_like(self.effective_area, dtype=float))
            index = np.dot([1, self.image_size[0]], np.floor(datum[:2] * self.ppmm)).astype(int)
            index_set = np.unique(index)
            luminosity = np.array(list(map(sum, [datum[-1, c] for c in [index == i for i in index_set]])))
            # vec_list[-1][index[index<=self.effective_area.size]] = datum[-1,index<=self.effective_area.size]
            flag = index_set <= self.effective_area.size
            vec_list[-1][index_set[flag]] = luminosity[flag]
            vec_list[-1] = vec_list[-1] * mask

        image_vec = sum(vec_list)
        return image_vec

    def __init__(self, sim_name=None, hole_list=None, f=14.3, screen_size=(17.0, 17.0), hole_size=0.5, aperture_z=58,
                 aperture_phi=21, image_size=(170, 170), plasma_data=Plasma()):

        self.sim_name = sim_name if sim_name else str(datetime.date.today())
        self.sim_path = file_rename(Path("./data/" + self.sim_name + "/simulation"))

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
        self.image_size = image_size
        self.ppmm = image_size[0] / screen_size[0]
        self.offset = self.screen_size / 2

        self.plasma_data = plasma_data
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
        self.image_vec = None

    def simulate(self):
        self.sim_path.mkdir()
        self.mk_mask()
        self.mk_light_vector()
        org_im = self.mk_image_vec()
        n = (self.hole_size / 2) // 2
        r_array = n - np.linalg.norm(np.meshgrid(np.arange(-n - 1, n + 1 + 1), (np.arange(-n - 1, n + 1 + 1))), axis=0)
        kernel = np.where(r_array > 0, 1.0, 0.0)
        kernel = kernel / kernel / sum()
        blur_im = ndimage.convolve(org_im, kernel, mode='constant', cval=0)

        org_pil = Image.fromarray((org_im.reshape(self.image_size) * 255 / org_im.max()).astype("uint8"))
        org_pil.save(self.sim_path / "org_image.png")

        blur_pil = Image.fromarray((blur_im.reshape(self.image_size) * 255 / blur_im.max()).astype("uint8"))
        blur_pil.save(self.sim_path / "blur_image.png")

        # self.image_vec = org_im[self.effective_area]
        return org_im,blur_im,org_pil,blur_pil


if __name__ == '__main__':
    o = OpticalSystem()
    o.plasma_data.voxel[-1, np.linalg.norm(o.plasma_data.voxel[:3] - [[12], [0], [300]], axis=0) < 10] = 10

    o.simulate()
