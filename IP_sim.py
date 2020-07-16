import numpy as np
from scipy.sparse import csr_matrix
import scipy.signal as sp
from PIL import Image
import datetime
from pathlib import Path
from tqdm import tqdm
import tables


def file_rename(original_name):
    file_path = Path(original_name)  # ディレクトリに分割 -> list
    parent = file_path.parent
    if parent.exists():
        pass
    else:
        parent.mkdir(parents=True)
    n = len(list(parent.glob(file_path.stem + "*" + file_path.suffix)))
    if n:
        filename = parent / "{}({}){}".format(file_path.stem, n, file_path.suffix)  # nameと拡張子を結合
        return filename
    else:  # n=0のときoriginal_nameを返す
        return original_name


class Voxel:

    def __init__(self, shape=None, xyz_range=None, o_xyz=None):
        super().__init__()
        if shape is None:
            shape = [10, 10, 10, 5]
        if xyz_range is None:
            xyz_range = [100, 100, 100]
        if o_xyz is None:
            o_xyz = [0, 0, 300]

        d_xyz = np.array(xyz_range) / shape[:3]  # size of voxel
        voxel_start = o_xyz - np.array(xyz_range) / 2
        self.voxel = np.array([np.append(d_xyz * [i, j, k] + voxel_start, [1, 0]) for k in range(shape[2]) for j in
                               range(shape[1]) for i in range(shape[0])])  # voxel_size * voxel_num -> x,y,z
        self.M = self.voxel.shape[0]


class Pinhole:
    def __init__(self, hole_list=None, f=20, screen_size=(30, 30), hole_size=0.5):
        super().__init__()
        if hole_list is None:
            # hole_list = [[0.0, 0.0], [-2.50, 4.33], [2.50, 4.33], [5.0, 0.0],
            #              [2.50, -4.33], [-2.50, -4.33], [-5.0, 0.0]]
            hole_list = [[0.0, 0.0], [-2.50, -4.33], [2.50, 4.33]]

        self.f = f
        self.hole_xyz = list(np.array(hole_list))
        self.hole_num = len(hole_list)
        self.screen_size = np.asarray(screen_size)
        self.offset = self.screen_size / 2
        self.hole_size = hole_size

        self.mat_K_tensor = [np.vstack(([f, 0, 0, 0], [0, f, 0, 0],  # 1-2列目
                                        np.append(h + self.offset, [1, 0]),  # 3列目
                                        [0, 0, 0, 0], [0, 0, 0, 1])).T for h in
                             self.hole_xyz]  # make K matrix tensor(7*4*5)

        self.translation_tensor = [np.identity(5) + np.vstack(([[0] * 5] * 3, np.append(-v, [0, 0, 0]), [[0] * 5])).T
                                   for v in self.hole_xyz]  # make translation tensor (7*5*5)

        self.mat_P_tensor = np.einsum('ijk,ikl->ijl', self.mat_K_tensor, self.translation_tensor,
                                      optimize="optimal")  # calculate Projection matrix tensor (7*4*5)4*5


class CCDImage(Pinhole):

    def gen_psf(self, r=0.5, ismask: bool = False):
        n = int(np.ceil(r * self.ppmm - 0.5))
        d = np.linspace(-(n + 0.5), n + 0.5, 129)
        xx, yy = np.meshgrid(d, d)
        mesh = np.vstack([xx.flatten(), yy.flatten()]).T
        circ = mesh[np.linalg.norm(mesh, axis=1) <= r]
        p = circ.shape[0]
        x, y = np.round(circ).T.astype(int)
        k = csr_matrix((np.ones(p), (x - x.min(), y - y.min())),
                       shape=(x.max() - x.min() + 1, y.max() - y.min() + 1)).toarray()
        kernel = np.copy(k)

        if not ismask:
            kernel = kernel / kernel.sum()
        else:
            a = k.max() / 4
            kernel[k < a] = k[k < a] / a
            kernel[k >= a] = 1

        return np.round(kernel, decimals=3)

    def __init__(self, ppmm=5, image_size=(150, 150), threshold=10):
        super().__init__()
        self.ppmm = ppmm
        self.image_size = image_size
        self.psf = self.gen_psf(r=self.hole_size / 2)
        self.Im = np.zeros(self.image_size)
        self.threshold = threshold

    def shoot(self, point, image_mask):
        row, col = np.floor((point[:2]) * self.ppmm).astype(int)
        image_mat = np.copy(self.Im)
        image_mat[row, col] = point[2]
        image_mat = sp.convolve2d(image_mat, self.psf, boundary='fill', mode='same') * image_mask
        # image_mat = image_mat[image_mat>self.threshold]
        return image_mat


class OpticalSystem(Voxel, CCDImage):

    def light_vector(self, voxel):
        lambda_im = np.einsum('ijl,l->ij', self.mat_P_tensor, voxel, optimize="optimal")

        l_z = lambda_im[:, 2:3]
        r = np.array([np.linalg.norm(voxel[:3] - np.append(hole, 0)) for hole in self.hole_xyz])
        luminosity = lambda_im[:, -1] / (r ** 2)
        xy = lambda_im[:, :2] / l_z
        return np.concatenate((xy, luminosity[:, np.newaxis]), axis=1)

    def aperture(self):
        mask_r = self.f * self.aperture_alpha / (2 * self.aperture_z)
        center_points = np.einsum('ijl,ml->imj', self.mat_P_tensor, np.array([[0, 0, self.aperture_z, 1, 0]]),
                                  optimize="optimal")
        mask_center = center_points[:, 0, :2] / center_points[:, 0, 2:3]
        # print(f"{mask_center=}")
        return mask_r, mask_center

    def gen_mask(self, r, offset):
        mask = []
        for offset_i in offset:
            m = np.copy(self.Im)
            o = (offset_i * self.ppmm).astype(int)
            m[o[0], o[1]] = 1
            kernel = self.gen_psf(r, ismask=True)
            # kernel = self.gen_psf(r)
            mask.append(sp.convolve2d(m, kernel, boundary='fill', mode='same'))
        return mask

    def __init__(self, sim_name=None):
        super(OpticalSystem, self).__init__(shape=[100, 100, 100, 5])
        self.aperture_z = 100
        self.aperture_alpha = 120
        self.sim_name = sim_name if sim_name else str(datetime.date.today())
        self.sim_path = Path("./data/" + self.sim_name)
        self.mask_r, self.mask_center = self.aperture()
        self.image_mask = self.gen_mask(r=self.mask_r, offset=self.mask_center)
        self.mat_H = None

    def mk_image_data(self, voxel, save_image=False):
        lambda_im = self.light_vector(voxel)
        # print(f"{lambda_im=}")
        image_array = np.sum([self.shoot(l_i, image_mask=mask) for l_i, mask in zip(lambda_im, self.image_mask)],
                             axis=0)

        return np.ravel(image_array)

    def calculate(self):
        base_voxel = np.copy(self.voxel)
        base_voxel[:, -1] = 1
        print(f"{base_voxel.shape=}")

        filename = file_rename(self.sim_path / "npy/{}.npy".format(self.sim_name))
        f = tables.open_file(filename, mode='w')
        atom = tables.Float64Atom()
        array_c = f.create_earray(f.root, 'data', atom, (0, self.M))
        for base in tqdm(base_voxel):
            array_c.append(base)
        f.close()

    def voxel2img(self):
        print(f"{self.voxel[:, -1].shape=}")
        image_ndarray = np.einsum("nm,m->n", self.mat_H, self.voxel[:, -1]).reshape(self.image_size)

        max_value = np.max(image_ndarray)
        image = Image.fromarray((image_ndarray / max_value * 255 if max_value else image_ndarray).astype(np.uint8))
        image.save("test.png")


if __name__ == '__main__':
    o = OpticalSystem()
    o.voxel[800:1000, -1] = 1
    # o.voxel[852, -1] = 5
    # o.voxel[857, -1] = 5
    # o.voxel[862, -1] = 5
    # o.voxel[867, -1] = 5

    o.calculate()
    o.voxel2img()

    # image_ = o.mk_image_data(o.voxel[855, :]).reshape(o.image_size) + o.mk_image_data(o.voxel[857, :]).reshape(
    #     o.image_size)
    # image_ = Image.fromarray((image_ * 255 / image_.max()).astype(np.uint8))
    # image_.save("855.png")
    # image_.show()


