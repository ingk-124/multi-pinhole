from shooting import *
import seaborn as sns
from scipy import linalg as LA

sns.set(context="notebook", style='ticks', rc={'xtick.direction': 'in', 'ytick.direction': 'in', })


class NotExistPath(Exception):
    # hoge
    pass


class Calculation:

    def __init__(self, sim_name=None):
        self.path = Path(f"./data/{sim_name}/")
        if not self.path.exists():
            raise NotExistPath("I can't find the path :(")

        self.mode_list = []
        self.P = None
        self.shape = (333, 511, 333)

        dir_list = list(self.path.glob("fb_mode*"))
        print(*dir_list, sep="\n")
        no = input("Input fb_mode directory No.") if len(dir_list) > 1 else ""
        self.fb_dir = self.path / ("fb_mode" + f"({no})") if no else self.path / "fb_mode"

        if (self.path / "blur_mat.npz").exists() and (self.path / "trans_mat_org.npz").exists():
            if (self.path / "mat_P.npz").exists():
                self.P = sparse.load_npz(self.path / "mat_P.npz")
            else:
                self.P = sparse.load_npz(self.path / "blur_mat.npz") * sparse.load_npz(self.path / "trans_mat_org.npz")
                sparse.save_npz(self.path / "mat_P.npz", self.P)
            self.mode_list = np.load(self.path / "mode_array.npy", allow_pickle=True)
        else:
            print("Please set both blur_mat.npz and trans_mat_org.npz at the directory.")
            quit()
        print("mat_P is OK.")
        self.mode_dict = {}

        self.fb_matrix = None
        self.F_mat = None
        self.G_x = None
        self.G_y = None
        self.G_z = None
        self.effective = None
        self.A_0 = None
        self.w = None
        self.A = None
        self.shape = (333, 511, 333)
        self.N = np.prod(self.shape)
        self.M = len(self.mode_list)
        self.x_range = [-249, 249]
        self.y_range = [0, 768]
        self.z_range = [-249, 249]

        self.x_lim = [-250, 250]
        self.y_lim = [-10, 800]
        self.z_lim = [-250, 250]

        one = np.ones(self.shape)
        one[[0, -1], ...] = 0
        one[:, [0, -1], :] = 0
        one[..., [0, -1]] = 0

        self.R = sparse.diags(one.ravel())

    def fb_mode(self, n, load_only=True, add_dict=False):
        if load_only:
            mode = sparse.load_npz(self.fb_dir / f"mode_No{n}.npz").T
            return mode
        else:
            try:
                mode = self.mode_dict[n]
            except KeyError:
                mode = sparse.load_npz(self.fb_dir / f"mode_No{n}.npz").T
                if add_dict:
                    self.mode_dict[n] = mode
            return mode

    def small_j(self, n, r=5):
        return sparse.csr_matrix(self.fb_mode(n).toarray().reshape(*self.shape)[::r, ::r, ::r].ravel()).T

    def fb_img(self, n):
        return self.P * self.fb_mode(n)

    def cross_sections(self, d_l, x=166, z=166, figsize=(5, 10), space=0.05,
                       xlim1=(0, 758), ylim1=(-250, 250), xlim2=(0, 758), ylim2=(250, -250)):
        c = "coolwarm"
        d_l = np.array(d_l, ndmin=2)

        x1, y1, x2, y2, y_, z_ = [], [], [], [], [], []
        for theta in np.linspace(0, -2 * np.pi, 360):
            y_.append(508 + 250 * np.cos(theta))
            z_.append(250 * np.sin(theta))
        for phi in np.linspace(-np.pi / 2, np.pi / 2, 100):
            y1.append(258 * np.cos(phi))
            x1.append(258 * np.sin(phi))
            y2.append(758 * np.cos(phi))
            x2.append(758 * np.sin(phi))

        rows, cols = d_l.shape

        fig = plt.figure(constrained_layout=False, figsize=(figsize[0] * cols, figsize[1] * rows))
        gs = fig.add_gridspec(rows, cols, wspace=0.3, hspace=space)

        widths = [9, 1]
        heights = [1, 1, 1]
        for row in range(rows):
            for col in range(cols):
                d = d_l[row, col]
                gs_n = gs[row, col].subgridspec(3, 2, width_ratios=widths, height_ratios=heights, hspace=0.4)
                ax1 = fig.add_subplot(gs_n[0, :])
                ax2 = fig.add_subplot(gs_n[1, :])
                ax3 = fig.add_subplot(gs_n[2, 0])
                cbar_ax = fig.add_subplot(gs_n[2, 1])

                max_j = abs(d.max()) if abs(d.max()) > abs(d.min()) else abs(d.min())
                ax1.set_aspect("equal")
                ax2.set_aspect("equal")

                ax1.set_xlim(*xlim1)
                ax1.set_ylim(*ylim1)
                ax1.set_xlabel("y[mm]", fontsize=14)
                ax1.set_ylabel("z[mm]", fontsize=14)
                ax1.plot(y_, z_, 'y')
                extent1 = [*self.y_range, *self.z_range[::-1]]  # [0, 765, 249, -249]
                ax1.imshow(d.toarray().reshape(*self.shape)[x, :, :].T,
                           extent=extent1, aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)

                ax2.set_xlim(*xlim2)
                ax2.set_ylim(*ylim2)
                ax2.set_xlabel("y[mm]", fontsize=14)
                ax2.set_ylabel("x[mm]", fontsize=14)
                ax2.plot(y1, x1, 'y')
                ax2.plot(y2, x2, 'y')
                extent2 = [*self.y_range, *self.x_range[::-1]]  # [0, 765, 249, -249]
                ax2.imshow(d.toarray().reshape(*self.shape)[:, :, z],
                           extent=extent2, aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)

                im = (self.P * d).reshape(128, 128).toarray()
                max_v = abs(im.max()) if abs(im.max()) > abs(im.min()) else abs(im.min())
                sns.heatmap(im, xticklabels=False, yticklabels=False, square=True,
                            ax=ax3, cmap="RdBu_r", cbar_ax=cbar_ax, vmin=-max_v, vmax=max_v)
                cbar_ax.tick_params(axis='y', labelsize=10)

    def cross_sections_n(self, n_l):
        n_l = np.array(n_l, ndmin=2)
        pprint([self.mode_list[_] for _ in n_l.ravel()])
        d_l = np.frompyfunc(lambda n: self.fb_mode(n, load_only=False, add_dict=True), 1, 1)(n_l)
        print("loaded")
        self.cross_sections(d_l=d_l.tolist())

    def cross_xy(self, d=None, n=0, z=166, c="coolwarm"):
        if d is None:
            d = self.fb_mode(n)

        x1, y1, x2, y2 = [], [], [], []
        for p in np.linspace(-np.pi, np.pi, 100):
            y1.append(258 * np.cos(p))
            x1.append(258 * np.sin(p))
            y2.append(758 * np.cos(p))
            x2.append(758 * np.sin(p))

        max_j = abs(d.max()) if abs(d.max()) > abs(d.min()) else abs(d.min())
        fig, ax = plt.subplots()
        ax.plot(y1, x1, 'y')
        ax.plot(y2, x2, 'y')

        ax.set_ylim(*self.x_lim[::-1])  # 250, -250
        ax.set_xlim(*self.y_lim)  # -10, 1000
        extent = [*self.y_range, *self.x_range[::-1]]  # [0, 765, 249, -249]
        ax.imshow(d.toarray().reshape(*self.shape)[:, :, z],
                  extent=extent, aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)

        return fig

    def cross_yz(self, d=None, n=0, x=142, c="coolwarm"):
        if d is None:
            d = self.fb_mode(n)

        x_, y_ = [], []
        for t in np.linspace(0, -2 * np.pi, 360):
            x_.append(508 + 250 * np.cos(t))
            y_.append(250 * np.sin(t))

        max_j = abs(d.max()) if abs(d.max()) > abs(d.min()) else abs(d.min())
        fig, ax = plt.subplots()
        ax.plot(x_, y_, 'y')
        ax.set_xlim(*self.y_lim)  # -10, 1000
        ax.set_ylim(*self.z_lim)  # -250, 250
        extent = [*self.y_range, *self.z_range[::-1]]  # [0, 765, 249, -249]
        ax.imshow(d.toarray().reshape(*self.shape)[x, :, :].T,
                  extent=extent, aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)

        return fig

    def cross_zx(self, d=None, n=0, y=0, c="coolwarm"):
        if d is None:
            d = self.fb_mode(n)

        max_j = abs(d.max()) if abs(d.max()) > abs(d.min()) else abs(d.min())
        fig, ax = plt.subplots()

        ax.set_xlim(*self.x_lim[::-1])  # 250, -250
        ax.set_ylim(*self.z_lim)  # -250, 250
        extent = [*self.x_range, *self.z_range]  # [-249, 249, 249, -249]
        ax.imshow(d.toarray().reshape(*self.shape)[:, y, :].T,
                  extent=extent, aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)

        ax.set_aspect("equal")

        return fig

    def mk_fb_matrix(self):

        if Path(self.path / "fb_matrix.npz").exists():
            self.fb_matrix = sparse.load_npz(self.path / "fb_matrix.npz")
            print("fb_matrix is OK.")
        else:
            num = int(input(f"multi-process num(max={multi.cpu_count()}): "))

            load_img = Parallel(n_jobs=num, verbose=10)([delayed(self.fb_img)(n) for n in range(self.M)])
            self.fb_matrix = sparse.hstack(load_img)

            sparse.save_npz(self.path / "fb_matrix.npz", self.fb_matrix)
            print("fb_matrix.npz: saved!")

    def mk_F_matrix(self):
        if Path(self.path / "F_mat.npz").exists():
            self.F_mat = sparse.load_npz(self.path / "F_mat.npz")
            print("F_mat is OK.")
        else:
            num = int(input(f"multi-process num(max={multi.cpu_count()}): "))

            load_f = Parallel(n_jobs=num, verbose=10)([delayed(self.small_j)(n) for n in range(self.M)])
            self.F_mat = sparse.hstack(load_f)

            sparse.save_npz(self.path / "F_mat.npz", self.fb_matrix)
            print("F_mat.npz: saved!")

    def mk_A(self):
        self.mk_fb_matrix()
        self.effective = np.array(self.fb_matrix.sum(axis=1)).astype(bool).T[0]
        self.A_0 = self.fb_matrix.toarray()[self.effective]
        self.w = LA.norm(self.A_0, axis=0) / np.sqrt(self.A_0.shape[0])
        self.A = self.A_0 / self.w

    def div_x(self, n):
        L_x = sparse.diags([1, -2, 1], [-self.shape[1] * self.shape[2], 0, self.shape[1] * self.shape[2]],
                           shape=(self.N, self.N))
        return self.R * L_x * self.fb_mode(n)

    def div_y(self, n):
        L_y = sparse.diags([1, -2, 1], [-self.shape[2], 0, self.shape[2]], shape=(self.N, self.N))
        return self.R * L_y * self.fb_mode(n)

    def div_z(self, n):
        L_z = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(self.N, self.N))
        return self.R * L_z * self.fb_mode(n)

    def mk_G_x(self, num):
        if Path(self.path / "G_x.npz").exists():
            self.G_x = sparse.load_npz(self.path / "G_x.npz")
            print("G_x is OK.")
        else:
            load = Parallel(n_jobs=num, verbose=10)([delayed(self.div_x)(n) for n in range(self.M)])
            self.G_x = sparse.hstack(load)
            sparse.save_npz(self.path / "G_x.npz", self.G_x)
            print("G_x.npz: saved!")

    def mk_G_y(self, num):
        if Path(self.path / "G_y.npz").exists():
            self.G_y = sparse.load_npz(self.path / "G_y.npz")
            print("G_y is OK.")
        else:
            load = Parallel(n_jobs=num, verbose=10)([delayed(self.div_x)(n) for n in range(self.M)])
            self.G_y = sparse.hstack(load)
            sparse.save_npz(self.path / "G_y.npz", self.G_y)
            print("G_y.npz: saved!")

    def mk_G_z(self, num):
        if Path(self.path / "G_z.npz").exists():
            self.G_z = sparse.load_npz(self.path / "G_z.npz")
            print("G_z is OK.")
        else:
            load = Parallel(n_jobs=num, verbose=10)([delayed(self.div_x)(n) for n in range(self.M)])
            self.G_z = sparse.hstack(load)
            sparse.save_npz(self.path / "G_z.npz", self.G_z)
            print("G_z.npz: saved!")

    def mk_G(self):
        num = int(input(f"multi-process num(max={multi.cpu_count()}): "))
        print("G_x")
        self.mk_G_x(num=num)
        print("G_y")
        self.mk_G_y(num=num)
        print("G_z")
        self.mk_G_z(num=num)


def option():
    # noinspection PyTypeChecker
    argparser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    argparser.add_argument('-s', '--sim_name', type=str,
                           default=None, help='simulation name. (default=None)')
    return argparser.parse_args()


if __name__ == '__main__':
    opt = option()
    dir_name = opt.sim_name if opt.sim_name else "Test_test"
    cl = Calculation(sim_name=dir_name)
    while True:
        case = input("fb_matrix(fb)/F_matrix(F)/Quit(q): ")
        if case == "q":
            quit()
        elif case == "fb":
            cl.mk_fb_matrix()
        elif case == "F":
            cl.mk_F_matrix()
