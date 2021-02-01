from shooting import *
import seaborn as sns
from scipy import linalg as LA
from scipy.sparse import linalg as sLA

sns.set(context="notebook", style='ticks', rc={'xtick.direction': 'in', 'ytick.direction': 'in', })


class NotExistPath(Exception):
    # hoge
    pass


def load_file(file):
    with open(file) as f:
        json_dic = json.load(f)
    return json_dic


class Calculation:

    def __init__(self, config_file):

        kwargs = load_file(config_file)

        sim_name = kwargs.get("sim_name")
        shape = kwargs.get("coarse_shape")
        x_range = kwargs.get("x_range")
        y_range = kwargs.get("y_range")
        z_range = kwargs.get("z_range")
        image_size = kwargs.get("image_size")

        self.path = Path(f"./data/{sim_name}/")
        if not self.path.exists():
            raise NotExistPath("I can't find the path :(")

        self.shape = (100, 150, 100) if shape is None else shape
        self.x_range = np.array([-249, 249] if x_range is None else x_range)
        self.y_range = np.array([0, 765] if y_range is None else y_range)
        self.z_range = np.array([-249, 249] if z_range is None else z_range)
        self.image_size = (128, 128) if image_size is None else image_size

        self.mode_list = []
        self.P = None

        dir_list = list(self.path.glob("fb_mode*"))
        print(*dir_list, sep="\n")
        no = input("Input fb_mode directory No.") if len(dir_list) > 1 else ""

        self.fb_dir = self.path / ("fb_mode" + f"({no})") if no else self.path / "fb_mode"
        self.mode_list = np.load(self.path / "mode_array.npy", allow_pickle=True)
        self.mode_dict = {}

        self.A_0 = None
        self.F = None
        self.view_area = None
        self.W = None
        self.A = None
        self.rank = None

        self.J = np.prod(self.shape)
        self.M = len(self.mode_list)
        self.N = np.prod(self.image_size)

        self.x_lim, self.y_lim, self.z_lim = [(a - a.sum() / 2) * 1.02 + a.sum() / 2 for a in
                                              [self.x_range, self.y_range, self.z_range]]

        self.opti = OpticalSystem(**kwargs, read_only=True)
        self.opti.coarse_object.set_rtp()
        self.inside = np.where(self.opti.coarse_object.r < 1, 1, 0)

        self.R = None
        self.L_x = None
        self.L_y = None
        self.L_z = None

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

    def fb_img(self, n, add_dict=True):
        return self.P * self.fb_mode(n, add_dict)

    def cross_sections(self, j_l, x=None, z=None, figsize=(5, 10), space=0.05, titles=None, titlesize=23,
                       xlim1=None, ylim1=None, xlim2=None, ylim2=None, show_im=True):
        x = self.shape[0] // 2 + 1 if x is None else x
        z = self.shape[2] // 2 + 1 if z is None else z

        if xlim1 is None:
            xlim1 = self.y_lim
        if ylim1 is None:
            ylim1 = self.z_lim
        if xlim2 is None:
            xlim2 = self.y_lim
        if ylim2 is None:
            ylim2 = self.x_lim[::-1]

        c = "coolwarm"
        j_l = np.array(j_l, ndmin=2)
        titles = np.array(titles, ndim=2)
        print_title = (j_l.shape == titles.shape)

        x1, y1, x2, y2, y_, z_ = [], [], [], [], [], []
        for theta in np.linspace(0, -2 * np.pi, 360):
            y_.append(508 + 250 * np.cos(theta))
            z_.append(250 * np.sin(theta))
        for phi in np.linspace(-np.pi / 2, np.pi / 2, 100):
            y1.append(258 * np.cos(phi))
            x1.append(258 * np.sin(phi))
            y2.append(758 * np.cos(phi))
            x2.append(758 * np.sin(phi))

        rows, cols = j_l.shape

        fig = plt.figure(constrained_layout=False, figsize=(figsize[0] * cols, figsize[1] * rows))
        gs = fig.add_gridspec(rows, cols, wspace=0.3, hspace=space)

        widths = [9, 1]
        heights = [1, 1, 1] if show_im else [1, 1]
        n = 3 if show_im else 2
        for row in range(rows):
            for col in range(cols):
                j = j_l[row, col].toarray() if isinstance(j_l[row, col], sparse.spmatrix) else j_l[row, col]
                gs_n = gs[row, col].subgridspec(n, 2, width_ratios=widths, height_ratios=heights, hspace=0.4)
                ax1 = fig.add_subplot(gs_n[0, :], xlim=xlim1, ylim=ylim1, aspect="equal")
                ax2 = fig.add_subplot(gs_n[1, :], xlim=xlim2, ylim=ylim2, aspect="equal")
                if print_title:
                    ax1.set_title(titles[row, col] + " Poloidal", fontsize=titlesize)
                    ax2.set_title(titles[row, col] + " Toroidal", fontsize=titlesize)
                else:
                    ax1.set_title("Poloidal", fontsize=titlesize)
                    ax2.set_title("Toroidal", fontsize=titlesize)

                max_j = abs(j.max()) if abs(j.max()) > abs(j.min()) else abs(j.min())

                ax1.set_xlabel("y[mm]", fontsize=14)
                ax1.set_ylabel("z[mm]", fontsize=14)
                ax1.plot(y_, z_, 'y')
                extent1 = [*self.y_range, *self.z_range[::-1]]  # [0, 765, 249, -249]
                ax1.imshow(j.reshape(*self.shape)[x, :, :].T,
                           extent=extent1, cmap=c, vmin=-max_j, vmax=max_j)

                ax2.set_xlabel("y[mm]", fontsize=14)
                ax2.set_ylabel("x[mm]", fontsize=14)
                ax2.plot(y1, x1, 'y')
                ax2.plot(y2, x2, 'y')
                extent2 = [*self.y_range, *self.x_range[::-1]]  # [0, 765, 249, -249]
                ax2.imshow(j.reshape(*self.shape)[:, :, z],
                           extent=extent2, cmap=c, vmin=-max_j, vmax=max_j)

                if show_im:
                    ax3 = fig.add_subplot(gs_n[2, 0])
                    cbar_ax = fig.add_subplot(gs_n[2, 1])

                    im = (self.P * j).reshape(*self.image_size)
                    max_v = abs(im.max()) if abs(im.max()) > abs(im.min()) else abs(im.min())
                    sns.heatmap(im, xticklabels=False, yticklabels=False, square=True,
                                ax=ax3, cmap="RdBu_r", cbar_ax=cbar_ax, vmin=-max_v, vmax=max_v)
                    cbar_ax.tick_params(axis='y', labelsize=10)

    def cross_sections_n(self, n_l, show_im=True, titlesize=23):
        n_l = np.array(n_l, ndmin=2)
        pprint([self.mode_list[_] for _ in n_l.ravel()])
        j_l = np.frompyfunc(lambda n: self.fb_mode(n, load_only=False, add_dict=True), 1, 1)(n_l)
        print("loaded")
        titles = [[f"No.{i}" for i in v] for v in n_l]
        self.cross_sections(j_l=j_l.tolist(), show_im=show_im, titles=titles, titlesize=titlesize)

    def cross_xy(self, j=None, n=0, z=None, c="coolwarm"):
        z = self.shape[-1] // 2 + 1 if z is None else z
        if j is None:
            j = self.fb_mode(n)

        x1, y1, x2, y2 = [], [], [], []
        for p in np.linspace(-np.pi, np.pi, 100):
            y1.append(258 * np.cos(p))
            x1.append(258 * np.sin(p))
            y2.append(758 * np.cos(p))
            x2.append(758 * np.sin(p))

        max_j = abs(j.max()) if abs(j.max()) > abs(j.min()) else abs(j.min())
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=self.y_lim, ylim=self.x_lim, aspect='equal')
        ax.plot(y1, x1, 'y')
        ax.plot(y2, x2, 'y')

        extent = [*self.y_range, *self.x_range[::-1]]  # [0, 765, 249, -249]
        ax.imshow(j.toarray().reshape(*self.shape)[:, :, z],
                  extent=extent, cmap=c, vmin=-max_j, vmax=max_j)

        return fig

    def cross_yz(self, j=None, n=0, x=None, c="coolwarm"):
        x = self.shape[0] // 2 + 1 if x is None else x
        if j is None:
            j = self.fb_mode(n)

        x_, y_ = [], []
        for t in np.linspace(0, -2 * np.pi, 360):
            x_.append(508 + 250 * np.cos(t))
            y_.append(250 * np.sin(t))

        max_j = abs(j.max()) if abs(j.max()) > abs(j.min()) else abs(j.min())
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=self.y_lim, ylim=self.z_lim, aspect='equal')
        ax.plot(x_, y_, 'y')
        extent = [*self.y_range, *self.z_range[::-1]]  # [0, 765, 249, -249]
        ax.imshow(j.toarray().reshape(*self.shape)[x, :, :].T,
                  extent=extent, cmap=c, vmin=-max_j, vmax=max_j)

        return fig

    def cross_zx(self, j=None, n=0, y=0, c="coolwarm"):
        if j is None:
            j = self.fb_mode(n)

        max_j = abs(j.max()) if abs(j.max()) > abs(j.min()) else abs(j.min())
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=self.x_lim[::-1], ylim=self.z_lim, aspect='equal')
        extent = [*self.x_range, *self.z_range]  # [-249, 249, 249, -249]
        ax.imshow(j.toarray().reshape(*self.shape)[:, y, :].T,
                  extent=extent, cmap=c, vmin=-max_j, vmax=max_j)

        return fig

    def mk_P_matrix(self):
        if (self.path / "P_matrix.npz").exists():
            self.P = sparse.load_npz(self.path / "P_matrix.npz")
            print("P_matrix is OK.")
        else:
            if (self.path / "blur_mat.npz").exists() and (self.path / "trans_mat_org.npz").exists():
                self.P = sparse.load_npz(self.path / "blur_mat.npz") * sparse.load_npz(self.path / "trans_mat_org.npz")
                sparse.save_npz(self.path / "P_matrix.npz", self.P)
                print("P_matrix.npz: saved!")
            else:
                print("Please set both blur_mat.npz and trans_mat_org.npz at the directory.")
                quit()
        self.view_area = np.any(self.P.astype(bool), axis=0)

    def mk_A_0_matrix(self):
        if Path(self.path / "A_0_matrix.npz").exists():
            self.A_0 = sparse.load_npz(self.path / "A_0_matrix.npz")
            print("A_0_matrix is OK.")
        else:
            self.mk_P_matrix()
            num = int(input(f"multi-process num(max={multi.cpu_count()}): "))
            load_img = Parallel(n_jobs=num, verbose=10)([delayed(self.fb_img)(n) for n in range(self.M)])
            self.A_0 = sparse.hstack(load_img)
            sparse.save_npz(self.path / "A_0_matrix.npz", self.A_0)
            print("A_0_matrix.npz: saved!")

    def mk_F_matrix(self):
        if Path(self.path / "F_matrix.npz").exists():
            self.F = sparse.load_npz(self.path / "F_matrix.npz")
            print("F_matrix is OK.")
        else:
            num = int(input(f"multi-process num(max={multi.cpu_count()}): "))
            load_f = Parallel(n_jobs=num, verbose=10)([delayed(self.fb_mode)(n) for n in range(self.M)])
            self.F = sparse.hstack(load_f)
            sparse.save_npz(self.path / "F_matrix.npz", self.F)
            print("F_matrix.npz: saved!")

    def mk_L_matrix(self):

        one = np.ones(self.shape)
        one[[0, -1], ...] = 0
        one[:, [0, -1], :] = 0
        one[..., [0, -1]] = 0

        self.R = sparse.diags(one.ravel() * self.inside)

        d = np.prod(self.shape)
        self.L_x = self.R * sparse.diags([1, -2, 1], [-self.shape[1] * self.shape[2], 0, self.shape[1] * self.shape[2]],
                                         shape=(d, d))
        self.L_y = self.R * sparse.diags([1, -2, 1], [-self.shape[2], 0, self.shape[2]], shape=(d, d))
        self.L_z = self.R * sparse.diags([1, -2, 1], [-1, 0, 1], shape=(d, d))

    def mk_A(self):
        if self.A_0 is None:
            self.mk_A_0_matrix()
        self.rank = np.linalg.matrix_rank(self.A_0.toarray())
        self.W = sLA.norm(self.A_0, axis=0)
        self.A = self.A_0 * sparse.diags(1 / self.W)

    def load_all(self):
        self.mk_P_matrix()
        self.mk_A_0_matrix()
        self.mk_F_matrix()
        self.mk_L_matrix()
        self.mk_A()


def option():
    # noinspection PyTypeChecker
    argparser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    argparser.add_argument('-s', '--sim_name', type=str,
                           default=None, help='simulation name. (default=None)')
    argparser.add_argument('-f', '--file', type=str,
                           default=None, help='Configuration file. (default=None)')
    return argparser.parse_args()


if __name__ == '__main__':
    opt = option()
    cl = Calculation(opt.file)

    while True:
        case = input("A_0_matrix(A0)/F_matrix(F)/Quit(q): ")
        if case == "q":
            quit()
        elif case == "A0":
            cl.mk_A_0_matrix()
        elif case == "F":
            cl.mk_F_matrix()
