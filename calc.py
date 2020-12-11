from shooting import *
import seaborn as sns


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

        dir_list = list(self.path.glob("fb_mode*"))
        print(*dir_list, sep="\n")
        no = input("Input fb_mode directory No.") if len(dir_list) > 1 else ""
        self.fb_dir = self.path / ("fb_mode" + f"({no})") if no else self.path / "fb_mode"

        if (self.path / "mat_P.npz").exists():
            self.P = sparse.load_npz(self.path / "mat_P.npz")

        if (self.path / "blur_mat.npz").exists() and (self.path / "trans_mat_org.npz").exists():
            self.P = sparse.load_npz(self.path / "blur_mat.npz") * sparse.load_npz(self.path / "trans_mat_org.npz")
            sparse.save_npz(self.path / "mat_P.npz", self.P)
            self.mode_list = np.load(self.path / "mode_array.npy", allow_pickle=True)
        else:
            print("Please set both blur_mat.npz and trans_mat_org.npz at the directory.")
            quit()
        print("mat_P is OK.")
        self.mode_dict = {}

        self.fb_matrix = None

    def fb_mode(self, n, add_dict=False):
        try:
            mode = self.mode_dict[n]
        except KeyError:
            mode = sparse.load_npz(self.fb_dir / f"mode_No{n}.npz").T
            if add_dict:
                self.mode_dict[n] = mode
        return mode

    def fb_img(self, n):
        return self.P * self.fb_mode(n)

    def cross_sections(self, n_l, x_=166, z_=166, xlim1=(0, 758), ylim1=(-250, 250), xlim2=(0, 758), ylim2=(-250, 250),
                       figsize=(5, 10), suptitle_size=40, title_size=25, suptitle="Cross sections",
                       pol_title="Poloidal", tor_title="Toroidal", image_title="Image"):
        c = "coolwarm"
        if isinstance(n_l, int):
            n_l = [n_l]
        elif isinstance(n_l, list):
            pass
        else:
            return 1

        pprint([self.mode_list[_] for _ in n_l])
        j_l = [self.fb_mode(n) for n in n_l]

        x1, y1, x2, y2, y, z = [], [], [], [], [], []
        for theta in np.linspace(0, -2 * np.pi, 360):
            y.append(508 + 250 * np.cos(theta))
            z.append(250 * np.sin(theta))
        for phi in np.linspace(-np.pi / 2, np.pi / 2, 100):
            y1.append(258 * np.cos(phi))
            x1.append(258 * np.sin(phi))
            y2.append(750 * np.cos(phi))
            x2.append(750 * np.sin(phi))

        w = len(n_l)

        fig, axes = plt.subplots(3, w, figsize=(figsize[0] * w, figsize[1]))
        print(axes.shape)

        for i, j in enumerate(j_l):
            no = n_l[i]
            max_j = abs(j.max()) if abs(j.max()) > abs(j.min()) else abs(j.min())
            ax1, ax2, ax3 = axes[:, i] if len(axes.shape) == 2 else axes

            ax1.set_aspect("equal")
            ax2.set_aspect("equal")
            ax3.set_aspect("equal")

            ax1.set_xlim(*xlim1)
            ax1.set_ylim(*ylim1)
            ax1.set_xlabel("y[mm]", fontsize=14)
            ax1.set_ylabel("z[mm]", fontsize=14)
            ax1.plot(y, z, 'y')
            ax1.imshow(j.reshape(333, -1).tocsr()[x_].reshape(511, 333).toarray().T,
                       extent=[0, 758, -250, 250], aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)
            ax1.set_title(f"No.{no}: {pol_title}", fontsize=title_size)

            ax2.set_xlim(*xlim2)
            ax2.set_ylim(*ylim2)
            ax2.set_xlabel("y[mm]", fontsize=14)
            ax2.set_ylabel("x[mm]", fontsize=14)
            ax2.plot(y1, x1, 'y')
            ax2.plot(y2, x2, 'y')
            ax2.imshow(j.reshape(-1, 333).tocsr()[:, z_].reshape(333, 511).toarray(),
                       extent=[0, 758, -250, 250], aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)
            ax2.set_title(f"No.{no}: {tor_title}", fontsize=title_size)

            im = (self.P * j).reshape(128, 128).toarray()
            max_v = abs(im.max()) if abs(im.max()) > abs(im.min()) else abs(im.min())
            ax3.set_title(f"No.{no}: {image_title}", fontsize=title_size)
            ax3.set_aspect("equal")
            sns.heatmap(im, xticklabels=False, yticklabels=False, ax=ax3, cmap="RdBu_r", vmin=-max_v, vmax=max_v)

        fig.suptitle(f"{suptitle}", fontsize=suptitle_size)
        fig.tight_layout(rect=[0, 0, 1, 0.9])

        return fig

    def cross_section_j(self, j_l=None, n=0, x_=166, z_=166, xlim1=(0, 758), ylim1=(-250, 250), xlim2=(0, 758),
                        ylim2=(-250, 250), figsize=(5, 10), suptitle_size=40, title_size=25, suptitle="Cross sections",
                        pol_title="Poloidal", tor_title="Toroidal", image_title="Image"):
        if j_l is None:
            j_l = self.fb_mode(n)
        elif isinstance(j_l, list):
            pass
        else:
            j_l = [j_l, ]

        w = len(j_l)

        c = "coolwarm"

        x1, y1, x2, y2, y, z = [], [], [], [], [], []
        for theta in np.linspace(0, -2 * np.pi, 360):
            y.append(508 + 250 * np.cos(theta))
            z.append(250 * np.sin(theta))
        for phi in np.linspace(-np.pi / 2, np.pi / 2, 100):
            y1.append(258 * np.cos(phi))
            x1.append(258 * np.sin(phi))
            y2.append(750 * np.cos(phi))
            x2.append(750 * np.sin(phi))

        fig, axes = plt.subplots(3, w, figsize=(figsize[0] * w, figsize[1]))

        for i, j in enumerate(j_l):
            max_j = abs(j.max()) if abs(j.max()) > abs(j.min()) else abs(j.min())
            ax1, ax2, ax3 = axes[:, i] if len(axes.shape) == 2 else axes

            ax1.set_aspect("equal")
            ax2.set_aspect("equal")
            ax3.set_aspect("equal")

            ax1.set_xlim(*xlim1)
            ax1.set_ylim(*ylim1)
            ax1.set_xlabel("y[mm]", fontsize=14)
            ax1.set_ylabel("z[mm]", fontsize=14)
            ax1.plot(y, z, 'y')
            ax1.imshow(j.reshape(333, -1).tocsr()[x_].reshape(511, 333).toarray().T,
                       extent=[0, 758, -250, 250], aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)
            ax1.set_title(f"{pol_title}", fontsize=title_size)

            ax2.set_xlim(*xlim2)
            ax2.set_ylim(*ylim2)
            ax2.set_xlabel("y[mm]", fontsize=14)
            ax2.set_ylabel("x[mm]", fontsize=14)
            ax2.plot(y1, x1, 'y')
            ax2.plot(y2, x2, 'y')
            ax2.imshow(j.reshape(-1, 333).tocsr()[:, z_].reshape(333, 511).toarray(),
                       extent=[0, 758, -250, 250], aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)
            ax2.set_title(f"{tor_title}", fontsize=title_size)

            im = (self.P * j).reshape(128, 128).toarray()
            max_v = abs(im.max()) if abs(im.max()) > abs(im.min()) else abs(im.min())
            ax3.set_title(f"{image_title}", fontsize=title_size)
            ax3.set_aspect("equal")
            sns.heatmap(im, xticklabels=False, yticklabels=False, ax=ax3, cmap="RdBu_r", vmin=-max_v, vmax=max_v)

        fig.suptitle(f"{suptitle}", fontsize=suptitle_size)
        fig.tight_layout(rect=[0, 0, 1, 0.9])

        return fig

    def cross_xy(self, j=None, n=0, z_=166, c="coolwarm"):
        if j is None:
            j = self.fb_mode(n)

        x1, y1, x2, y2 = [], [], [], []
        for p in np.linspace(-np.pi, np.pi, 100):
            y1.append(258 * np.cos(p))
            x1.append(258 * np.sin(p))
            y2.append(750 * np.cos(p))
            x2.append(750 * np.sin(p))

        max_j = abs(j.max()) if abs(j.max()) > abs(j.min()) else abs(j.min())
        fig, ax = plt.subplots()
        ax.plot(y1, x1, 'y')
        ax.plot(y2, x2, 'y')

        ax.set_ylim(-250, 250)
        ax.set_xlim(-10, 800)

        ax.imshow(j.toarray().reshape(333, 511, 333)[:, :, z_],
                  extent=[0, 765, -249, 249], aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)

        return fig

    def cross_yz(self, j=None, n=0, x_=142, c="coolwarm"):
        if j is None:
            j = self.fb_mode(n)

        x, y = [], []
        for t in np.linspace(0, -2 * np.pi, 360):
            x.append(508 + 250 * np.cos(t))
            y.append(250 * np.sin(t))

        max_j = abs(j.max()) if abs(j.max()) > abs(j.min()) else abs(j.min())
        fig, ax = plt.subplots()
        ax.plot(x, y, 'y')
        ax.set_xlim(-10, 800)
        ax.set_ylim(-250, 250)

        ax.imshow(j.toarray().reshape(333, 511, 333)[x_].T,
                  extent=[0, 765, -249, 249], aspect='equal', cmap=c, vmin=-max_j, vmax=max_j)

        return fig

    def cross_zx(self, j=None, n=0, y_=0, c="coolwarm"):
        if j is None:
            j = self.fb_mode(n)

        max_j = abs(j.max()) if abs(j.max()) > abs(j.min()) else abs(j.min())
        fig, ax = plt.subplots()

        ax.imshow(j.toarray().reshape(333, 511, 333)[:, y_, :].T,
                  aspect='auto', cmap=c, vmin=-max_j, vmax=max_j)

        ax.set_aspect("equal")

        return fig

    def mk_fb_matrix(self):

        if Path(self.path / "fb_matrix.npz").exists():
            self.fb_matrix = sparse.load_npz(self.path / "fb_matrix.npz")

        else:
            load_img = Parallel(n_jobs=-1, verbose=10)([delayed(self.fb_img)(n) for n in range(len(self.mode_list))])
            self.fb_matrix = sparse.hstack(load_img)
            sparse.save_npz(self.path / "fb_matrix.npz", self.fb_matrix)


def option():
    argparser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    argparser.add_argument('-s', '--sim_name', type=str,
                           default=None, help='simulation name. (default=None)')
    return argparser.parse_args()


if __name__ == '__main__':
    opt = option()
    sim_name = opt.sim_name if opt.sim_name else "Test_test"
    cal = Calculation(sim_name=sim_name)
    cal.mk_fb_matrix()
