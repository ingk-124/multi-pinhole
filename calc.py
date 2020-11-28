from conditional import *
import seaborn as sns


class NotExistPath(Exception):
    pass


class Calculation:

    def __init__(self, blur_mat=None, trans_mat=None, fb_path=None):
        self.fb_path = Path(f"./fb_mode/{fb_path}/")
        self.mode_list = []

        if Path(f"./npz/{blur_mat}").exists() and Path(f"./npz/{trans_mat}").exists() and self.fb_path.exists():
            self.A = sparse.load_npz(f"./npz/{blur_mat}/blur_mat.npz") * sparse.load_npz(
                f"./npz/{trans_mat}/trans_mat_org.npz")
            self.mode_list = np.load(self.fb_path / "mode_array.npy", allow_pickle=True).tolist()
        else:
            raise NotExistPath("I can't find these path...")

        self.mode_dict = {}

        self.fb_matrix = None

    def fb_mode(self, n):
        try:
            mode = self.mode_dict[n]
        except KeyError:
            mode = sparse.load_npz(self.fb_path / f"{tuple(self.mode_list[n])}.npz").T
            self.mode_dict[n] = mode
        return mode

    def fb_img(self, n):
        return self.A * self.fb_mode(n)

    def cross_sections(self, n_l, x_=166, z_=166, xlim1=(0, 758), ylim1=(-250, 250), xlim2=(0, 758), ylim2=(-250, 250)):
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

        I = len(n_l)

        fig, axes = plt.subplots(3, I, figsize=(5 * I, 10))
        print(axes.shape)

        for i, j in enumerate(j_l):
            no = n_l[i]
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
                       extent=[0, 758, -250, 250], aspect='equal', cmap=c)
            ax1.set_title(f"No.{no}: Cross section[Poloidal]", fontsize=16)

            ax2.set_xlim(*xlim2)
            ax2.set_ylim(*ylim2)
            ax2.set_xlabel("y[mm]", fontsize=12)
            ax2.set_ylabel("x[mm]", fontsize=12)
            ax2.plot(y1, x1, 'y')
            ax2.plot(y2, x2, 'y')
            ax2.imshow(j.reshape(-1, 333).tocsr()[:, z_].reshape(333, 511).toarray(),
                       extent=[0, 758, -250, 250], aspect='equal', cmap=c)
            ax2.set_title(f"No.{no}: Cross section[Toroidal]", fontsize=16)

            im = (self.A * j).reshape(128, 128).toarray()
            ax3.set_title(f"No.{no}: Image Simulation", fontsize=16)
            ax3.set_aspect("equal")
            sns.heatmap(im, xticklabels=False, yticklabels=False, ax=ax3)

        fig.suptitle("Fourier-Bessel Cross sections", fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def cross_section_j(self, J=None, n=0, x_=166, z_=166, xlim1=(0, 758), ylim1=(-250, 250), xlim2=(0, 758),
                        ylim2=(-250, 250)):
        if J is None:
            J = self.fb_mode(n)
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

        fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(5, 10))

        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax3.set_aspect("equal")

        ax1.set_xlim(*xlim1)
        ax1.set_ylim(*ylim1)
        ax1.set_xlabel("y[mm]", fontsize=14)
        ax1.set_ylabel("z[mm]", fontsize=14)
        ax1.plot(y, z, 'y')
        ax1.imshow(J.reshape(333, -1).tocsr()[x_].reshape(511, 333).toarray().T,
                   extent=[0, 758, -250, 250], aspect='equal', cmap=c)
        ax1.set_title(f"Cross section[Poloidal]", fontsize=16)

        ax2.set_xlim(*xlim2)
        ax2.set_ylim(*ylim2)
        ax2.set_xlabel("y[mm]", fontsize=12)
        ax2.set_ylabel("x[mm]", fontsize=12)
        ax2.plot(y1, x1, 'y')
        ax2.plot(y2, x2, 'y')
        ax2.imshow(J.reshape(-1, 333).tocsr()[:, z_].reshape(333, 511).toarray(),
                   extent=[0, 758, -250, 250], aspect='equal', cmap=c)
        ax2.set_title(f"Cross section[Toroidal]", fontsize=16)

        im = (self.A * J).reshape(128, 128).toarray()
        ax3.set_title(f"Image Simulation", fontsize=16)
        ax3.set_aspect("equal")
        sns.heatmap(im, xticklabels=False, yticklabels=False, ax=ax3)

        fig.suptitle("Fourier-Bessel Cross sections", fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def cross_xy(self, J=None, n=0, z_=166, c="coolwarm"):
        if J is None:
            J = self.fb_mode(n)

        x1, y1, x2, y2 = [], [], [], []
        for p in np.linspace(-np.pi, np.pi, 100):
            y1.append(258 * np.cos(p))
            x1.append(258 * np.sin(p))
            y2.append(750 * np.cos(p))
            x2.append(750 * np.sin(p))

        fig, ax = plt.subplots()
        ax.plot(y1, x1, 'y')
        ax.plot(y2, x2, 'y')

        ax.set_ylim(-250, 250)
        ax.set_xlim(-10, 800)

        ax.imshow(J.toarray().reshape(333, 511, 333)[:, :, z_],
                  extent=[0, 765, -249, 249], aspect='equal', cmap=c)

        return fig

    def cross_yz(self, J=None, n=0, x_=142, c="coolwarm"):
        if J is None:
            J = self.fb_mode(n)

        x, y = [], []
        for t in np.linspace(0, -2 * np.pi, 360):
            x.append(508 + 250 * np.cos(t))
            y.append(250 * np.sin(t))

        fig, ax = plt.subplots()
        ax.plot(x, y, 'y')
        ax.set_xlim(-10, 800)
        ax.set_ylim(-250, 250)

        ax.imshow(J.toarray().reshape(333, 511, 333)[x_].T,
                  extent=[0, 765, -249, 249], aspect='equal', cmap=c)

        return fig

    def cross_zx(self, J=None, n=0, y_=0, c="coolwarm"):
        if J is None:
            J = self.fb_mode(n)

        fig, ax = plt.subplots()

        ax.imshow(J.toarray().reshape(333, 511, 333)[:, y_, :].T,
                  aspect='auto', cmap=c)

        ax.set_aspect("equal")

        return fig

    def mk_fb_matrix(self, save=False):
        if Path(self.fb_path/"fb_matrix.npz").exists():
            self.fb_matrix = sparse.load_npz(self.fb_path/"fb_matrix.npz")
        else:
            load = Parallel(n_jobs=-1, verbose=10)(
                [delayed(self.fb_img)(n) for n in range(len(self.mode_list))])
            self.fb_matrix = sparse.hstack(load)
            if save:
                sparse.save_npz(self.fb_path/"fb_matrix.npz",self.fb_matrix)


def option():
    argparser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    argparser.add_argument('-b', '--blur', type=str,
                           default=None, help='Blur mat directory. (default=None)')
    argparser.add_argument('-t', '--trans', type=str,
                           default=None, help='Trans mat directory. (default=None)')
    argparser.add_argument('-f', '--fb', type=str,
                           default=None, help='FB directory. (default=None)')
    return argparser.parse_args()


if __name__ == '__main__':
    opt = option()
    cal = Calculation(blur_mat=opt.blur, trans_mat=opt.trans, fb_path=opt.fb)
    cal.mk_fb_matrix(save=True)

