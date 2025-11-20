# import libraries
import gc

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from stl import mesh

from multi_pinhole import *
from utils import stl_utils

plt.rcParams.update({'font.size': 12})


def shifted_torus(r, theta, phi, delta):
    R = r * np.cos(theta) + delta
    Z = r * np.sin(theta)
    r_shifted = np.sqrt(R ** 2 + Z ** 2)
    theta_shifted = np.arctan2(Z, R)
    return r_shifted, theta_shifted, phi


def helical_displacement(r, theta, phi, m_, n_, phi_0, d, r_1, xi_0):
    r_ = r * np.exp(m_ * theta * 1j)
    xi = xi_0 * np.exp(-(r / r_1) ** d) * np.exp(n_ * (phi - phi_0) * 1j)
    r_new_complex = r_ - xi
    r_new = np.abs(r_new_complex)
    theta = np.angle(r_new_complex)
    phi = phi

    return r_new, theta, phi


def hollow(r, A, p, q, h, w):
    f1 = (1 - r ** p) ** q
    f2 = np.exp(-(r / w) ** 2)
    return A * (f1 - h * f2)

def helical_axis(r, theta, phi, m_, n_, r_a, phi_0):
    psi = n_/m_ * phi + phi_0
    dx = r_a * np.cos(psi)
    dy = r_a * np.sin(psi)
    _x = r * np.cos(theta)
    _y = r * np.sin(theta)
    r_new = np.sqrt((_x - dx) ** 2 + (_y - dy) ** 2)
    return r_new


def emission_profile(r, theta, phi, allow_negative=False, **params):
    m_ = params.get("m_", 1)
    n_ = params.get("n_", -1)
    delta = params.get("delta", 0)
    phi_0 = params.get("phi_0", 0)
    d = params.get("d", 2)
    r_1 = params.get("r_1", 0.5)
    xi_0 = params.get("xi_0", 0.1)
    A = params.get("A", 1)
    p = params.get("p", 2)
    q = params.get("q", 3)
    h = params.get("h", 0)
    w = params.get("w", 0.5)

    r_shifted, theta_shifted, phi_shifted = shifted_torus(r, theta, phi, delta)
    r_new, theta_new, phi_new = helical_displacement(r_shifted, theta_shifted, phi_shifted,
                                                     m_=m_, n_=n_, phi_0=phi_0, d=d, r_1=r_1, xi_0=xi_0)
    y = hollow(r_new, A=A, p=p, q=q, h=h, w=w)
    if not allow_negative:
        y = np.maximum(y, 0)
    return y


if __name__ == '__main__':
    file_is_exist = True
    mst_wall = mesh.Mesh.from_file("MST_wall-mesh.stl")
    force_rebuild = True
    # force_rebuild = False
    FILE_NAME = "MST_tangential_double-pinhole_wide2.pkl"
    try:
        if force_rebuild:
            raise FileNotFoundError
        world = World.load_world(filename=FILE_NAME)
        print("World loaded from file.")
    except FileNotFoundError:
        file_is_exist = False
        print("Building world...")
        # Aperture
        max_size = 25
        size = 1.8
        resolution = 40
        # points on xy-plane
        x_arr, y_arr = np.linspace(-max_size, max_size, 5), np.linspace(-max_size, max_size, 5)
        outer_points = np.array(np.meshgrid(x_arr, y_arr, indexing="ij")).reshape(2, -1).T
        t = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        edge_points1 = np.array([size * np.cos(t) - 4.15, size * np.sin(t)]).T
        edge_points2 = np.array([size * np.cos(t) + 4.15, size * np.sin(t)]).T
        points = np.vstack([outer_points, edge_points1, edge_points2])


        def condition(x, y):
            c1 = ((x + 4.15) / size) ** 2 + (y / size) ** 2 <= 1
            c2 = ((x - 4.15) / size) ** 2 + (y / size) ** 2 <= 1
            return c1 | c2


        model_aperture = stl_utils.make_2D_surface(points, condition)

        camera_0 = Camera(eyes=[
            Eye(eye_type="pinhole", eye_shape="circle", eye_size=1, focal_length=25,
                position=[-4.15, 0]),
            Eye(eye_type="pinhole", eye_shape="circle", eye_size=1, focal_length=25,
                position=[4.15, 0])
        ],
            screen=Screen(screen_shape="rectangle", screen_size=[61*0.13, 125*0.13], pixel_shape=(61, 125),
                          subpixel_resolution=5),
            apertures=[Aperture(stl_model=model_aperture, position=[0, 0, 13]),
                       # Aperture(shape="circle", size=1.8, position=[-4.15, 0, 13]).set_model(resolution=40,
                       #                                                                  max_size=50),
                       # Aperture(shape="rectangle", size=(100, 10), position=[0, 0, 80]).set_model(
                       #     resolution=40, max_size=100)
                       ],
            camera_position=[-1522.5, -1500.7, 210.7],
        ).set_rotation_matrix("zxz",
                              (2.9, 98, -19),
                              degrees=True)

        # voxel = Voxel.uniform_voxel(ranges=[[-2005, 5], [-2005, 2005], [-505, 505]],
        #                             # shape=[50, 100, 25],
        #                             shape=[25, 15, 25],
        #                             coordinate_type="torus", coordinate_parameters=dict(a=500, R_0=1500))
        # voxel = Voxel.uniform_voxel(ranges=[[-2010, 10], [-2010, 2010], [-510, 510]],
        #                             shape=[101, 201, 51],
        #                             coordinate_type="torus", coordinate_parameters=dict(a=500, R_0=1500))

        voxel = Voxel.uniform_voxel(ranges=[[-2010, 2010], [-2010, 2010], [-510, 510]],
                                    shape=[201, 201, 51],
                                    coordinate_type="torus", coordinate_parameters=dict(a=500, R_0=1500))

        world = World(voxel=voxel,
                      cameras=[camera_0],
                      walls=mst_wall)


        def inside_condition(x, y, z):
            R = np.sqrt(x ** 2 + y ** 2)
            Z = z
            r = np.sqrt((R - 1500) ** 2 + Z ** 2)
            return r <= 500


        world.set_inside_vertices(inside_condition)
        del camera_0, voxel
        gc.collect()

    world.set_projection_matrix(res=5, verbose=1, parallel=12)
    P = world.P_matrix[0]

    if not file_is_exist or force_rebuild:
        world.save_world(filename=FILE_NAME)
        print("World saved to file.")

    # reset variables
    camera_0 = world.cameras[0]
    voxel = world.voxel

    R_grid = np.sqrt(voxel.gravity_center[:, 0] ** 2 + voxel.gravity_center[:, 1] ** 2)
    phi_grid = np.arctan2(voxel.gravity_center[:, 1], voxel.gravity_center[:, 0])
    Z_grid = voxel.gravity_center[:, 2]
    r_grid = np.sqrt((R_grid - 1500) ** 2 + Z_grid ** 2)
    theta_grid = np.arctan2(Z_grid, R_grid - 1500)
    X, Y, Z = voxel.gravity_center.T

    toroidal_axis_x = 1500 * np.cos(np.linspace(np.pi / 2, 4 * np.pi / 3, 200))
    toroidal_axis_y = 1500 * np.sin(np.linspace(np.pi / 2, 4 * np.pi / 3, 200))
    toroidal_axis_z = np.zeros_like(toroidal_axis_x)
    toroidal_axis = np.vstack([toroidal_axis_x, toroidal_axis_y, toroidal_axis_z]).T

    fig, ax = plt.subplots()
    for i in range(len(camera_0.eyes)):
        toroidal_axis_UV = world.trace_line(toroidal_axis,
                                            camera_idx=0, eye_idx=i, coord_type="UV")
        ax.plot(toroidal_axis_UV[:, 1], toroidal_axis_UV[:, 0],
                label=f"Toroidal Axis Projection eye {i}")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("v [mm]")
    ax.set_ylabel("u [mm]")
    ax.set_xlim(0, camera_0.screen.screen_size[1])
    ax.set_ylim(camera_0.screen.screen_size[0], 0)
    fig.show()

    # fig = go.Figure()
    # camera_0.draw_camera_orientation_plotly(fig, axis_length=50, show_fig=False)
    # stl_utils.plotly_show_stl(mst_wall, fig, color="lightgrey", linearg={'width': 0.5, 'color': 'black'},
    #                           opacity=0.2, show_fig=False)
    # fig.show()

    ax = camera_0.draw_optical_system()
    # ax.set_xlim(-10, 50)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20., azim=60)
    ax.figure.show()

    r, theta, phi = r_grid / 500, theta_grid, phi_grid
    toroidal_axis_UV = [world.trace_line(toroidal_axis,
                                         camera_idx=0, eye_idx=i, coord_type="UV") for i in range(len(camera_0.eyes))]
    vis = world.find_visible_points(toroidal_axis, camera_idx=0, verbose=0)


    def plot_emission_profile_sample(f):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3),
                                 gridspec_kw={'width_ratios': [1, 2], 'wspace': 0.4})
        axes[0].set_title("Emission Profile at Y=0")
        cm = axes[0].contourf(voxel.cx_axis, voxel.cz_axis,
                              f.reshape(voxel.shape)[:, voxel.shape[1] // 2, :].T,
                              levels=15, cmap="viridis")
        axes[0].plot(-1500 + 500 * np.cos(np.linspace(0, 2 * np.pi, 100)),
                     500 * np.sin(np.linspace(0, 2 * np.pi, 100)), "w--", linewidth=2,
                     label="Chamber wall")
        axes[0].set_xlim(-2000, -1000)
        axes[0].set_ylim(*voxel.cz_axis[[0, -1]])
        axes[0].set_xlabel("X [mm]")
        axes[0].set_ylabel("Z [mm]")
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].legend(loc="upper right")
        fig.colorbar(cm, label="Emission Intensity", ax=axes[0], fraction=0.046, pad=0.04)

        im = P @ f
        camera_0.screen.show_image(im, ax=axes[1], pm=False, cmap="viridis", colorbar=False)
        axes[1].plot(toroidal_axis_UV[0][vis[0], 1], toroidal_axis_UV[0][vis[0], 0], "k--")
        axes[1].plot(toroidal_axis_UV[1][vis[1], 1], toroidal_axis_UV[1][vis[1], 0], "k--")
        fig.colorbar(axes[1]._children[0], label="Projected Intensity (a.u.)", ax=axes[1], fraction=0.046, pad=0.04)
        return fig, axes


    # --- Plot sample emission profiles ---
    # sample 0: axisymmetric
    sample_params = {"m_": 0, "n_": 0, "delta": 0.,
                     "d": 3, "r_1": 0.8, "xi_0": 0.,
                     "A": 1, "p": 2, "q": 17, "h": 0, "w": 0.2}
    symmetric_f = emission_profile(r, theta, phi, **sample_params)
    fig, axes = plot_emission_profile_sample(symmetric_f)
    fig.suptitle("Sample Emission Profile: Axisymmetric")
    fig.show()

    # sample 1: hollow tube
    sample_params = {"m_": 1, "n_": 0, "delta": 0.,
                     "d": 3, "r_1": 0.8, "xi_0": 0.,
                     "A": 1, "p": 2, "q": 17, "h": 1, "w": 0.2}
    tube_f = emission_profile(r, theta, phi, **sample_params)
    fig, axes = plot_emission_profile_sample(tube_f)
    fig.suptitle("Sample Emission Profile: Hollow Tube")
    fig.show()

    # sample 2: helical
    sample_params = {"m_": 1, "n_": 1, "delta": 0.1,
                     "d": 3, "r_1": 0.8, "xi_0": 0.3,
                     "A": 1, "p": 2, "q": 9, "h": 0., "w": 0.2}
    helical_f = emission_profile(r, theta, phi, **sample_params)
    fig, axes = plot_emission_profile_sample(helical_f)
    fig.suptitle("Sample Emission Profile: Helical Structure")
    fig.show()

    # --- Plot sample emission profile ---

    sample_params = {"m_": 1, "n_": 1, "delta": 0.1,
                     "d": 3, "r_1": 0.8, "xi_0": 0.3,
                     "A": 1, "p": 2, "q": 9, "h": 0.8, "w": 0.2}
    hellical_hollow_f = emission_profile(r, theta, phi, **sample_params)
    fig, axes = plot_emission_profile_sample(hellical_hollow_f)
    fig.suptitle("Sample Emission Profile: Helical Hollow Structure")
    fig.show()

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    f_list = [emission_profile(r, theta, phi, **sample_params, phi_0=2 * np.pi * i / 12) for i in range(axes.size)]
    ims = P * np.stack(f_list, axis=1)
    ims -= np.mean(ims, axis=1, keepdims=True)
    for i, (im, ax) in enumerate(zip(ims.T, axes.ravel())):
        vmax = np.abs(ims).max()
        _ = camera_0.screen.show_image(im, pm=True, cmap="bwr", ax=ax, colorbar=False,
                                       vmin=-vmax, vmax=vmax)
        ax.text(0.1, 0.9, rf"2$\pi$*{i}/{len(axes.ravel())}",
                transform=ax.transAxes,
                color="black", fontsize=12, fontweight="bold",
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))
    fig.colorbar(_._children[0],
                 ax=axes, location='right', fraction=0.02, pad=0.04, label="Projected Intensity (a.u.)")
    fig.show()

    fig = go.Figure()
    camera_0.draw_camera_orientation_plotly(fig, axis_length=50, show_fig=False)
    stl_utils.plotly_show_stl(mst_wall, fig, color="lightgrey", linearg={'width': 0.5, 'color': 'black'},
                              opacity=0.1, show_fig=False)
    fig.add_trace(go.Volume(x=X, y=Y, z=Z, value=hellical_hollow_f,
                            isomin=0.01 * np.max(hellical_hollow_f), isomax=np.max(hellical_hollow_f),
                            colorscale="Jet", opacity=0.5, surface_count=21,
                            # opacityscale=[[0, 0], [0.1, 0.05], [0.3, 0.1], [0.6, 0.3], [1, 0.6]],
                            opacityscale="extreme",
                            caps=dict(x_show=True, y_show=False, z_show=False),  # no caps
                            ))
    fig.show()
