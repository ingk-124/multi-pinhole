# import libraries
import gc

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from stl import mesh

from multi_pinhole import *
from utils import stl_utils


def shifted_torus(r, theta, phi, delta):
    R = r * np.cos(theta) - delta
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
    # force_rebuild = True
    force_rebuild = False
    try:
        if force_rebuild:
            raise FileNotFoundError
        world = World.load_world("MST_tangential_double-pinhole_wide.pkl")
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
            screen=Screen(screen_shape="rectangle", screen_size=[7.5, 16], pixel_shape=(61, 125),
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

    if not file_is_exist:
        world.save_world(filename="MST_tangential_double-pinhole_wide.pkl")

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

    # --- Plot sample emission profile ---

    sample_params = {"m_": 1, "n_": -1, "delta": 0.1,
                     "d": 2, "r_1": 0.6, "xi_0": 0.4,
                     "A": 1, "p": 3, "q": 7, "h": 0, "w": 1.0}

    sample_f = emission_profile(r, theta, phi, **sample_params)
    plt.contourf(voxel.cx_axis, voxel.cz_axis,
                 sample_f.reshape(voxel.shape)[:, voxel.shape[1] // 2, :].T,
                 levels=15, cmap="viridis")
    plt.plot(-1500 + 500 * np.cos(np.linspace(0, 2 * np.pi, 100)),
             500 * np.sin(np.linspace(0, 2 * np.pi, 100)), "w--", linewidth=2,
             label="Chamber wall")
    plt.xlim(*voxel.cx_axis[[0, -1]])
    plt.ylim(*voxel.cz_axis[[0, -1]])
    plt.title("Sample Emission Profile Slice at Y=0")
    plt.xlabel("X [mm]")
    plt.ylabel("Z [mm]")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.colorbar(label="Emission Intensity")
    plt.show()

    fig = go.Figure()
    camera_0.draw_camera_orientation_plotly(fig, axis_length=50, show_fig=False)
    stl_utils.plotly_show_stl(mst_wall, fig, color="lightgrey", linearg={'width': 0.5, 'color': 'black'},
                              opacity=0.1, show_fig=False)
    fig.add_trace(go.Volume(x=X, y=Y, z=Z, value=sample_f,
                            isomin=0.01 * np.max(sample_f), isomax=np.max(sample_f),
                            colorscale="Jet", opacity=0.5, surface_count=21,
                            # opacityscale=[[0, 0], [0.1, 0.05], [0.3, 0.1], [0.6, 0.3], [1, 0.6]],
                            opacityscale="extreme",
                            caps=dict(x_show=True, y_show=False, z_show=False),  # no caps
                            ))
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
