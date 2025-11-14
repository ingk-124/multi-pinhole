import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from multi_pinhole import *

if __name__ == "__main__":
    # Create objects
    voxel = Voxel.uniform_voxel(ranges=[[-10, 10], [-10, 10], [-10, 10]], shape=[10, 10, 10])
    camera = Camera(eyes=[Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.5, focal_length=25, position=[0, 0])],
                    screen=Screen(screen_shape="rectangle", screen_size=[10, 10], pixel_shape=(20, 20),
                                  subpixel_resolution=3),
                    apertures=Aperture(shape="circle", size=10, position=[0, 0, 40]).set_model(resolution=30,
                                                                                               max_size=50),
                    camera_position=[0, 0, -150]).set_rotation_matrix("zxz", (0, 0, 0), degrees=True)
    world = World(voxel=voxel, cameras=camera)
    x, y, z = voxel.gravity_center.T
    world.set_inside_vertices(lambda x, y, z: (x ** 2 + y ** 2 + z ** 2) <= 9 ** 2)

    fig = go.Figure()
    camera.draw_camera_orientation_plotly(fig, axis_length=50, show_fig=False)
    fig.show()

    # Simulate imaging
    world.set_projection_matrix(res=3, verbose=1, parallel=5)

    # sample_profile
    f = x ** 2 + y ** 2
    im_sub = world.projection[0][0] @ f.flatten()
    im = world.P_matrix[0] @ f.flatten()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].set_title("Subpixel profile")
    camera.screen.show_image(im_sub, ax=axes[0])
    axes[1].set_title("Pixel profile")
    camera.screen.show_image(im, ax=axes[1])
    fig.subplots_adjust(wspace=0.7)
    fig.show()

    # Visualize results
    fig = go.Figure()
    fig.add_trace(go.Volume(x=x, y=y, z=z, value=f,
                            opacity=0.1,
                            surface_count=15, colorscale='Viridis', name='Object'))
    fig.show()
