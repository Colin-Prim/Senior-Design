import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_3d_keypoints(keypoints_3d, skeleton_structure=None, elev=20, azim=135):
    """
    Visualizes 3D keypoints as a skeleton using Matplotlib.

    Args:
        keypoints_3d (np.ndarray): 3D keypoints (shape: [num_keypoints, 3]).
        skeleton_structure (list): List of tuples indicating connections between joints.
        elev (int): Elevation angle for the 3D plot.
        azim (int): Azimuth angle for the 3D plot.
    """
    if keypoints_3d is None or len(keypoints_3d) == 0:
        print("No 3D keypoints to visualize.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # Plot keypoints
    for x, y, z in keypoints_3d:
        ax.scatter(x, y, z, color='b', s=20)  # Blue for keypoints

    # Plot skeleton connections
    if skeleton_structure:
        for (start, end) in skeleton_structure:
            if start < len(keypoints_3d) and end < len(keypoints_3d):
                xs, ys, zs = [keypoints_3d[start, i] for i in range(3)], [keypoints_3d[end, i] for i in range(3)]
                ax.plot(xs, ys, zs, color='r')  # Red for skeleton lines

    # Set plot limits
    max_range = np.array([keypoints_3d[:, 0].max() - keypoints_3d[:, 0].min(),
                          keypoints_3d[:, 1].max() - keypoints_3d[:, 1].min(),
                          keypoints_3d[:, 2].max() - keypoints_3d[:, 2].min()]).max() / 2.0

    mid_x = (keypoints_3d[:, 0].max() + keypoints_3d[:, 0].min()) * 0.5
    mid_y = (keypoints_3d[:, 1].max() + keypoints_3d[:, 1].min()) * 0.5
    mid_z = (keypoints_3d[:, 2].max() + keypoints_3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
