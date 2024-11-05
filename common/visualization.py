import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from common.utils import read_video


# Timing function to checkpoint elapsed time
def ckpt_time(ckpt=None, display=0, desc=''):
    if not ckpt:
        return time.time()
    else:
        if display:
            print(desc + ' consume time {:0.4f}'.format(time.time() - float(ckpt)))
        return time.time() - float(ckpt), time.time()


# Ensure 3D plots have equal axis proportions
def set_equal_aspect(ax, data):
    X, Y, Z = data[..., 0], data[..., 1], data[..., 2]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    
    # Create bounding box for equal aspect ratio
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


# Reduce the size of a tensor by downsampling
def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


# Render a 3D animation using 2D keypoints and 3D poses
def render_animation(keypoints, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
    """
    Render animation in different formats: interactive, .mp4, .gif.

    Arguments:
    - keypoints: 2D keypoints of the skeleton
    - poses: Dictionary of 3D poses to visualize
    - skeleton: Skeleton object for parent-child joint structure
    - fps: Frames per second of the output animation
    - bitrate: Bitrate of the output file (for video files)
    - azim: Azimuth angle for the 3D plot
    - output: Output file path
    - viewport: Viewport dimensions
    - limit: Number of frames to limit rendering (default -1 renders all frames)
    - downsample: Factor to reduce frames per second
    - size: Size of the figure
    - input_video_path: Optional path to overlay input video as background
    - input_video_skip: Frames to skip at the start of the video
    """
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    # Initialize 3D axes and set properties
    _ = Axes3D.__class__.__name__  # Workaround for matplotlib Axes3D issue
    ax_3d, lines_3d, trajectories = [], [], []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.dist = 12.5
        ax.set_title(title)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])  # Track root joint for centering

    poses = list(poses.values())  # Flatten poses dictionary

    # Decode video if provided; otherwise create black background
    if input_video_path is None:
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        all_frames = [f for f in read_video(input_video_path, fps=None, skip=input_video_skip)]
        all_frames = all_frames[:min(keypoints.shape[0], len(all_frames))]

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        poses = [downsample_tensor(p, downsample) for p in poses]
        trajectories = [downsample_tensor(t, downsample) for t in trajectories]
        fps /= downsample

    # Initialize animation variables
    initialized = False
    image = None
    points = None
    limit = min(limit if limit > 0 else len(all_frames), len(all_frames))
    parents = skeleton.parents()

    # Progress bar for rendering
    pbar = tqdm(total=limit)

    # Animation frame update function
    def update_video(i):
        nonlocal initialized, image, points

        # Center each frame based on root joint trajectory
        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

        # Initialize plot on first frame
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
            points = ax_in.scatter(*keypoints[i].T, 5, color='red', edgecolors='white', zorder=10)
            initialized = True
        else:
            image.set_data(all_frames[i])  # Update video frame
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
            points.set_offsets(keypoints[i])

        pbar.update()

    # Create animation
    fig.tight_layout()
    anim = FuncAnimation(fig, update_video, frames=limit, interval=1000.0 / fps, repeat=False)
    if output.endswith('.mp4'):
        writer = writers['ffmpeg'](fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=60, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    pbar.close()
    plt.close()


def render_animation_test(keypoints, poses, skeleton, fps, bitrate, azim, output, viewport, limit=-1, downsample=1, size=6, input_video_frame=None,
                          input_video_skip=0, num=None):
    # Set up for individual frame testing
    t0 = ckpt_time()
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    fig.add_subplot(121)
    plt.imshow(input_video_frame)
    ax = fig.add_subplot(122, projection='3d')
    ax.view_init(elev=15., azim=azim)

    radius = 1.7
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    parents = skeleton.parents()
    pos = poses['Reconstruction'][-1]
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        col = 'red' if j in skeleton.joints_right() else 'black'
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                [pos[j, 1], pos[j_parent, 1]],
                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)

    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    cv2.imshow('im', image)
    cv2.waitKey(5)
    return image
