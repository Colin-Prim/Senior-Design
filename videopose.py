import os
import time

from common.arguments import parse_args
from common.camera import *
from common.generators import UnchunkedGenerator
from common.loss import *
from common.model import *
from common.utils import Timer, evaluate, add_path
import cv2
import numpy as np
from bvh_skeleton import openpose_skeleton, h36m_skeleton, cmu_skeleton, smartbody_skeleton

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
}

add_path()


# Record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


time0 = ckpt_time()


def get_detector_2d(detector_name):
    def get_alpha_pose():
        from joints_detectors.Alphapose.gene_npz import generate_kpts as alpha_pose
        return alpha_pose

    def get_hr_pose():
        from joints_detectors.hrnet.pose_estimation.video import generate_kpts as hr_pose
        return hr_pose

    detector_map = {
        'alpha_pose': get_alpha_pose,
        'hr_pose': get_hr_pose,
    }

    assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

    return detector_map[detector_name]()


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]


def main(args):
    # Step 1: Detect 2D keypoints
    detector_2d = get_detector_2d(args.detector_2d)

    assert detector_2d, 'detector_2d should be in (alpha, hr)'

    # Load or generate 2D keypoints
    if not args.input_npz:
        video_name = args.viz_video
        keypoints = detector_2d(video_name)
    else:
        npz = np.load(args.input_npz)
        keypoints = npz['kpts']  # (N, 17, 2)

    # Step 2: Convert 2D keypoints to 3D keypoints
    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    # Normalize keypoints using the camera parameters
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)

    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal,
                              dropout=args.dropout, channels=args.channels, dense=args.dense)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    ckpt, time1 = ckpt_time(time0)
    print('-------------- Load data takes {:.2f} seconds'.format(ckpt))

    # Load trained model
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location='cpu')  # Use 'cpu' for compatibility
    model_pos.load_state_dict(checkpoint['model_pos'])

    ckpt, time2 = ckpt_time(time1)
    print('-------------- Load 3D model takes {:.2f} seconds'.format(ckpt))

    # Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    print('Rendering...')
    input_keypoints = keypoints.copy()
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, model_pos, return_predictions=True)

    # Save 3D joint points
    np.save('outputs/test_3d_output.npy', prediction, allow_pickle=True)

    # Step 3: Convert predicted 3D points from camera coordinates to world coordinates
    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    prediction = camera_to_world(prediction, R=rot, t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])  # Rebase height

    # Step 4: Output 3D keypoints and convert predicted 3D points to BVH skeleton
    write_3d_point(args.viz_output, prediction)
    prediction_copy = np.copy(prediction)
    write_standard_bvh(args.viz_output, prediction_copy)  # Convert to standard BVH
    write_smartbody_bvh(args.viz_output, prediction_copy)  # Convert to SmartBody BVH

    anim_output = {'Reconstruction': prediction}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

    ckpt, time3 = ckpt_time(time2)
    print('-------------- Generate reconstruction 3D data takes {:.2f} seconds'.format(ckpt))

    if not args.viz_output:
        args.viz_output = 'outputs/outputvideo/alpha_result.mp4'

    # Step 5: Generate output video
    # from common.visualization import render_animation
    # render_animation(input_keypoints, anim_output,
    #                  Skeleton(), 25, args.viz_bitrate, np.array(70., dtype=np.float32), args.viz_output,
    #                  limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
    #                  input_video_path=args.viz_video, viewport=(1000, 1002),
    #                  input_video_skip=args.viz_skip)

    ckpt, time4 = ckpt_time(time3)
    print('Total time spent: {:2f} seconds'.format(ckpt))


def inference_video(video_path, detector_2d):
    """
    Do image -> 2D points -> 3D points to video.
    :param detector_2d: used 2D joints detector. Can be {alpha_pose, hr_pose}
    :param video_path: relative to outputs
    :return: None
    """
    args = parse_args()

    args.detector_2d = detector_2d
    dir_name = os.path.dirname(video_path)
    dir_name_split = dir_name[:dir_name.rfind('/')]
    new_dir_name = os.path.join(dir_name_split, 'outputvideo')

    basename = os.path.basename(video_path)
    video_name = basename[:basename.rfind('.')]

    args.viz_video = video_path
    args.viz_output = f'{new_dir_name}/{args.detector_2d}_{video_name}.mp4'

    args.evaluate = 'pretrained_h36m_detectron_coco.bin'

    with Timer(video_path):
        main(args)


def modify_video_frame_rate(videoPath, destFps):
    dir_name = os.path.dirname(videoPath)
    basename = os.path.basename(videoPath)
    video_name = basename[:basename.rfind('.')]
    video_name = video_name + "modify_frame_rate"
    resultVideoPath = f'{dir_name}/{video_name}.mp4'

    videoCapture = cv2.VideoCapture(videoPath)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if fps != destFps:
        frameSize = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        videoWriter = cv2.VideoWriter(resultVideoPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), destFps, frameSize)

        i = 0
        print('Starting video frame rate conversion')
        while True:
            success, frame = videoCapture.read()
            if success:
                i += 1
                print('Converted to frame %d' % i)
                videoWriter.write(frame)
            else:
                print('Video frame conversion ended')
                break
    return resultVideoPath


def write_3d_point(outvideopath, prediction3dpoint):
    '''
    :param prediction3dpoint: Predicted 3D points dictionary
    :param outvideopath: Output path for the 3D points file
    :return:
    '''
    dir_name = os.path.dirname(outvideopath)
    basename = os.path.basename(outvideopath)
    video_name = basename[:basename.rfind('.')]

    frameNum = 1

    for frame in prediction3dpoint:
        outfileDirectory = os.path.join(dir_name, video_name, "3dpoint")
        if not os.path.exists(outfileDirectory):
            os.makedirs(outfileDirectory)
        outfilename = os.path.join(dir_name, video_name, "3dpoint", "3dpoint{}.txt".format(frameNum))
        file = open(outfilename, 'w')
        frameNum += 1
        for point3d in frame:
            str = '{},{},{}\n'.format(point3d[0], point3d[1], point3d[2])
            file.write(str)
        file.close()


def write_standard_bvh(outbvhfilepath, prediction3dpoint):
    '''
    :param outbvhfilepath: Output BVH action file path
    :param prediction3dpoint: Predicted 3D joint points
    :return:
    '''

    # Scale the predicted points by 100 times
    for frame in prediction3dpoint:
        for point3d in frame:
            point3d[0] *= 100
            point3d[1] *= 100
            point3d[2] *= 100

    dir_name = os.path.dirname(outbvhfilepath)
    basename = os.path.basename(outbvhfilepath)
    video_name = basename[:basename.rfind('.')]
    bvhfileDirectory = os.path.join(dir_name, video_name, "bvh")
    if not os.path.exists(bvhfileDirectory):
        os.makedirs(bvhfileDirectory)
    bvhfileName = os.path.join(dir_name, video_name, "bvh", "{}.bvh".format(video_name))

    human36m_skeleton = h36m_skeleton.H36mSkeleton()
    human36m_skeleton.poses2bvh(prediction3dpoint, output_file=bvhfileName)


def write_smartbody_bvh(outbvhfilepath, prediction3dpoint):
    '''
    :param outbvhfilepath: Output BVH action file path
    :param prediction3dpoint: Predicted 3D joint points
    :return:
    '''

    # Convert predicted points to SmartBody format
    for frame in prediction3dpoint:
        for point3d in frame:
            X = point3d[0]
            Y = point3d[1]
            Z = point3d[2]

            point3d[0] = -X
            point3d[1] = Z
            point3d[2] = Y

    dir_name = os.path.dirname(outbvhfilepath)
    basename = os.path.basename(outbvhfilepath)
    video_name = basename[:basename.rfind('.')]
    bvhfileDirectory = os.path.join(dir_name, video_name, "bvh")
    if not os.path.exists(bvhfileDirectory):
        os.makedirs(bvhfileDirectory)
    bvhfileName = os.path.join(dir_name, video_name, "bvh", "{}.bvh".format(video_name))

    SmartBody_skeleton = smartbody_skeleton.SmartBodySkeleton()
    SmartBody_skeleton.poses2bvh(prediction3dpoint, output_file=bvhfileName)


if __name__ == '__main__':
    inference_video('TaiChi.mp4', 'alpha_pose')
