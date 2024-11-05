# Github arguments.py

import argparse

def parse_args():
    """
    Parse command-line arguments for configuring training and evaluation settings.

    Returns:
        Parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME',
                        help='Target dataset (e.g., "h36m" or "humaneva")')
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME',
                        help='2D keypoints data to use')
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='Comma-separated list of training subjects')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST',
                        help='Comma-separated list of test subjects')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='Comma-separated list of unlabeled subjects for self-supervision')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='Comma-separated list of actions to train/test on, or "*" for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-frequency', default=10, type=int, metavar='N',
                        help='Frequency (in epochs) to create checkpoints')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='File name of checkpoint to resume training from')
    parser.add_argument('--evaluate', default='pretrained_h36m_detectron_coco.bin', type=str, metavar='FILENAME',
                        help='File name of checkpoint to evaluate')
    parser.add_argument('--render', action='store_true', help='Visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='Break down evaluation error by subject')
    parser.add_argument('--export-training-curves', action='store_true', help='Save training curves as .png images')

    # Model arguments
    parser.add_argument('-s', '--stride', default=1, type=int, metavar='N',
                        help='Chunk size for training')
    parser.add_argument('-e', '--epochs', default=60, type=int, metavar='N',
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', default=1024, type=int, metavar='N',
                        help='Batch size in terms of predicted frames')
    parser.add_argument('-drop', '--dropout', default=0.25, type=float, metavar='P',
                        help='Dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='Initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR',
                        help='Learning rate decay per epoch')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
                        help='Disable data augmentation during training')
    parser.add_argument('-no-tta', '--no-test-time-augmentation', dest='test_time_augmentation', action='store_false',
                        help='Disable test-time augmentation')
    parser.add_argument('-arc', '--architecture', default='3,3,3,3,3', type=str, metavar='LAYERS',
                        help='Filter widths for convolutional layers, comma-separated')
    parser.add_argument('--causal', action='store_true', help='Use causal convolutions for real-time processing')
    parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N',
                        help='Number of channels in convolution layers')

    # Experimental settings for semi-supervised and optimization settings
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION',
                        help='Fraction of dataset to use')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='Downsample frame rate by a factor (semi-supervised)')
    parser.add_argument('--warmup', default=1, type=int, metavar='N',
                        help='Number of warm-up epochs for semi-supervision')
    parser.add_argument('--no-eval', action='store_true', help='Disable epoch evaluation during training')
    parser.add_argument('--dense', action='store_true', help='Use dense convolutions instead of dilated')
    parser.add_argument('--disable-optimizations', action='store_true', help='Disable optimized model for single-frame predictions')
    parser.add_argument('--linear-projection', action='store_true', help='Use linear coefficients for semi-supervised projection')
    parser.add_argument('--no-bone-length', action='store_false', dest='bone_length_term',
                        help='Disable bone length term in semi-supervised settings')
    parser.add_argument('--no-proj', action='store_true', help='Disable projection for semi-supervised setting')

    # Visualization settings for rendering videos
    parser.add_argument('--viz-subject', type=str, metavar='STR', help='Subject to render')
    parser.add_argument('--viz-action', type=str, metavar='STR', help='Action to render')
    parser.add_argument('--viz-camera', type=int, default=0, metavar='N', help='Camera to render')
    parser.add_argument('--viz-video', type=str, metavar='PATH', help='Path to input video')
    parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='Skip the first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH', help='Output file name (.gif or .mp4)')
    parser.add_argument('--viz-bitrate', type=int, default=30000, metavar='N', help='Bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true', help='Do not show ground-truth poses in visualization')
    parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='Limit rendering to the first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='Downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='Image size for rendering')

    # Custom options for additional input files
    parser.add_argument('--input-npz', dest='input_npz', type=str, default='', help='Input 2D numpy file')
    parser.add_argument('--video', dest='input_video', type=str, default='', help='Input video file name')

    # Set default values for certain options
    parser.set_defaults(bone_length_term=True)
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True)

    args = parser.parse_args()

    # Validate configuration: check for incompatible options
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()

    return args
