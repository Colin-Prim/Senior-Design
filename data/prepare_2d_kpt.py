import argparse
import os
import sys

import numpy as np
from data_utils import suggest_metadata, suggest_pose_importer

# Adding the parent directory to the Python path for imports
sys.path.append('../')

# Prefix for output file names
output_prefix_2d = 'data_2d_h36m_'

# Mapping of camera IDs to indices for the Human3.6M dataset
cam_map = {
    '54138969': 0,
    '55011271': 1,
    '58860488': 2,
    '60457274': 3,
}

if __name__ == '__main__':
    # Ensure the script is run from the correct directory
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)

    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description='Human3.6M dataset converter')

    # Input argument for specifying the path to 2D detections
    parser.add_argument('-i', '--input', default='', type=str, metavar='PATH', help='input path to 2D detections')

    # Output suffix for the processed 2D detections
    parser.add_argument('-o', '--output', default='detectron_pt_coco', type=str, metavar='PATH',
                        help='output suffix for 2D detections (e.g. detectron_pt_coco)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the input directory is specified
    if not args.input:
        print('Please specify the input directory')
        exit(0)

    # Select the appropriate function and metadata based on the output name
    import_func = suggest_pose_importer('detectron_pt_coco')
    metadata = suggest_metadata('detectron_pt_coco')

    # Parse the 2D detections from the specified input path
    print('Parsing 2D detections from', args.input)
    keypoints = import_func(args.input)

    # Convert keypoints to float32 format
    output = keypoints.astype(np.float32)

    # Save the processed 2D detections and metadata in a compressed .npz format
    np.savez_compressed(output_prefix_2d + 'test' + args.output, positions_2d=output, metadata=metadata)
    print('npz name is ', output_prefix_2d + 'test' + args.output)
