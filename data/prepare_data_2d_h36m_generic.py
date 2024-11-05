import argparse
import os
import re
import sys
from glob import glob

import ipdb
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
    # Check if the script is run from the correct directory
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)

    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description='Human3.6M dataset converter')
    parser.add_argument('-i', '--input', default='', type=str, metavar='PATH', help='input path to 2D detections')
    parser.add_argument('-o', '--output', default='', type=str, metavar='PATH', help='output suffix for 2D detections (e.g. detectron_pt_coco)')
    args = parser.parse_args()

    # Check if input directory is specified
    if not args.input:
        print('Please specify the input directory')
        exit(0)

    # Check if output suffix is specified
    if not args.output:
        print('Please specify an output suffix (e.g. detectron_pt_coco)')
        exit(0)

    # Select appropriate function and metadata based on the output suffix
    import_func = suggest_pose_importer(args.output)
    metadata = suggest_metadata(args.output)

    print('Parsing 2D detections from', args.input)
    output = {}

    # Directly import keypoints from the input path
    keypoints = import_func(args.input)
    output['S1'] = {'Walking': [None, None, None, None]}
    output['S1']['Walking'][0] = keypoints.astype(np.float32)

    # Save parsed keypoints to compressed .npz format
    np.savez_compressed(output_prefix_2d + '00' + args.output, positions_2d=output, metadata=metadata)

    # Load the data for verification
    data = np.load('data_2d_h36m_detectron_pt_coco.npz')
    data1 = np.load('data_2d_h36m_00detectron_pt_coco.npz')
    actions = data['positions_2d'].item()
    actions1 = data1['positions_2d'].item()
    meta = data['metadata']

    # Update action data with new keypoints
    actions['S1']['Walking'][0] = actions1['S1']['Walking'][0][:, :, :]
    np.savez_compressed('data_2d_h36m_lxy_cpn_ft_h36m_dbb.npz', positions_2d=actions, metadata=meta)

    # Terminate program
    os.exit()
    ipdb.set_trace()

    # Match all files with the specified format
    file_list = glob(args.input + '/S*/*.mp4.npz')
    for f in file_list:
        path, fname = os.path.split(f)
        subject = os.path.basename(path)
        assert subject.startswith('S'), f'{subject} does not look like a subject directory'

        # Skip files with '_ALL' in the filename
        if '_ALL' in fname:
            continue

        # Extract action and camera details from filename
        m = re.search(r'(.*)\.([0-9]+)\.mp4\.npz', fname)
        action, camera = m.group(1), m.group(2)
        camera_idx = cam_map[camera]

        # Skip corrupted video file
        if subject == 'S11' and action == 'Directions':
            continue

        # Normalize action names for consistency
        canonical_name = action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')

        # Import keypoints and verify shape
        keypoints = import_func(f)
        assert keypoints.shape[1] == metadata['num_joints']

        # Organize data structure by subject and action
        if subject not in output:
            output[subject] = {}
        if canonical_name not in output[subject]:
            output[subject][canonical_name] = [None, None, None, None]
        output[subject][canonical_name][camera_idx] = keypoints.astype('float32')

    # Save final output to a compressed .npz file
    print('Saving...')
    np.savez_compressed(output_prefix_2d + args.output, positions_2d=output, metadata=metadata)
    print('Done.')
