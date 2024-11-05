import argparse
import os
import sys
import tarfile
import zipfile
from glob import glob
from shutil import rmtree

import h5py
import numpy as np

# Adding parent directory to sys.path for imports
sys.path.append('../')

# Define output file names and subject list for Human3.6M dataset
output_filename_pt = 'data_2d_h36m_sh_pt_mpii'  # Output filename for pre-trained data
output_filename_ft = 'data_2d_h36m_sh_ft_h36m'  # Output filename for fine-tuned data
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']  # Human3.6M subjects

# Mapping of camera IDs to indices
cam_map = {
    '54138969': 0,
    '55011271': 1,
    '58860488': 2,
    '60457274': 3,
}

# Metadata for the dataset, including joint symmetry information
metadata = {
    'num_joints': 16,
    'keypoints_symmetry': [
        [3, 4, 5, 13, 14, 15],  # Left-side joints
        [0, 1, 2, 10, 11, 12],  # Right-side joints
    ]
}


def process_subject(subject, file_list, output):
    """
    Process each subject and extracts poses from .h5 files.

    Parameters:
    subject -- Human3.6M subject identifier (e.g., 'S1', 'S5', etc.)
    file_list -- List of .h5 files containing 2D keypoint data
    output -- Dictionary to store the 2D keypoints organized by subject and action
    """
    # Ensure expected file count for each subject
    if subject == 'S11':
        assert len(file_list) == 119, f"Expected 119 files for subject {subject}, got {len(file_list)}"
    else:
        assert len(file_list) == 120, f"Expected 120 files for subject {subject}, got {len(file_list)}"

    for f in file_list:
        # Extract action and camera ID from filename
        action, cam = os.path.splitext(os.path.basename(f))[0].replace('_', ' ').split('.')

        # Skip corrupted video in subject S11
        if subject == 'S11' and action == 'Directions':
            continue

        # Initialize action entry in output dictionary if not already present
        if action not in output[subject]:
            output[subject][action] = [None, None, None, None]

        # Read pose data from .h5 file and store it in output dictionary
        with h5py.File(f) as hf:
            positions = hf['poses'][:]
            output[subject][action][cam_map[cam]] = positions.astype('float32')


if __name__ == '__main__':
    # Ensure script is run from the 'data' directory
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)

    # Define command-line arguments for specifying paths to pre-trained and fine-tuned datasets
    parser = argparse.ArgumentParser(description='Human3.6M dataset downloader/converter')
    parser.add_argument('-pt', '--pretrained', default='', type=str, metavar='PATH', help='convert pretrained dataset')
    parser.add_argument('-ft', '--fine-tuned', default='', type=str, metavar='PATH', help='convert fine-tuned dataset')
    args = parser.parse_args()

    # Process pre-trained dataset
    if args.pretrained:
        print('Converting pretrained dataset from', args.pretrained)

        # Extract pre-trained dataset
        print('Extracting...')
        with zipfile.ZipFile(args.pretrained, 'r') as archive:
            archive.extractall('sh_pt')

        # Convert extracted data
        print('Converting...')
        output = {}
        for subject in subjects:
            output[subject] = {}
            file_list = glob(f'sh_pt/h36m/{subject}/StackedHourglass/*.h5')
            process_subject(subject, file_list, output)

        # Save the 2D keypoints and metadata as compressed .npz file
        print('Saving...')
        np.savez_compressed(output_filename_pt, positions_2d=output, metadata=metadata)

        # Clean up extracted files
        print('Cleaning up...')
        rmtree('sh_pt')

        print('Done.')

    # Process fine-tuned dataset
    if args.fine_tuned:
        print('Converting fine-tuned dataset from', args.fine_tuned)

        # Extract fine-tuned dataset
        print('Extracting...')
        with tarfile.open(args.fine_tuned, 'r:gz') as archive:
            archive.extractall('sh_ft')

        # Convert extracted data
        print('Converting...')
        output = {}
        for subject in subjects:
            output[subject] = {}
            file_list = glob(f'sh_ft/{subject}/StackedHourglassFineTuned240/*.h5')
            process_subject(subject, file_list, output)

        # Save the 2D keypoints and metadata as compressed .npz file
        print('Saving...')
        np.savez_compressed(output_filename_ft, positions_2d=output, metadata=metadata)

        # Clean up extracted files
        print('Cleaning up...')
        rmtree('sh_ft')

        print('Done.')
