import h5py
import numpy as np

# Metadata definitions for different keypoint layouts
mpii_metadata = {
    'layout_name': 'mpii',
    'num_joints': 16,
    'keypoints_symmetry': [
        [3, 4, 5, 13, 14, 15],
        [0, 1, 2, 10, 11, 12],
    ]
}

coco_metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ]
}

h36m_metadata = {
    'layout_name': 'h36m',
    'num_joints': 17,
    'keypoints_symmetry': [
        [4, 5, 6, 11, 12, 13],
        [1, 2, 3, 14, 15, 16],
    ]
}

humaneva15_metadata = {
    'layout_name': 'humaneva15',
    'num_joints': 15,
    'keypoints_symmetry': [
        [2, 3, 4, 8, 9, 10],
        [5, 6, 7, 11, 12, 13]
    ]
}

humaneva20_metadata = {
    'layout_name': 'humaneva20',
    'num_joints': 20,
    'keypoints_symmetry': [
        [3, 4, 5, 6, 11, 12, 13, 14],
        [7, 8, 9, 10, 15, 16, 17, 18]
    ]
}


# Function to suggest metadata based on layout name
def suggest_metadata(name):
    names = []
    # Check each metadata set for a matching layout name
    for metadata in [mpii_metadata, coco_metadata, h36m_metadata, humaneva15_metadata, humaneva20_metadata]:
        if metadata['layout_name'] in name:
            return metadata  # Return the matching metadata if found
        names.append(metadata['layout_name'])
    # Raise an error if no matching metadata layout is found
    raise KeyError('Cannot infer keypoint layout from name "{}". Tried {}.'.format(name, names))


# Function to import poses from Detectron file
def import_detectron_poses(path):
    # Load data with Latin1 encoding as Detectron typically uses Python 2.7
    data = np.load(path, encoding='latin1')
    kp = data['keypoints']
    bb = data['boxes']
    results = []
    
    # Process each bounding box and keypoint entry
    for i in range(len(bb)):
        if len(bb[i][1]) == 0:  # Handle cases with no detections
            assert i > 0
            results.append(results[-1])  # Use last pose if no detection in current frame
            continue
        best_match = np.argmax(bb[i][1][:, 4])  # Find detection with the highest confidence score
        keypoints = kp[i][1][best_match].T.copy()  # Extract keypoints for the best match
        results.append(keypoints)
        
    results = np.array(results)
    return results[:, :, [0, 1, 3]]  # Extract x, y, and score for each keypoint


# Empty function for custom pose processing; can be expanded as needed
def my_pose(path):
    data = np.load(path, encoding='latin1')


# Function to import poses from CPN (Cascaded Pyramid Network) format
def import_cpn_poses(path):
    data = np.load(path)
    kp = data['keypoints']
    return kp[:, :, :2]  # Return only x, y positions


# Function to import SH (Stanford Hand) pose data from HDF5 file
def import_sh_poses(path):
    with h5py.File(path) as hf:
        positions = hf['poses'].value  # Load pose data from the HDF5 file
    return positions.astype('float32')  # Ensure the data is in float32 format


# Function to suggest a pose importer based on file naming conventions
def suggest_pose_importer(name):
    if 'detectron' in name:
        return import_detectron_poses
    if 'cpn' in name:
        return import_cpn_poses
    if 'sh' in name:
        return import_sh_poses
    raise KeyError('Cannot infer keypoint format from name "{}". Tried detectron, cpn, sh.'.format(name))
