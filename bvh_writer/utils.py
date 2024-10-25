import numpy as np

def scale_3d_keypoints(keypoints, scale_factor):
    """
    Scales 3D keypoints by a given factor.

    Args:
        keypoints (list): List of 3D keypoints.
        scale_factor (float): Scaling factor.

    Returns:
        list: Scaled 3D keypoints.
    """
    return [[x * scale_factor, y * scale_factor, z * scale_factor] for x, y, z in keypoints]

def normalize_3d_keypoints(keypoints):
    """
    Normalizes 3D keypoints to have a mean of 0.

    Args:
        keypoints (list): List of 3D keypoints.

    Returns:
        list: Normalized 3D keypoints.
    """
    keypoints = np.array(keypoints)
    mean = np.mean(keypoints, axis=0)
    return (keypoints - mean).tolist()

def convert_to_bvh_coordinate_system(keypoints):
    """
    Converts 3D keypoints to the BVH coordinate system (Z-up, Y-forward).

    Args:
        keypoints (list): List of 3D keypoints.

    Returns:
        list: 3D keypoints in the BVH coordinate system.
    """
    converted_keypoints = []
    for x, y, z in keypoints:
        converted_keypoints.append([x, z, -y])  # Swap Y and Z, and negate Y

    return converted_keypoints

def get_joint_offsets(skeleton):
    """
    Extracts joint offsets from the skeleton definition.

    Args:
        skeleton (dict): The skeleton structure.

    Returns:
        list: List of joint offsets.
    """
    offsets = []

    def _extract_offsets(joint):
        offsets.append(joint['offset'])
        for child in joint.get('children', []):
            _extract_offsets(child)

    _extract_offsets(skeleton['root'])
    return offsets

def calculate_bone_lengths(keypoints_3d, skeleton):
    """
    Calculates bone lengths based on 3D keypoints and the skeleton structure.

    Args:
        keypoints_3d (list): List of 3D keypoints.
        skeleton (dict): The skeleton structure.

    Returns:
        dict: Dictionary of bone lengths.
    """
    bone_lengths = {}

    def _calculate_length(joint, parent_pos):
        joint_name = joint['name']
        joint_pos = keypoints_3d[joint['index']]

        length = np.linalg.norm(np.array(joint_pos) - np.array(parent_pos))
        bone_lengths[joint_name] = length

        for child in joint.get('children', []):
            _calculate_length(child, joint_pos)

    root_pos = keypoints_3d[0]  # Assuming the first keypoint is the root
    _calculate_length(skeleton['root'], root_pos)

    return bone_lengths
