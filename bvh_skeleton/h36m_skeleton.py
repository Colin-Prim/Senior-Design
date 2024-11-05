# This code is adapted from a different GitHub repository.
# It has been modified to define an H36M skeleton structure with keypoints, hierarchical relationships,
# and functions to convert 3D poses to BVH format.

from . import math3d
from . import bvh_helper

import numpy as np


class H36mSkeleton(object):
    """
    Represents the H36M skeleton structure with predefined joint hierarchies and relationships.
    This class provides methods to initialize the skeleton, set joint relationships,
    calculate initial offsets, and convert pose data to Euler angles suitable for BVH format.
    """

    def __init__(self):
        # Define the root joint, which is the Hip.
        self.root = 'Hip'
        
        # Map each keypoint name to an index, as defined in the H36M skeleton format.
        # Each keypoint corresponds to a specific joint in the human skeleton.
        self.keypoint2index = {
            'Hip': 0,
            'RightHip': 1,
            'RightKnee': 2,
            'RightAnkle': 3,
            'LeftHip': 4,
            'LeftKnee': 5,
            'LeftAnkle': 6,
            'Spine': 7,
            'Thorax': 8,
            'Neck': 9,
            'HeadEndSite': 10,
            'LeftShoulder': 11,
            'LeftElbow': 12,
            'LeftWrist': 13,
            'RightShoulder': 14,
            'RightElbow': 15,
            'RightWrist': 16,
            'RightAnkleEndSite': -1,
            'LeftAnkleEndSite': -1,
            'LeftWristEndSite': -1,
            'RightWristEndSite': -1
        }

        # Reverse mapping from indices to keypoints for easier lookup.
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        
        # Count the total number of keypoints.
        self.keypoint_num = len(self.keypoint2index)

        # Define the hierarchy of child joints for each keypoint.
        # Each entry shows the parent joint and its corresponding child joints.
        self.children = {
            'Hip': ['RightHip', 'LeftHip', 'Spine'],
            'RightHip': ['RightKnee'],
            'RightKnee': ['RightAnkle'],
            'RightAnkle': ['RightAnkleEndSite'],
            'RightAnkleEndSite': [],  # End site has no children
            'LeftHip': ['LeftKnee'],
            'LeftKnee': ['LeftAnkle'],
            'LeftAnkle': ['LeftAnkleEndSite'],
            'LeftAnkleEndSite': [],  # End site has no children
            'Spine': ['Thorax'],
            'Thorax': ['Neck', 'LeftShoulder', 'RightShoulder'],
            'Neck': ['HeadEndSite'],
            'HeadEndSite': [],  # Head is an end site and has no children
            'LeftShoulder': ['LeftElbow'],
            'LeftElbow': ['LeftWrist'],
            'LeftWrist': ['LeftWristEndSite'],
            'LeftWristEndSite': [],  # End site has no children
            'RightShoulder': ['RightElbow'],
            'RightElbow': ['RightWrist'],
            'RightWrist': ['RightWristEndSite'],
            'RightWristEndSite': []  # End site has no children
        }

        # Initialize the parent mapping for each keypoint, starting with the root.
        self.parent = {self.root: None}  # Root has no parent
        for parent, children in self.children.items():
            for child in children:
                # Set each child keypoint's parent to the current parent keypoint.
                self.parent[child] = parent
        
        # Identify left and right joints for convenience.
        # Left joints contain "Left" in their name, while right joints contain "Right".
        self.left_joints = [joint for joint in self.keypoint2index if 'Left' in joint]
        self.right_joints = [joint for joint in self.keypoint2index if 'Right' in joint]

        # Define initial T-pose direction for each joint in the Human3.6M coordinate system.
        # In this system, Z is up, Y is forward, and X is to the left.
        self.initial_directions = {
            'Hip': [0, 0, 0],
            'RightHip': [-1, 0, 0],
            'RightKnee': [0, 0, -1],
            'RightAnkle': [0, 0, -1],
            'RightAnkleEndSite': [0, -1, 0],
            'LeftHip': [1, 0, 0],
            'LeftKnee': [0, 0, -1],
            'LeftAnkle': [0, 0, -1],
            'LeftAnkleEndSite': [0, -1, 0],
            'Spine': [0, 0, 1],
            'Thorax': [0, 0, 1],
            'Neck': [0, 0, 1],
            'HeadEndSite': [0, 0, 1],
            'LeftShoulder': [1, 0, 0],
            'LeftElbow': [1, 0, 0],
            'LeftWrist': [1, 0, 0],
            'LeftWristEndSite': [1, 0, 0],
            'RightShoulder': [-1, 0, 0],
            'RightElbow': [-1, 0, 0],
            'RightWrist': [-1, 0, 0],
            'RightWristEndSite': [-1, 0, 0]
        }

    def get_initial_offset(self, poses_3d):
        """
        Calculate the initial offset of each joint based on 3D poses.

        Args:
            poses_3d: Array of 3D joint positions.

        Returns:
            A dictionary of initial offsets for each joint.
        """
        # Initialize dictionary to store bone lengths.
        bone_lens = {self.root: [0]}
        stack = [self.root]
        
        # Traverse the hierarchy using a stack to calculate each joint's length.
        while stack:
            parent = stack.pop()
            p_idx = self.keypoint2index[parent]
            for child in self.children[parent]:
                # For end sites, assign a default bone length of 0.4 of the parent's length.
                if 'EndSite' in child:
                    bone_lens[child] = 0.4 * bone_lens[parent]
                    continue
                stack.append(child)

                c_idx = self.keypoint2index[child]
                # Calculate the bone length between the parent and child joints.
                bone_lens[child] = np.linalg.norm(
                    poses_3d[:, p_idx] - poses_3d[:, c_idx],
                    axis=1
                )

        # Average the bone lengths for symmetric joints to ensure balance in the skeleton.
        bone_len = {}
        for joint in self.keypoint2index:
            if 'Left' in joint or 'Right' in joint:
                base_name = joint.replace('Left', '').replace('Right', '')
                left_len = np.mean(bone_lens['Left' + base_name])
                right_len = np.mean(bone_lens['Right' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])

        # Calculate the initial offset by scaling direction vectors with bone lengths.
        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]

        return initial_offset

    def get_bvh_header(self, poses_3d):
        """
        Generate the BVH header based on initial offsets and joint hierarchy.

        Args:
            poses_3d: Array of 3D joint positions.

        Returns:
            A BvhHeader object representing the BVH file header.
        """
        # Calculate initial offsets for each joint.
        initial_offset = self.get_initial_offset(poses_3d)

        # Create BvhNode for each joint in the hierarchy.
        nodes = {}
        for joint in self.keypoint2index:
            is_root = joint == self.root
            is_end_site = 'EndSite' in joint
            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',  # Define rotation order
                is_root=is_root,
                is_end_site=is_end_site,
            )
        
        # Link each joint to its children in the hierarchy.
        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        # Create and return the header containing the entire skeleton structure.
        header = bvh_helper.BvhHeader(root=nodes[self.root], nodes=nodes)
        return header

    def pose2euler(self, pose, header):
        """
        Convert a 3D pose to Euler angles based on the BVH hierarchy.

        Args:
            pose: 3D joint positions.
            header: BvhHeader representing the skeleton hierarchy.

        Returns:
            A list of Euler angles for each joint in the skeleton.
        """
        channel = []  # List to hold the channel data (Euler angles).
        quats = {}    # Dictionary to store quaternions for each joint.
        eulers = {}   # Dictionary to store Euler angles for each joint.
        stack = [header.root]  # Stack to traverse the skeleton hierarchy
        
        # Traverse each joint in the hierarchy to calculate and store Euler angles.
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]
            
            # Root joint includes positional data.
            if node.is_root:
                channel.extend(pose[joint_idx])

            # Determine local coordinate directions and rotation order for each joint type.
            index = self.keypoint2index
            order = None
            
            # Define axis directions (x_dir, y_dir, z_dir) based on joint type and hierarchy.
            # Each joint type has specific orientation based on its child and neighboring joints.
            if joint == 'Hip':
                x_dir = pose[index['LeftHip']] - pose[index['RightHip']]
                y_dir = None
                z_dir = pose[index['Spine']] - pose[joint_idx]
                order = 'zyx'
            elif joint in ['RightHip', 'RightKnee']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['Hip']] - pose[index['RightHip']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'
            elif joint in ['LeftHip', 'LeftKnee']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['LeftHip']] - pose[index['Hip']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'
            elif joint == 'Spine':
                x_dir = pose[index['LeftHip']] - pose[index['RightHip']]
                y_dir = None
                z_dir = pose[index['Thorax']] - pose[joint_idx]
                order = 'zyx'
            elif joint == 'Thorax':
                x_dir = pose[index['LeftShoulder']] - pose[index['RightShoulder']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[index['Spine']]
                order = 'zyx'
            elif joint == 'Neck':
                x_dir = None
                y_dir = pose[index['Thorax']] - pose[joint_idx]
                z_dir = pose[index['HeadEndSite']] - pose[index['Thorax']]
                order = 'zxy'
            elif joint == 'LeftShoulder':
                x_dir = pose[index['LeftElbow']] - pose[joint_idx]
                y_dir = pose[index['LeftElbow']] - pose[index['LeftWrist']]
                z_dir = None
                order = 'xzy'
            elif joint == 'LeftElbow':
                x_dir = pose[index['LeftWrist']] - pose[joint_idx]
                y_dir = pose[joint_idx] - pose[index['LeftShoulder']]
                z_dir = None
                order = 'xzy'
            elif joint == 'RightShoulder':
                x_dir = pose[joint_idx] - pose[index['RightElbow']]
                y_dir = pose[index['RightElbow']] - pose[index['RightWrist']]
                z_dir = None
                order = 'xzy'
            elif joint == 'RightElbow':
                x_dir = pose[joint_idx] - pose[index['RightWrist']]
                y_dir = pose[joint_idx] - pose[index['RightShoulder']]
                z_dir = None
                order = 'xzy'
            
            # Calculate rotation if order is defined, otherwise copy from parent.
            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()
            
            # Calculate local quaternion relative to the parent.
            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(q=quats[joint], r=quats[node.parent.name])
            
            # Convert quaternion to Euler angles based on joint rotation order.
            euler = math3d.quat2euler(q=local_quat, order=node.rotation_order)
            euler = np.rad2deg(euler)  # Convert to degrees
            eulers[joint] = euler  # Store Euler angles for each joint
            channel.extend(euler)  # Append to channel list

            # Add child nodes to stack for traversal.
            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel

    def poses2bvh(self, poses_3d, header=None, output_file=None):
        """
        Convert a sequence of 3D poses to a BVH file.

        Args:
            poses_3d: Array of 3D joint positions for each frame.
            header: BvhHeader, optional. If not provided, it will be generated.
            output_file: Output file path to save the BVH file.

        Returns:
            A tuple of (channels, header), where channels is a list of Euler angles for each frame.
        """
        # Generate header if not provided.
        if not header:
            header = self.get_bvh_header(poses_3d)

        # Generate channels for each frame by converting poses to Euler angles.
        channels = []
        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))

        # Write BVH file if output path is provided.
        if output_file:
            bvh_helper.write_bvh(output_file, header, channels)
        
        return channels, header
