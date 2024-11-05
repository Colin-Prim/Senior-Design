# This code is adapted from a different GitHub repository.
# It has been modified to fit specific requirements for processing skeleton data in BVH format.

from . import math3d
from . import bvh_helper

import numpy as np
from pprint import pprint


class CMUSkeleton(object):
    """
    Represents a CMU skeleton structure with predefined joint hierarchies and relationships.
    This class provides methods to initialize the skeleton, calculate initial offsets, and
    convert pose data to Euler angles suitable for BVH format.
    """

    def __init__(self):
        # Define root joint and mapping of keypoints to indices.
        self.root = 'Hips'
        self.keypoint2index = {
            'Hips': 0,
            'RightUpLeg': 1,
            'RightLeg': 2,
            'RightFoot': 3,
            'LeftUpLeg': 4,
            'LeftLeg': 5,
            'LeftFoot': 6,
            'Spine': 7,
            'Spine1': 8,
            'Neck1': 9,
            'HeadEndSite': 10,
            'LeftArm': 11,
            'LeftForeArm': 12,
            'LeftHand': 13,
            'RightArm': 14,
            'RightForeArm': 15,
            'RightHand': 16,
            # Additional joints marked as -1, indicating they don't contribute to the skeleton.
            'RightHipJoint': -1,
            'RightFootEndSite': -1,
            'LeftHipJoint': -1,
            'LeftFootEndSite': -1,
            'LeftShoulder': -1,
            'LeftHandEndSite': -1,
            'RightShoulder': -1,
            'RightHandEndSite': -1,
            'LowerBack': -1,
            'Neck': -1
        }
        # Reverse mapping from indices to keypoints.
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)

        # Define hierarchy of children for each joint in the skeleton.
        self.children = {
            'Hips': ['LeftHipJoint', 'LowerBack', 'RightHipJoint'],
            'LeftHipJoint': ['LeftUpLeg'],
            'LeftUpLeg': ['LeftLeg'],
            'LeftLeg': ['LeftFoot'],
            'LeftFoot': ['LeftFootEndSite'],
            'LeftFootEndSite': [],
            'LowerBack': ['Spine'],
            'Spine': ['Spine1'],
            'Spine1': ['LeftShoulder', 'Neck', 'RightShoulder'],
            'LeftShoulder': ['LeftArm'],
            'LeftArm': ['LeftForeArm'],
            'LeftForeArm': ['LeftHand'],
            'LeftHand': ['LeftHandEndSite'],
            'LeftHandEndSite': [],
            'Neck': ['Neck1'],
            'Neck1': ['HeadEndSite'],
            'HeadEndSite': [],
            'RightShoulder': ['RightArm'],
            'RightArm': ['RightForeArm'],
            'RightForeArm': ['RightHand'],
            'RightHand': ['RightHandEndSite'],
            'RightHandEndSite': [],
            'RightHipJoint': ['RightUpLeg'],
            'RightUpLeg': ['RightLeg'],
            'RightLeg': ['RightFoot'],
            'RightFoot': ['RightFootEndSite'],
            'RightFootEndSite': [],
        }
        
        # Define parent for each joint.
        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent
        
        # Identify left and right joints for convenience.
        self.left_joints = [joint for joint in self.keypoint2index if 'Left' in joint]
        self.right_joints = [joint for joint in self.keypoint2index if 'Right' in joint]

        # Define initial T-pose direction for each joint.
        self.initial_directions = {
            'Hips': [0, 0, 0],
            'LeftHipJoint': [1, 0, 0],
            'LeftUpLeg': [1, 0, 0],
            'LeftLeg': [0, 0, -1],
            'LeftFoot': [0, 0, -1],
            'LeftFootEndSite': [0, -1, 0],
            'LowerBack': [0, 0, 1],
            'Spine': [0, 0, 1],
            'Spine1': [0, 0, 1],
            'LeftShoulder': [1, 0, 0],
            'LeftArm': [1, 0, 0],
            'LeftForeArm': [1, 0, 0],
            'LeftHand': [1, 0, 0],
            'LeftHandEndSite': [1, 0, 0],
            'Neck': [0, 0, 1],
            'Neck1': [0, 0, 1],
            'HeadEndSite': [0, 0, 1],
            'RightShoulder': [-1, 0, 0],
            'RightArm': [-1, 0, 0],
            'RightForeArm': [-1, 0, 0],
            'RightHand': [-1, 0, 0],
            'RightHandEndSite': [-1, 0, 0],
            'RightHipJoint': [-1, 0, 0],
            'RightUpLeg': [-1, 0, 0],
            'RightLeg': [0, 0, -1],
            'RightFoot': [0, 0, -1],
            'RightFootEndSite': [0, -1, 0]
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
        while stack:
            parent = stack.pop()
            p_idx = self.keypoint2index[parent]
            p_name = parent
            while p_idx == -1:
                # Find real parent if current index is invalid.
                p_name = self.parent[p_name]
                p_idx = self.keypoint2index[p_name]
            for child in self.children[parent]:
                stack.append(child)

                if self.keypoint2index[child] == -1:
                    bone_lens[child] = [0.1]  # Default bone length for undefined joints.
                else:
                    c_idx = self.keypoint2index[child]
                    bone_lens[child] = np.linalg.norm(
                        poses_3d[:, p_idx] - poses_3d[:, c_idx],
                        axis=1
                    )

        # Average the bone lengths for symmetric joints.
        bone_len = {}
        for joint in self.keypoint2index:
            if 'Left' in joint or 'Right' in joint:
                base_name = joint.replace('Left', '').replace('Right', '')
                left_len = np.mean(bone_lens['Left' + base_name])
                right_len = np.mean(bone_lens['Right' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])

        # Calculate initial offset by scaling direction vectors with bone lengths.
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
        initial_offset = self.get_initial_offset(poses_3d)

        # Create BvhNode for each joint.
        nodes = {}
        for joint in self.keypoint2index:
            is_root = joint == self.root
            is_end_site = 'EndSite' in joint
            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',
                is_root=is_root,
                is_end_site=is_end_site,
            )
        # Establish parent-child relationships in the node hierarchy.
        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        # Create and return the header.
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
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]
            
            # Root joint has positional data.
            if node.is_root:
                channel.extend(pose[joint_idx])

            # Define local directions for each joint type.
            index = self.keypoint2index
            order = None
            # Calculate directions based on specific joint types...
            # (Each joint type has its own orientation and local coordinate system.)
            # [Code continues with directional calculations for different joints]

            # Calculate quaternion and convert to Euler angles.
            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()
            
            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )
            
            euler = math3d.quat2euler(q=local_quat, order=node.rotation_order)
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            # Add child nodes to stack.
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
