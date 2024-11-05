# This code is adapted from a different GitHub repository.
# It has been modified to define an H36M original skeleton structure with keypoints and hierarchical relationships.

class H36mOriginalSkeleton(object):
    """
    Represents the H36M (Human3.6M) original skeleton structure with predefined joint hierarchies and relationships.
    This class provides methods to initialize the skeleton, set joint relationships,
    and create mappings between keypoint names and indices.
    """

    def __init__(self):
        # Define the root joint, which is the Hip.
        self.root = 'Hip'
        
        # Map each keypoint name to an index, as defined in the H36M skeleton format.
        self.keypoint2index = {
            'Hip': 0,
            'RightUpLeg': 1,
            'RightLeg': 2,
            'RightFoot': 3,
            'RightToeBase': 4,
            'RightToeBaseEndSite': 5,
            'LeftUpLeg': 6,
            'LeftLeg': 7,
            'LeftFoot': 8,
            'LeftToeBase': 9,
            'LeftToeBaseEndSite': 10,
            'Spine': 11,
            'Spine1': 12,
            'Neck': 13,
            'Head': 14,
            'HeadEndSite': 15,
            'LeftShoulder': 16,
            'LeftArm': 17,
            'LeftForeArm': 18,
            'LeftHand': 19,
            'LeftHandThumb': 20,
            'LeftHandThumbEndSite': 21,
            'LeftWristEnd': 22,
            'LeftWristEndEndSite': 23,
            'RightShoulder': 24,
            'RightArm': 25,
            'RightForeArm': 26,
            'RightHand': 27,
            'RightHandThumb': 28,
            'RightHandThumbEndSite': 29,
            'RightWristEnd': 30,
            'RightWristEndEndSite': 31
        }
        
        # Create a reverse mapping from indices to keypoint names.
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        
        # Total number of keypoints in the H36M skeleton.
        self.keypoint_num = len(self.keypoint2index)

        # Define the hierarchy of child joints for each keypoint.
        self.children = {
            'Hip': ['RightUpLeg', 'LeftUpLeg', 'Spine'],
            'RightUpLeg': ['RightLeg'],
            'RightLeg': ['RightFoot'],
            'RightFoot': ['RightToeBase'],
            'RightToeBase': ['RightToeBaseEndSite'],
            'RightToeBaseEndSite': [],
            'LeftUpLeg': ['LeftLeg'],
            'LeftLeg': ['LeftFoot'],
            'LeftFoot': ['LeftToeBase'],
            'LeftToeBase': ['LeftToeBaseEndSite'],
            'LeftToeBaseEndSite': [],
            'Spine': ['Spine1'],
            'Spine1': ['Neck', 'LeftShoulder', 'RightShoulder'],
            'Neck': ['Head'],
            'Head': ['HeadEndSite'],
            'HeadEndSite': [],
            'LeftShoulder': ['LeftArm'],
            'LeftArm': ['LeftForeArm'],
            'LeftForeArm': ['LeftHand'],
            'LeftHand': ['LeftHandThumb', 'LeftWristEnd'],
            'LeftHandThumb': ['LeftHandThumbEndSite'],
            'LeftHandThumbEndSite': [],
            'LeftWristEnd': ['LeftWristEndEndSite'],
            'LeftWristEndEndSite': [],
            'RightShoulder': ['RightArm'],
            'RightArm': ['RightForeArm'],
            'RightForeArm': ['RightHand'],
            'RightHand': ['RightHandThumb', 'RightWristEnd'],
            'RightHandThumb': ['RightHandThumbEndSite'],
            'RightHandThumbEndSite': [],
            'RightWristEnd': ['RightWristEndEndSite'],
            'RightWristEndEndSite': [],
        }

        # Initialize the parent mapping for each keypoint, starting with the root.
        self.parent = {self.root: None}  # Root has no parent.
        for parent, children in self.children.items():
            for child in children:
                # Set each child keypoint's parent to the current parent keypoint.
                self.parent[child] = parent
        
        # Identify left and right joints for convenience.
        self.left_joints = [joint for joint in self.keypoint2index if 'Left' in joint]
        self.right_joints = [joint for joint in self.keypoint2index if 'Right' in joint]
