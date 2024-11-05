# This code is adapted from a different GitHub repository.
# It has been modified to define a COCO skeleton structure with keypoints and hierarchical relationships.

class COCOSkeleton(object):
    """
    Represents a COCO skeleton structure with predefined joint hierarchies and relationships.
    This class provides methods to initialize the skeleton, set joint relationships,
    and create mappings between keypoint names and indices.
    """
    
    def __init__(self):
        # Define the root joint, which is the Neck (computed as the median of left and right shoulders).
        self.root = 'Neck'
        
        # Map each keypoint name to an index, as defined in the COCO skeleton format.
        self.keypoint2index = {
            'Nose': 0,
            'LeftEye': 1,
            'RightEye': 2,
            'LeftEar': 3,
            'RightEar': 4,
            'LeftShoulder': 5,
            'RightShoulder': 6,
            'LeftElbow': 7,
            'RightElbow': 8,
            'LeftWrist': 9,
            'RightWrist': 10,
            'LeftHip': 11,
            'RightHip': 12,
            'LeftKnee': 13,
            'RightKnee': 14,
            'LeftAnkle': 15,
            'RightAnkle': 16,
            'Neck': 17  # Index for Neck, used as the root point in this skeleton.
        }
        
        # Create a reverse mapping from indices to keypoint names.
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        
        # Total number of keypoints in the COCO skeleton.
        self.keypoint_num = len(self.keypoint2index)

        # Define the hierarchy of child joints for each keypoint.
        self.children = {
            'Neck': ['Nose', 'LeftShoulder', 'RightShoulder', 'LeftHip', 'RightHip'],
            'Nose': ['LeftEye', 'RightEye'],
            'LeftEye': ['LeftEar'],
            'LeftEar': [],
            'RightEye': ['RightEar'],
            'RightEar': [],
            'LeftShoulder': ['LeftElbow'],
            'LeftElbow': ['LeftWrist'],
            'LeftWrist': [],
            'RightShoulder': ['RightElbow'],
            'RightElbow': ['RightWrist'],
            'RightWrist': [],
            'LeftHip': ['LeftKnee'],
            'LeftKnee': ['LeftAnkle'],
            'LeftAnkle': [],
            'RightHip': ['RightKnee'],
            'RightKnee': ['RightAnkle'],
            'RightAnkle': []
        }

        # Initialize the parent mapping for each keypoint, starting with the root.
        self.parent = {self.root: None}  # Root has no parent.
        for parent, children in self.children.items():
            for child in children:
                # Set each child keypoint's parent to the current parent keypoint.
                self.parent[child] = parent
