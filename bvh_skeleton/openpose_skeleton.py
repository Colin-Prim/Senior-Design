class OpenPoseSkeleton(object):
    """
    Defines the skeleton structure for OpenPose keypoints.
    This class specifies the root joint, maps keypoints to indices, and sets up hierarchical relationships.
    """

    def __init__(self):
        # Define the root joint of the skeleton, used as the central reference point.
        self.root = 'MidHip'

        # Map each keypoint to an index, representing joints commonly used in OpenPose.
        # Each keypoint corresponds to a specific joint or landmark on the human body.
        self.keypoint2index = {
            'Nose': 0,
            'Neck': 1,
            'RShoulder': 2,
            'RElbow': 3,
            'RWrist': 4,
            'LShoulder': 5,
            'LElbow': 6,
            'LWrist': 7,
            'MidHip': 8,
            'RHip': 9,
            'RKnee': 10,
            'RAnkle': 11,
            'LHip': 12,
            'LKnee': 13,
            'LAnkle': 14,
            'REye': 15,
            'LEye': 16,
            'REar': 17,
            'LEar': 18,
            'LBigToe': 19,
            'LSmallToe': 20,
            'LHeel': 21,
            'RBigToe': 22,
            'RSmallToe': 23,
            'RHeel': 24
        }

        # Reverse mapping for convenience: each index points to the corresponding keypoint name.
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}

        # Count the total number of keypoints in the skeleton structure.
        self.keypoint_num = len(self.keypoint2index)

        # Define the hierarchical relationship between keypoints.
        # Each keypoint is mapped to its children, showing the skeletal structure.
        self.children = {
            'MidHip': ['Neck', 'RHip', 'LHip'],
            'Neck': ['Nose', 'RShoulder', 'LShoulder'],
            'Nose': ['REye', 'LEye'],
            'REye': ['REar'],
            'REar': [],
            'LEye': ['LEar'],
            'LEar': [],
            'RShoulder': ['RElbow'],
            'RElbow': ['RWrist'],
            'RWrist': [],
            'LShoulder': ['LElbow'],
            'LElbow': ['LWrist'],
            'LWrist': [],
            'RHip': ['RKnee'],
            'RKnee': ['RAnkle'],
            'RAnkle': ['RBigToe', 'RSmallToe', 'RHeel'],
            'RBigToe': [],
            'RSmallToe': [],
            'RHeel': [],
            'LHip': ['LKnee'],
            'LKnee': ['LAnkle'],
            'LAnkle': ['LBigToe', 'LSmallToe', 'LHeel'],
            'LBigToe': [],
            'LSmallToe': [],
            'LHeel': [],
        }

        # Create a dictionary linking each keypoint to its parent, starting from the root.
        self.parent = {self.root: None}  # The root has no parent
        for parent, children in self.children.items():
            for child in children:
                # Set each child keypoint's parent to the current parent keypoint.
                self.parent[child] = parent
