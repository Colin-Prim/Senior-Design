import copy
import numpy as np

from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton

# Define the skeleton structure for the HumanEva dataset.
# The skeleton is defined with parent-child relationships between joints,
# and specific joints are marked as left or right for augmentation purposes.
humaneva_skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
    joints_left=[2, 3, 4, 8, 9, 10],
    joints_right=[5, 6, 7, 11, 12, 13]
)

# Camera intrinsic parameters for each camera in the HumanEva dataset.
# Each dictionary entry contains the resolution, azimuth for visualization,
# and an ID for each camera.
humaneva_cameras_intrinsic_params = [
    {
        'id': 'C1',
        'res_w': 640,
        'res_h': 480,
        'azimuth': 0,
    },
    {
        'id': 'C2',
        'res_w': 640,
        'res_h': 480,
        'azimuth': -90,
    },
    {
        'id': 'C3',
        'res_w': 640,
        'res_h': 480,
        'azimuth': 90,
    },
]

# Camera extrinsic parameters for each subject in the dataset.
# Each subject is associated with a list of dictionaries, where each dictionary
# contains the orientation and translation for a particular camera view.
humaneva_cameras_extrinsic_params = {
    'S1': [
        {
            'orientation': [0.424207, -0.4983646, -0.5802981, 0.4847012],
            'translation': [4062.227, 663.2477, 1528.397],
        },
        {
            'orientation': [0.6503354, -0.7481602, -0.0919284, 0.0941766],
            'translation': [844.8131, -3805.2092, 1504.9929],
        },
        {
            'orientation': [0.0664734, -0.0690535, 0.7416416, -0.6639132],
            'translation': [-797.67377, 3916.3174, 1433.6602],
        },
    ],
    'S2': [
        {
            'orientation': [0.4214752, -0.4961493, -0.5838273, 0.4851187],
            'translation': [4112.9121, 626.4929, 1545.2988],
        },
        {
            'orientation': [0.6501393, -0.7476588, -0.0954617, 0.0959808],
            'translation': [923.5740, -3877.9243, 1504.5518],
        },
        {
            'orientation': [0.0699353, -0.0712403, 0.7421637, -0.662742],
            'translation': [-781.4915, 3838.8853, 1444.9929],
        },
    ],
    'S3': [
        {
            'orientation': [0.424207, -0.4983646, -0.5802981, 0.4847012],
            'translation': [4062.2271, 663.2477, 1528.3970],
        },
        {
            'orientation': [0.6503354, -0.7481602, -0.0919284, 0.0941766],
            'translation': [844.8131, -3805.2092, 1504.9929],
        },
        {
            'orientation': [0.0664734, -0.0690535, 0.7416416, -0.6639132],
            'translation': [-797.6738, 3916.3174, 1433.6602],
        },
    ],
    'S4': [
        {}, {}, {},
    ],
}

class HumanEvaDataset(MocapDataset):
    # Initialize the dataset with the path to data and setup camera parameters
    def __init__(self, path):
        super().__init__(fps=60, skeleton=humaneva_skeleton)

        # Copy extrinsic camera parameters to avoid mutations
        self._cameras = copy.deepcopy(humaneva_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                # Integrate each camera's extrinsic parameters with intrinsic parameters
                cam.update(humaneva_cameras_intrinsic_params[i])
                # Ensure all parameters are in float32 for computational compatibility
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')
                # Convert translation units from millimeters to meters if available
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000

        # Map each subject to various prefixed versions, used for training/validation separation
        for subject in list(self._cameras.keys()):
            data = self._cameras[subject]
            del self._cameras[subject]
            for prefix in ['Train/', 'Validate/', 'Unlabeled/Train/', 'Unlabeled/Validate/', 'Unlabeled/']:
                self._cameras[prefix + subject] = data

        # Load serialized dataset
        data = np.load(path)['positions_3d'].item()

        # Store data in a dictionary format, indexed by subject and action names
        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                # Store positions and associated camera data for each action
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras[subject],
                }
