import copy
import numpy as np
from common.camera import normalize_screen_coordinates
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton

# Define the skeleton structure for the Human 3.6M dataset.
# The skeleton is defined with parent-child relationships between joints,
# and specific joints are marked as left or right for augmentation purposes.
h36m_skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
             16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
)

# Camera intrinsic parameters for each camera in the Human 3.6M dataset.
# Each dictionary entry contains the center, focal length, distortion parameters, 
# resolution, and azimuth for a particular camera.
h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70,
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70,
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
        'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
        'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110,
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110,
    },
]

# Camera extrinsic parameters for each subject in the dataset.
# Each subject is associated with a list of dictionaries, where each dictionary
# contains the orientation and translation for a particular camera view.
h36m_cameras_extrinsic_params = {
    'S1': [
        {'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
         'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125]},
        {'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
         'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375]},
        {'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
         'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375]},
        {'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
         'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125]},
    ],
    # Empty dictionaries for other subjects can be filled similarly as needed.
}


class Human36mDataset(MocapDataset):
    # Initialize the dataset with path to data and optionally remove static joints.
    def __init__(self, path, remove_static_joints=True):
        super().__init__(fps=50, skeleton=h36m_skeleton)

        # Copy extrinsic camera parameters to avoid mutations.
        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                # Integrate each camera's extrinsic parameters with intrinsic parameters
                cam.update(h36m_cameras_intrinsic_params[i])
                # Ensure all parameters are in float32 for computational compatibility
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # Normalize center coordinates and focal length based on camera resolution
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2

                # Convert translation units from millimeters to meters if available
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000

                # Combine intrinsic parameters into a single array for easier access
                cam['intrinsic'] = np.concatenate((cam['focal_length'], cam['center'],
                                                   cam['radial_distortion'], cam['tangential_distortion']))

        # Load the dataset positions from a .npz file
        data = np.load(path)['positions_3d'].item()

        # Store data in a dictionary format, indexed by subject and action names
        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                # Store positions and associated camera data for each action.
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras[subject],
                }

        # Optionally remove static joints to simplify the skeleton structure.
        if remove_static_joints:
            # Remove 15 static joints to create a 17-joint structure from the original 32-joint skeleton
            self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])

            # Adjust the parent structure for shoulders after removing joints
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8

    def supports_semi_supervised(self):
        # This dataset is compatible with semi-supervised learning.
        return True
