from pycocotools.coco import COCO
import h5py

# Load COCO annotations
coco = COCO('coco_dataset/annotations/instances_train2017.json')

# Create HDF5 file
with h5py.File('annot_coco.h5', 'w') as h5file:
    # Assuming you want to save keypoints
    keypoints = [ann['keypoints'] for ann in coco.loadAnns(coco.getAnnIds()) if 'keypoints' in ann]
    h5file.create_dataset('part', data=keypoints)



#currently using this to run the training models, the "Main" is in videopose.py for now