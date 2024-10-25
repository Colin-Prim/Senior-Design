class BVHWriter:
    def __init__(self, skeleton_definition):
        self.skeleton = skeleton_definition

    def write_bvh(self, filename, keypoints_3d_list):
        try:
            with open(filename, 'w') as f:
                f.write("HIERARCHY\n")
                self._write_joint_hierarchy(f, "Hips", self.skeleton["Hips"], indent=0)

                f.write("MOTION\n")
                f.write(f"Frames: {len(keypoints_3d_list)}\n")
                f.write(f"Frame Time: 0.04\n")  # Assuming 25 FPS, adjust if needed

                for frame in keypoints_3d_list:
                    frame_data = ' '.join(map(str, frame.flatten()))
                    f.write(f"{frame_data}\n")
        except Exception as e:
            print(f"Error writing BVH file: {e}")

    def _write_joint_hierarchy(self, f, joint_name, joint_data, indent):
        indent_str = '\t' * indent
        f.write(f"{indent_str}JOINT {joint_name}\n")
        f.write(f"{indent_str}{{\n")
        offset_str = ' '.join(map(str, joint_data["offset"]))
        f.write(f"{indent_str}\tOFFSET {offset_str}\n")
        channels_str = ' '.join(joint_data["channels"])
        f.write(f"{indent_str}\tCHANNELS {len(joint_data['channels'])} {channels_str}\n")

        for child_name, child_data in joint_data["children"].items():
            self._write_joint_hierarchy(f, child_name, child_data, indent + 1)

        f.write(f"{indent_str}}}\n")
