class BVHWriter:
    def __init__(self, skeleton_definition):
        self.skeleton = skeleton_definition

    def write_bvh(self, file_path, motion_data):
        with open(file_path, 'w') as f:
            f.write("HIERARCHY\n")
            self._write_joint_hierarchy(f, "ROOT", "Hips", self.skeleton["Hips"])
            f.write("\nMOTION\n")
            f.write(f"Frames: {len(motion_data)}\n")
            f.write("Frame Time: 0.04\n")
            for frame in motion_data:
                f.write(" ".join(map(str, frame)) + "\n")

    def _write_joint_hierarchy(self, f, joint_type, joint_name, joint_data, indent=0):
        indent_str = '\t' * indent
        f.write(f"{indent_str}{joint_type} {joint_name}\n")
        f.write(f"{indent_str}{{\n")
        f.write(f"{indent_str}\tOFFSET {' '.join(map(str, joint_data['offset']))}\n")
        f.write(f"{indent_str}\tCHANNELS {len(joint_data['channels'])} {' '.join(joint_data['channels'])}\n")
        for child_name, child_data in joint_data["children"].items():
            self._write_joint_hierarchy(f, "JOINT", child_name, child_data, indent + 1)
        f.write(f"{indent_str}}}\n")
