import os
from pathlib import Path


class BvhNode(object):
    """
    Represents a node in the BVH skeleton hierarchy, which can be a root, joint, or end site.
    """
    def __init__(
            self, name, offset, rotation_order,
            children=None, parent=None, is_root=False, is_end_site=False):
        if not is_end_site and \
                rotation_order not in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']:
            raise ValueError(f'Rotation order invalid.')
        
        self.name = name  # Name of the node (e.g., joint or root).
        self.offset = offset  # Offset values for the node in 3D space.
        self.rotation_order = rotation_order  # Order of rotations for the node.
        self.children = children if children else []  # List of child nodes.
        self.parent = parent  # Reference to the parent node.
        self.is_root = is_root  # Boolean indicating if this node is the root.
        self.is_end_site = is_end_site  # Boolean indicating if this node is an end site.


class BvhHeader(object):
    """
    Represents the BVH file's header structure, containing the root node and all nodes in the hierarchy.
    """
    def __init__(self, root, nodes):
        self.root = root  # Root node of the BVH skeleton.
        self.nodes = nodes  # List of all nodes in the hierarchy.


def write_header(writer, node, level):
    """
    Recursively writes the hierarchical structure of the skeleton to the BVH file.

    Args:
        writer: The file writer object.
        node: The current BvhNode being written.
        level: The depth level of the node in the hierarchy for indentation.
    """
    indent = ' ' * 4 * level

    # Write node type (ROOT, JOINT, or End Site) based on its position in hierarchy.
    if node.is_root:
        writer.write(f'{indent}ROOT {node.name}\n')
        channel_num = 6  # Root node includes position and rotation channels.
    elif node.is_end_site:
        writer.write(f'{indent}End Site\n')
        channel_num = 0  # End site has no channels.
    else:
        writer.write(f'{indent}JOINT {node.name}\n')
        channel_num = 3  # Joint node includes rotation channels only.
    
    writer.write(f'{indent}{"{"}\n')
    
    # Write the offset for the node.
    indent = ' ' * 4 * (level + 1)
    writer.write(
        f'{indent}OFFSET '
        f'{node.offset[0]} {node.offset[1]} {node.offset[2]}\n'
    )
    
    # Write the channel information if applicable.
    if channel_num:
        channel_line = f'{indent}CHANNELS {channel_num} '
        if node.is_root:
            channel_line += f'Xposition Yposition Zposition '
        channel_line += ' '.join([
            f'{axis.upper()}rotation'
            for axis in node.rotation_order
        ])
        writer.write(channel_line + '\n')

    # Recursively write child nodes.
    for child in node.children:
        write_header(writer, child, level + 1)

    indent = ' ' * 4 * level
    writer.write(f'{indent}{"}"}\n')


def write_bvh(output_file, header, channels, frame_rate=30):
    """
    Writes the complete BVH file, including both the header and motion data.

    Args:
        output_file: Path to the output BVH file.
        header: BvhHeader containing the root node and all nodes.
        channels: List of motion data for each frame.
        frame_rate: Frame rate for the animation, default is 30 FPS.
    """
    if not isinstance(header, BvhHeader):
        raise ValueError("Invalid header object.")
    if not all(isinstance(channel, (list, tuple)) for channel in channels):
        raise ValueError("Channels should be a list of lists or tuples.")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists.

    try:
        with output_file.open('w') as f:
            # Write the HIERARCHY section.
            f.write('HIERARCHY\n')
            write_header(writer=f, node=header.root, level=0)

            # Write the MOTION section.
            f.write('MOTION\n')
            f.write(f'Frames: {len(channels)}\n')
            f.write(f'Frame Time: {1 / frame_rate}\n')

            # Write each frame's data.
            for channel in channels:
                f.write(' '.join([f'{element}' for element in channel]) + '\n')
    except IOError as e:
        raise IOError(f"Failed to write to file {output_file}: {e}")
