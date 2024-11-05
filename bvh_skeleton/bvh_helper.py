import os
from pathlib import Path
from typing import List, Optional, Tuple


class BvhNode:
    """
    Represents a node in the BVH skeleton hierarchy, which can be a root, joint, or end site.
    """
    def __init__(self, name: str, offset: Tuple[float, float, float],
                 rotation_order: str, children: Optional[List['BvhNode']] = None,
                 parent: Optional['BvhNode'] = None, is_root: bool = False,
                 is_end_site: bool = False):
        # Validate rotation order if this node is not an end site.
        if not is_end_site and rotation_order not in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']:
            raise ValueError(f'Rotation order invalid.')
        
        # Initialize node properties.
        self.name = name  # Name of the node (e.g., "Hips" or "LeftArm").
        self.offset = offset  # Offset values for the node in 3D space (x, y, z).
        self.rotation_order = rotation_order  # Order of rotations (e.g., "xyz").
        self.children = children if children is not None else []  # List of child nodes.
        self.parent = parent  # Reference to the parent node.
        self.is_root = is_root  # Indicates if the node is the root.
        self.is_end_site = is_end_site  # Indicates if the node is an end site with no channels.


class BvhHeader:
    """
    Represents the BVH file's header structure, containing the root node and all nodes in the hierarchy.
    """
    def __init__(self, root: BvhNode, nodes: List[BvhNode]):
        self.root = root  # Root node of the BVH skeleton.
        self.nodes = nodes  # List of all nodes (root and children) in the hierarchy.


def format_offset(offset: Tuple[float, float, float]) -> str:
    """
    Formats the offset tuple into a space-separated string.

    Args:
        offset: A tuple containing (x, y, z) offset values.

    Returns:
        A string with space-separated offset values.
    """
    return f"{offset[0]} {offset[1]} {offset[2]}"


def write_header(writer, node: BvhNode, level: int):
    """
    Recursively writes the hierarchical structure of nodes in the BVH file.

    Args:
        writer: The file writer object.
        node: The current BvhNode being written.
        level: The depth level of the node in the hierarchy for indentation.
    """
    indent = ' ' * 4 * level
    
    # Write the node type and name (ROOT, JOINT, or End Site) based on its role in the hierarchy.
    if node.is_root:
        writer.write(f'{indent}ROOT {node.name}\n')
        channel_num = 6  # Root node includes 3 position and 3 rotation channels.
    elif node.is_end_site:
        writer.write(f'{indent}End Site\n')
        channel_num = 0  # End site has no channels.
    else:
        writer.write(f'{indent}JOINT {node.name}\n')
        channel_num = 3  # Joint node includes only rotation channels.
    
    writer.write(f'{indent}{"{"}\n')

    # Write the OFFSET for the node, representing its position relative to the parent.
    indent = ' ' * 4 * (level + 1)
    writer.write(f'{indent}OFFSET {format_offset(node.offset)}\n')
    
    # Write CHANNELS if the node has channels (position and/or rotation).
    if channel_num:
        channel_line = f'{indent}CHANNELS {channel_num} '
        
        # Root node includes position channels for X, Y, and Z axes.
        if node.is_root:
            channel_line += 'Xposition Yposition Zposition '
        
        # Add rotation channels based on the node's rotation order.
        channel_line += ' '.join([
            f'{axis.upper()}rotation'
            for axis in node.rotation_order
        ])
        writer.write(channel_line + '\n')
    
    # Recursively write each child node of the current node.
    for child in node.children:
        write_header(writer, child, level + 1)
    
    # Close the block for this node.
    indent = ' ' * 4 * level
    writer.write(f'{indent}{"}"}\n')


def write_bvh(output_file: str, header: BvhHeader, channels: List[List[float]], frame_rate: int = 30):
    """
    Writes the complete BVH file, including both the header (hierarchy) and motion data (frames).

    Args:
        output_file: Path to the output BVH file.
        header: BvhHeader containing the root node and all nodes.
        channels: List of motion data for each frame.
        frame_rate: Frame rate for the animation, default is 30 FPS.
    """
    # Validate the header and channels input to ensure they are correctly structured.
    if not isinstance(header.root, BvhNode):
        raise ValueError("Header root must be a BvhNode instance.")
    if not all(isinstance(channel, (list, tuple)) for channel in channels):
        raise ValueError("Channels must be a list of lists or tuples.")

    # Convert output_file to a Path object for easier manipulation.
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists.

    try:
        with output_file.open('w') as f:
            # Write the HIERARCHY section.
            f.write('HIERARCHY\n')
            write_header(writer=f, node=header.root, level=0)
            
            # Write the MOTION section.
            f.write('MOTION\n')
            f.write(f'Frames: {len(channels)}\n')  # Total number of frames.
            f.write(f'Frame Time: {1 / frame_rate}\n')  # Duration of each frame based on frame rate.

            # Write each frame's data, with each channel's data separated by spaces.
            for channel in channels:
                f.write(' '.join([f'{element}' for element in channel]) + '\n')
    except IOError as e:
        raise IOError(f"Failed to write to file {output_file}: {e}")
