import numpy as np

class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        """
        Initialize the Skeleton object with given joint hierarchy and joint sides.

        Arguments:
        parents -- list of parent indices for each joint (-1 indicates root)
        joints_left -- list of indices for left-side joints
        joints_right -- list of indices for right-side joints
        """
        assert len(joints_left) == len(joints_right), "Left and right joints should have equal length."

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()  # Initialize metadata for child relations

    def num_joints(self):
        """Returns the number of joints in the skeleton."""
        return len(self._parents)

    def parents(self):
        """Returns the array of parent indices for each joint."""
        return self._parents

    def has_children(self):
        """Returns a boolean array indicating if each joint has children."""
        return self._has_children

    def children(self):
        """Returns a list of lists where each list contains the children of each joint."""
        return self._children

    def remove_joints(self, joints_to_remove):
        """
        Remove specified joints from the skeleton.

        Arguments:
        joints_to_remove -- list of joint indices to remove

        Returns:
        A list of indices of joints that were retained.
        """
        valid_joints = [joint for joint in range(len(self._parents)) if joint not in joints_to_remove]

        # Adjust parent indices for joints that are not removed
        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        # Calculate index offsets and update parents to reflect the removal
        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        # Update left and right joint lists to remove invalid joints
        self._joints_left = [joint - index_offsets[joint] for joint in self._joints_left if joint in valid_joints]
        self._joints_right = [joint - index_offsets[joint] for joint in self._joints_right if joint in valid_joints]

        self._compute_metadata()  # Recompute metadata after removal

        return valid_joints

    def joints_left(self):
        """Returns the list of left-side joint indices."""
        return self._joints_left

    def joints_right(self):
        """Returns the list of right-side joint indices."""
        return self._joints_right

    def _compute_metadata(self):
        """Compute metadata for child relationships and joint hierarchy."""
        # Initialize child relations and parent check
        self._has_children = np.zeros(len(self._parents), dtype=bool)
        self._children = [[] for _ in range(len(self._parents))]
        
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True
                self._children[parent].append(i)
