"""
Quadtree implementation module
"""

import numpy as np
import threading
import os

class FastQuadTreeNode:
    _id_counter = 0
    _lock = threading.Lock()

    def __init__(self, boundary, depth, max_points=60000):
        with FastQuadTreeNode._lock:
            FastQuadTreeNode._id_counter += 1
            self.id = f"block_{FastQuadTreeNode._id_counter}"

        self.boundary = boundary
        self.depth = depth
        self.max_points = max_points
        self.indices = []
        self.divided = False
        self.children = None
        self.center = [
            (boundary[0] + boundary[2]) / 2,
            (boundary[1] + boundary[3]) / 2
        ]

    def insert_indices_batch(self, indices, coordinates):
        if len(indices) == 0:
            return

        coords = coordinates[indices]
        in_bounds = (
            (coords[:, 0] >= self.boundary[0] - 1e-6) &
            (coords[:, 0] <= self.boundary[2] + 1e-6) &
            (coords[:, 1] >= self.boundary[1] - 1e-6) &
            (coords[:, 1] <= self.boundary[3] + 1e-6)
        )
        valid_indices = indices[in_bounds]

        if len(valid_indices) == 0:
            return

        if len(valid_indices) <= self.max_points:
            self.indices = valid_indices.tolist()
            return

        if not self.divided:
            self.divide()

        valid_coords = coordinates[valid_indices]
        in_right = valid_coords[:, 0] > self.center[0]
        in_top = valid_coords[:, 1] > self.center[1]

        nw_mask = ~in_right & in_top
        ne_mask = in_right & in_top
        sw_mask = ~in_right & ~in_top
        se_mask = in_right & ~in_top

        if np.any(nw_mask):
            self.children['nw'].insert_indices_batch(valid_indices[nw_mask], coordinates)
        if np.any(ne_mask):
            self.children['ne'].insert_indices_batch(valid_indices[ne_mask], coordinates)
        if np.any(sw_mask):
            self.children['sw'].insert_indices_batch(valid_indices[sw_mask], coordinates)
        if np.any(se_mask):
            self.children['se'].insert_indices_batch(valid_indices[se_mask], coordinates)

    def divide(self):
        min_x, min_y, max_x, max_y = self.boundary
        mid_x, mid_y = self.center

        self.children = {
            'nw': FastQuadTreeNode([min_x, mid_y, mid_x, max_y], self.depth + 1, self.max_points),
            'ne': FastQuadTreeNode([mid_x, mid_y, max_x, max_y], self.depth + 1, self.max_points),
            'sw': FastQuadTreeNode([min_x, min_y, mid_x, mid_y], self.depth + 1, self.max_points),
            'se': FastQuadTreeNode([mid_x, min_y, max_x, mid_y], self.depth + 1, self.max_points)
        }
        self.divided = True

    def get_load_order(self):
        """
        Obtain the loading sequence for the quadtree traversal - Use breadth-first (BFS) traversal
        """
        order = []
        queue = [self]
        
        while queue:
            node = queue.pop(0)
            
            if node.indices:
                order.extend(sorted(node.indices))

            if node.divided:
                for direction in ['nw', 'ne', 'sw', 'se']:
                    queue.append(node.children[direction])
        
        return order

    def save_node_info(self, blocks_dir, coordinates):
        has_file = False
        if self.indices:
            os.makedirs(blocks_dir, exist_ok=True)
            block_file = os.path.join(blocks_dir, f"{self.id}.f32.bin")
            sorted_indices = sorted(self.indices)
            coords = coordinates[sorted_indices].flatten().astype(np.float32)
            coords.tofile(block_file)
            has_file = True

        info = {
            'id': self.id,
            'boundary': self.boundary,
            'points_count': len(self.indices),
            'has_file': has_file,
            'file': os.path.join('blocks', f"{self.id}.f32.bin").replace('\\', '/') if has_file else None
        }

        if self.divided:
            info['children'] = {}
            for direction, child in self.children.items():
                info['children'][direction] = child.save_node_info(blocks_dir, coordinates)

        return info
