"""
Sparse Gene Data Processing Module
"""

import struct
import numpy as np
import os
from collections import defaultdict

class SparseGeneData:
    @staticmethod
    def create_sparse_gene_chunk(gene_expressions, coordinate_order, threshold=0.01):
        """
        input: gene_expressions [num_genes, num_cells], coordinate_order [total_cells]
        output: dict { gene_idx: { 'values': float32[], 'indices': uint32[] } }
        """
        num_genes, num_cells = gene_expressions.shape
        sparse_data = {}

        original_to_quad_pos = np.full(num_cells, -1, dtype=np.int32)
        coordinate_order_array = np.array(coordinate_order, dtype=np.int32)
        valid_mask = coordinate_order_array < num_cells
        original_to_quad_pos[coordinate_order_array[valid_mask]] = np.arange(len(coordinate_order))[valid_mask]


        for gene_idx in range(num_genes):
            gene_values = gene_expressions[gene_idx, :]


            non_zero_mask = gene_values >= threshold
            non_zero_original_indices = np.where(non_zero_mask)[0]
            non_zero_values = gene_values[non_zero_mask]


            quad_positions = original_to_quad_pos[non_zero_original_indices]
            valid_positions_mask = quad_positions >= 0

            quad_tree_indices = quad_positions[valid_positions_mask]
            filtered_values = non_zero_values[valid_positions_mask]

            if len(quad_tree_indices) > 0:
                sort_indices = np.argsort(quad_tree_indices)
                quad_tree_indices = quad_tree_indices[sort_indices]
                filtered_values = filtered_values[sort_indices]

            sparse_data[gene_idx] = {
                'values': filtered_values.astype(np.float32),
                'indices': quad_tree_indices.astype(np.uint32)
            }

        return sparse_data

    @staticmethod
    def save_sparse_chunk(sparse_data, output_file):
        """
        Save sparse data blocks to binary file
        Binary format: [num_genes:u32] [non_zero_count:u32 [value:f32 index:u32]*]*
        """
        num_genes = len(sparse_data)

        with open(output_file, 'wb') as f:
            f.write(struct.pack('I', num_genes))

            for gene_idx in range(num_genes):
                gene_data = sparse_data.get(gene_idx, {
                    'values': np.array([], dtype=np.float32),
                    'indices': np.array([], dtype=np.uint32)
                })

                values = gene_data['values']
                indices = gene_data['indices']
                non_zero_count = len(values)
                f.write(struct.pack('I', non_zero_count))

                if non_zero_count > 0:
                    interleaved = np.empty(non_zero_count * 2, dtype=np.float32)
                    interleaved[0::2] = values
                    interleaved[1::2] = indices.view(np.float32)
                    interleaved.tofile(f)

    @staticmethod
    def get_compression_stats(original_size, sparse_data):
        sparse_size = 4
        total_non_zeros = 0

        for gene_data in sparse_data.values():
                non_zero_count = len(gene_data['values'])
                total_non_zeros += non_zero_count
                sparse_size += 4
                sparse_size += non_zero_count * 8

        compression_ratio = original_size / sparse_size if sparse_size > 0 else 0

        return {
            'original_size_bytes': original_size,
            'sparse_size_bytes': sparse_size,
            'compression_ratio': compression_ratio,
            'total_non_zeros': total_non_zeros,
            'space_saved_percent': ((original_size - sparse_size) / original_size * 100) if original_size > 0 else 0
        }
