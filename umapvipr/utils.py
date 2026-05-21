"""
Tool Function Module
"""

import json
import os
import numpy as np
import logging
import sys
from collections import defaultdict
import math

def setup_logging(log_level=logging.INFO, log_file=None):
    """Initialize the log system and ensure UTF-8 encoding output"""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    # construct logger
    logger = logging.getLogger('umapvipr')
    logger.setLevel(log_level)
    
    # delete handler
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def detect_integer_dtype(num_values):
    """Return the smallest unsigned integer dtype that can hold all category ids."""
    if num_values <= np.iinfo(np.uint8).max:
        return 'u8', np.uint8
    if num_values <= np.iinfo(np.uint16).max:
        return 'u16', np.uint16
    return 'u32', np.uint32

def save_integer_column(output_file, data, np_dtype):
    """Persist a reordered category-index column using an integer dtype."""
    np.asarray(data, dtype=np_dtype).tofile(output_file)

def compute_field_counts(field_data, unique_values):
    """Return counts aligned to the existing unique-values order."""
    value_counts = defaultdict(int)
    for value in field_data:
        value_counts[str(value)] += 1
    return [int(value_counts.get(str(value), 0)) for value in unique_values]

def normalize_field_colors(field_name, field_values, adata_uns, obs_series):
    """Map AnnData uns colors to the field_values order."""
    color_key = f"{field_name}_colors"
    if color_key not in adata_uns:
        return None

    if not hasattr(obs_series, 'cat') or not hasattr(obs_series.cat, 'categories'):
        return None

    raw_colors = adata_uns[color_key]
    cat_order = list(obs_series.cat.categories.astype(str))
    cat_to_color = {}
    for i, cat in enumerate(cat_order):
        if i < len(raw_colors):
            cat_to_color[cat] = str(raw_colors[i])

    return [cat_to_color.get(str(value)) for value in field_values]

def build_class_tree_catalog(l1_values, l2_values, l1_counts, l2_counts):
    """Build a simple two-level Class tree from L1/L2 labels."""
    l1_count_map = {
        str(value): int(count) for value, count in zip(l1_values, l1_counts)
    }
    tree = []
    used_l2 = set()

    for parent_name in l1_values:
        parent_name = str(parent_name)
        prefix = f"{parent_name}-"
        children = []
        for index, child_name in enumerate(l2_values):
            child_name = str(child_name)
            if child_name == parent_name or child_name.startswith(prefix):
                children.append({
                    'name': child_name,
                    'count': int(l2_counts[index]),
                    'valueIndex': index
                })
                used_l2.add(index)

        tree.append({
            'name': parent_name,
            'count': l1_count_map.get(parent_name, sum(child['count'] for child in children)),
            'children': children
        })

    for index, child_name in enumerate(l2_values):
        if index in used_l2:
            continue
        tree.append({
            'name': str(child_name),
            'count': int(l2_counts[index]),
            'children': [{
                'name': str(child_name),
                'count': int(l2_counts[index]),
                'valueIndex': index
            }]
        })

    return tree

def write_manifest_v2(output_dir, manifest):
    """write manifest.json"""
    with open(os.path.join(output_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def write_fields_summary_v2(output_dir, field_summaries):
    """write fields_summary.json"""
    with open(os.path.join(output_dir, 'fields_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'fields': field_summaries}, f, ensure_ascii=False, indent=2)

def write_field_catalog_v2(output_dir, field_name, catalog):
    catalog_dir = os.path.join(output_dir, 'field_catalog')
    os.makedirs(catalog_dir, exist_ok=True)
    with open(os.path.join(catalog_dir, f"{field_name}.json"), 'w', encoding='utf-8') as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

def calculate_genes_per_chunk(total_cells, max_size_mb=2):
    """
    Calculate the appropriate size of gene blocks
    Based on each cell occupying 4 bytes (float32), ensure that the size of a single chunk file does not exceed max_size_mb
    """
    if total_cells <= 0:
        return 100

    bytes_per_gene = total_cells * 4
    max_bytes = max_size_mb * 1024 * 1024
    genes_per_chunk = max(1, math.floor(max_bytes / bytes_per_gene))
    genes_per_chunk = min(1000, max(10, genes_per_chunk))
    genes_per_chunk = max(10, int(round(genes_per_chunk / 10.0)) * 10)

    return genes_per_chunk
