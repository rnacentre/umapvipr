"""
H5AD Umap Visual Processor
"""

from .config import ProcessingConfig
from .processor import process_h5ad_ultra_fast, FastDataProcessor
from .sparse_gene import SparseGeneData
from .quadtree import FastQuadTreeNode
from .utils import (
    setup_logging,
    detect_integer_dtype,
    save_integer_column,
    compute_field_counts,
    normalize_field_colors,
    build_class_tree_catalog,
    write_manifest_v2,
    write_fields_summary_v2,
    write_field_catalog_v2
)

__version__ = "1.0.0"
__all__ = [
    'ProcessingConfig',
    'process_h5ad_ultra_fast',
    'FastDataProcessor',
    'SparseGeneData',
    'FastQuadTreeNode',
    'setup_logging',
    'detect_integer_dtype',
    'save_integer_column',
    'compute_field_counts',
    'normalize_field_colors',
    'build_class_tree_catalog',
    'write_manifest_v2',
    'write_fields_summary_v2',
    'write_field_catalog_v2'
]