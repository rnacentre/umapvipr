"""
Core processor module
"""

import os
import json
import numpy as np
import pandas as pd
import time
import gc
import h5py
import scanpy as sc
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

from .config import ProcessingConfig
from .quadtree import FastQuadTreeNode
from .sparse_gene import SparseGeneData
from .utils import (
    detect_integer_dtype,
    save_integer_column,
    compute_field_counts,
    normalize_field_colors,
    build_class_tree_catalog,
    write_manifest_v2,
    write_fields_summary_v2,
    write_field_catalog_v2,
    calculate_genes_per_chunk
)

class FastDataProcessor:

    def __init__(self, adata, config=None):
        self.adata = adata
        self.config = config or ProcessingConfig()
        self.num_cells = adata.shape[0]
        self.num_genes = adata.shape[1]

        # get logger
        import logging
        self.logger = logging.getLogger('h5ad_processor')

        self.logger.info(f"Pre-allocated memory: {self.num_cells:,} cells, {self.num_genes:,} genes")

        # coordinates
        self.coordinates = adata.obsm['X_umap'].astype(np.float32)
        self.coordinates = np.round(self.coordinates, self.config.coordinate_precision)

        # Attribute data
        self.obs_fields = list(adata.obs.columns)
        self.obs_data = {}

        self.logger.info("Extract attribute data (vectorization)...")
        for field in self.obs_fields:
            self.obs_data[field] = adata.obs[field].astype(str).values

        # 计算边界
        self.logger.info("Calculate the coordinate boundaries...")
        self.bounds = [
            float(self.coordinates[:, 0].min()),
            float(self.coordinates[:, 1].min()),
            float(self.coordinates[:, 0].max()),
            float(self.coordinates[:, 1].max())
        ]
        self.logger.info(f"Boundary: {self.bounds}")

        # 统计字段唯一值
        self.logger.info("Unique values of statistical fields (in original order)...")
        self.field_values = {}
        self.field_counts = {}
        self.field_colors = {}
        self.field_storage_meta = {}
        for field in self.obs_fields:
            self.field_values[field] = list(pd.unique(self.obs_data[field]))
            self.field_counts[field] = compute_field_counts(
                self.obs_data[field], self.field_values[field]
            )
            self.field_colors[field] = normalize_field_colors(
                field, self.field_values[field], self.adata.uns, self.adata.obs[field]
            )
            self.logger.info(f"  Field '{field}': {len(self.field_values[field])} Unique values")

        gc.collect()

    def build_quadtree_fast(self):
        self.logger.info("Start the ultra-fast quadtree construction...")
        start_time = time.time()

        root = FastQuadTreeNode(self.bounds, 0, self.config.max_points_per_node)
        indices = np.arange(self.num_cells, dtype=np.int32)
        root.insert_indices_batch(indices, self.coordinates)

        build_time = time.time() - start_time
        self.logger.info(f"The quadtree has been constructed, taking: {build_time:.2f}seconds")

        self.logger.info("Calculate the loading sequence...")
        load_order = root.get_load_order()
        self.logger.info(f"The loading sequence includes {len(load_order):,} indexes")

        return root, load_order

    def save_attributes_fast(self, output_dir, load_order):
        self.logger.info("Start to rapidly save attribute data...")
        cell_types_dir = os.path.join(output_dir, 'cell_types')
        os.makedirs(cell_types_dir, exist_ok=True)

        # Create a mapping from values to indices
        value_to_index_maps = {}
        for field in self.obs_fields:
            value_to_index_maps[field] = {
                val: idx for idx, val in enumerate(self.field_values[field])
            }

        # Convert "load_order" to a numpy array
        load_order_array = np.array(load_order, dtype=np.int32)

        def save_field(field):
            """Save a single attribute field"""
            self.logger.info(f"  Processing field: {field}")
            field_data = self.obs_data[field]
            value_to_idx = value_to_index_maps[field]

            # Using numpy for vectorized mapping
            unique_vals = self.field_values[field]
            val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
            vectorized_lookup = np.vectorize(val_to_int.get)
            indexed_data = vectorized_lookup(field_data, -1).astype(np.int32)

            # Sort by "load_order"
            reordered_data = indexed_data[load_order_array]

            # save
            storage_type, np_dtype = detect_integer_dtype(len(unique_vals))
            output_file = os.path.join(cell_types_dir, f"{field}.{storage_type}.bin")
            save_integer_column(output_file, reordered_data, np_dtype)
            self.field_storage_meta[field] = {
                'storageType': storage_type,
                'storageFile': os.path.join('cell_types', f"{field}.{storage_type}.bin").replace('\\', '/'),
                'uniqueCount': len(unique_vals),
                'isHierarchical': field in ('Siletti_modf_SCANVI_L1', 'Siletti_modf_SCANVI_L2', 'Class')
            }
            self.logger.info(f"  Field '{field}' saved!")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            list(executor.map(save_field, self.obs_fields))

        values_dir = os.path.join(output_dir, 'cell_type_values')
        os.makedirs(values_dir, exist_ok=True)

        for field, values in self.field_values.items():
            field_values_file = os.path.join(values_dir, f"{field}.json")
            with open(field_values_file, 'w', encoding='utf-8') as f:
                json.dump(values, f, ensure_ascii=False)
            self.logger.info(f"  The value mapping for field '{field}' has been saved!")

        self._save_field_colors(output_dir)

        self.logger.info("Attribute data saving completed!")
        return self.field_storage_meta

    def _save_field_colors(self, output_dir):
        """
        Extract the color mapping fields from the "uns" section of h5ad and save them.
        """
        colors_dir = os.path.join(output_dir, 'field_colors')
        os.makedirs(colors_dir, exist_ok=True)

        uns = self.adata.uns
        saved_count = 0

        for field in self.obs_fields:
            color_key = f"{field}_colors"
            if color_key not in uns:
                continue

            raw_colors = uns[color_key]
            field_vals = self.field_values[field]

            # Obtain the original order of categories
            obs_series = self.adata.obs[field]
            if hasattr(obs_series, 'cat') and hasattr(obs_series.cat, 'categories'):
                cat_order = list(obs_series.cat.categories.astype(str))
            else:
                continue

            # Build a mapping from category to color
            cat_to_color = {}
            for i, cat in enumerate(cat_order):
                if i < len(raw_colors):
                    cat_to_color[cat] = str(raw_colors[i])

            ordered_colors = []
            for val in field_vals:
                if val in cat_to_color:
                    ordered_colors.append(cat_to_color[val])
                else:
                    ordered_colors.append(None)

            # save
            color_file = os.path.join(colors_dir, f"{field}.json")
            with open(color_file, 'w', encoding='utf-8') as f:
                json.dump(ordered_colors, f, ensure_ascii=False)
            saved_count += 1
            self.logger.info(f"  The color mapping for field '{field}' has been saved (with {len(ordered_colors)} colors)")

        self.logger.info(f"Color mapping saved successfully: {saved_count}/{len(self.obs_fields)} fields have custom colors")

def _process_gene_chunk_worker(args):
    """
    Multi-process Worker: Processes sparse data of a single gene chunk and writes it to a bin file
    """
    (chunk_idx, gene_start, gene_end, expr_data, expr_indices, expr_indptr,
     num_cells, original_to_quad_pos, threshold, genes_dir, total_cells) = args

    chunk_name = f"chunk_{chunk_idx}"
    num_chunk_genes = gene_end - gene_start

    try:
        chunk_indptr = expr_indptr[gene_start:gene_end + 1]
        data_start = chunk_indptr[0]
        data_end = chunk_indptr[-1]
        chunk_data = expr_data[data_start:data_end]
        chunk_indices_arr = expr_indices[data_start:data_end]
        chunk_indptr_adjusted = chunk_indptr - data_start

        sparse_data = {}
        empty_values = np.array([], dtype=np.float32)
        empty_indices = np.array([], dtype=np.uint32)

        for local_gene_idx in range(num_chunk_genes):
            start_ptr = chunk_indptr_adjusted[local_gene_idx]
            end_ptr = chunk_indptr_adjusted[local_gene_idx + 1]
            gene_values = chunk_data[start_ptr:end_ptr]
            gene_indices = chunk_indices_arr[start_ptr:end_ptr]

            if len(gene_values) == 0:
                sparse_data[local_gene_idx] = {'values': empty_values, 'indices': empty_indices}
                continue

            threshold_mask = gene_values >= threshold
            if not np.any(threshold_mask):
                sparse_data[local_gene_idx] = {'values': empty_values, 'indices': empty_indices}
                continue

            filtered_values = gene_values[threshold_mask].astype(np.float32, copy=False)
            filtered_indices = gene_indices[threshold_mask]
            quad_positions = original_to_quad_pos[filtered_indices]
            valid_positions_mask = quad_positions >= 0

            if not np.any(valid_positions_mask):
                sparse_data[local_gene_idx] = {'values': empty_values, 'indices': empty_indices}
                continue

            quad_tree_indices = quad_positions[valid_positions_mask].astype(np.uint32, copy=False)
            filtered_values = np.round(filtered_values[valid_positions_mask], 2).astype(np.float32, copy=False)

            if len(quad_tree_indices) > 1:
                sort_indices = np.argsort(quad_tree_indices)
                quad_tree_indices = quad_tree_indices[sort_indices]
                filtered_values = filtered_values[sort_indices]

            sparse_data[local_gene_idx] = {
                'values': filtered_values,
                'indices': quad_tree_indices
            }

        output_file = os.path.join(genes_dir, f"{chunk_name}.bin")
        SparseGeneData.save_sparse_chunk(sparse_data, output_file)

        original_size = num_chunk_genes * total_cells * 4
        actual_size = os.path.getsize(output_file)
        stats = SparseGeneData.get_compression_stats(original_size, sparse_data)

        return (True, chunk_idx, stats['compression_ratio'], actual_size / 1024 / 1024)

    except Exception as e:
        return (False, chunk_idx, 0, 0, str(e))

def load_expression_matrix_fast(file_path, adata_backed, num_cells, num_genes, logger):
    """
    Load the expression matrix with a fast path for on-disk CSR matrices.
    Falls back to the existing backed/dense handling when needed.
    """
    expr_matrix = adata_backed.X

    if sparse.issparse(expr_matrix):
        logger.info("The expression matrix has been stored in memory in a sparse format, and has been directly converted to CSC format....")
        return expr_matrix.tocsc()

    try:
        with h5py.File(file_path, 'r') as h5_file:
            x_node = h5_file.get('X')
            if (isinstance(x_node, h5py.Group)
                and x_node.attrs.get('encoding-type') == 'csr_matrix'
                and all(key in x_node for key in ('data', 'indices', 'indptr'))):
                
                logger.info("Detected the CSR matrix within H5AD. Proceeding with the direct reading fast path of HDF5...")
                csr_data = x_node['data'][()]
                csr_indices = x_node['indices'][()]
                csr_indptr = x_node['indptr'][()]
                csr_shape = tuple(x_node.attrs.get('shape', (num_cells, num_genes)))

                expr_csr = sparse.csr_matrix(
                    (csr_data, csr_indices, csr_indptr),
                    shape=csr_shape
                )
                logger.info(f"CSR direct reading completed: shape={expr_csr.shape}, nnz={expr_csr.nnz:,}, starting conversion to CSC...")
                expr_csc = expr_csr.tocsc()
                del expr_csr
                gc.collect()
                return expr_csc
    except Exception as error:
        logger.warning(f"The CSR fast path failed, so we reverted to the old logic: {error}")

    if hasattr(expr_matrix, 'shape') and not isinstance(expr_matrix, np.ndarray):
        logger.info("Detected HDF5-backed dataset. Switching to batch reading mode...")
        batch_size = min(50000, num_cells)
        chunks = []
        for start in range(0, num_cells, batch_size):
            end = min(start + batch_size, num_cells)
            chunk = expr_matrix[start:end]
            if sparse.issparse(chunk):
                chunks.append(chunk.tocsr())
            else:
                chunks.append(sparse.csr_matrix(chunk))
            if (start // batch_size) % 10 == 0:
                logger.info(f"  Reading progress: {end:,}/{num_cells:,} ({100 * end / num_cells:.1f}%)")

        logger.info("Merge batch data...")
        expr_csr = sparse.vstack(chunks, format='csr')
        del chunks
        gc.collect()

        logger.info("Convert to CSC format...")
        expr_csc = expr_csr.tocsc()
        del expr_csr
        gc.collect()
        return expr_csc

    logger.info("Convert the dense matrix to CSC sparse format...")
    return sparse.csc_matrix(expr_matrix)

def _build_fields_summary(processor):
    """Build field summary information"""
    field_summaries = []
    for field in processor.obs_fields:
        storage_meta = processor.field_storage_meta.get(field, {})
        field_summaries.append({
            'name': field,
            'sourceField': field,
            'storageFile': storage_meta.get('storageFile'),
            'storageType': storage_meta.get('storageType'),
            'uniqueCount': storage_meta.get('uniqueCount', len(processor.field_values.get(field, []))),
            'isHierarchical': storage_meta.get('isHierarchical', False)
        })

    if 'Siletti_modf_SCANVI_L2' in processor.field_storage_meta:
        class_meta = processor.field_storage_meta['Siletti_modf_SCANVI_L2']
        field_summaries.append({
            'name': 'Class',
            'sourceField': 'Siletti_modf_SCANVI_L2',
            'storageFile': class_meta.get('storageFile'),
            'storageType': class_meta.get('storageType'),
            'uniqueCount': class_meta.get('uniqueCount', 0),
            'isHierarchical': True
        })

    return field_summaries

def _write_field_catalogs_v2(output_dir, processor):
    """Write into field classification directory"""
    for field in processor.obs_fields:
        storage_meta = processor.field_storage_meta.get(field, {})
        catalog = {
            'name': field,
            'sourceField': field,
            'storageFile': storage_meta.get('storageFile'),
            'storageType': storage_meta.get('storageType'),
            'isHierarchical': storage_meta.get('isHierarchical', False),
            'values': processor.field_values.get(field, []),
            'counts': processor.field_counts.get(field, []),
            'colors': processor.field_colors.get(field)
        }
        write_field_catalog_v2(output_dir, field, catalog)

    if 'Siletti_modf_SCANVI_L1' in processor.field_values and 'Siletti_modf_SCANVI_L2' in processor.field_values:
        l2_meta = processor.field_storage_meta.get('Siletti_modf_SCANVI_L2', {})
        class_catalog = {
            'name': 'Class',
            'sourceField': 'Siletti_modf_SCANVI_L2',
            'storageFile': l2_meta.get('storageFile'),
            'storageType': l2_meta.get('storageType'),
            'isHierarchical': True,
            'sourceFields': ['Siletti_modf_SCANVI_L1', 'Siletti_modf_SCANVI_L2'],
            'tree': build_class_tree_catalog(
                processor.field_values['Siletti_modf_SCANVI_L1'],
                processor.field_values['Siletti_modf_SCANVI_L2'],
                processor.field_counts['Siletti_modf_SCANVI_L1'],
                processor.field_counts['Siletti_modf_SCANVI_L2']
            )
        }
        write_field_catalog_v2(output_dir, 'Class', class_catalog)

def _save_metadata(output_dir, blocks_dir, quad_tree, processor, all_genes,
                  gene_to_chunk, chunk_to_genes, h5ad_filename,
                  total_genes, total_cells, genes_per_chunk, num_chunks,
                  coordinates):
    """Save all the metadata JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(blocks_dir, exist_ok=True)
    
    logger = processor.logger

    # Field information
    field_info = {
        'total': total_cells,
        'fields': processor.obs_fields,
        'bounds': processor.bounds,
        'h5ad_filename': os.path.basename(h5ad_filename)
    }
    with open(os.path.join(output_dir, 'fields_info.json'), 'w', encoding='utf-8') as f:
        json.dump(field_info, f, ensure_ascii=False)

    field_summaries = _build_fields_summary(processor)
    write_fields_summary_v2(output_dir, field_summaries)
    _write_field_catalogs_v2(output_dir, processor)

    # gene list
    with open(os.path.join(output_dir, 'gene_list.json'), 'w', encoding='utf-8') as f:
        json.dump(all_genes, f, ensure_ascii=False)

    # gene mapping
    with open(os.path.join(output_dir, 'gene_to_chunk.json'), 'w', encoding='utf-8') as f:
        json.dump(gene_to_chunk, f, ensure_ascii=False)

    with open(os.path.join(output_dir, 'chunk_to_genes.json'), 'w', encoding='utf-8') as f:
        json.dump(chunk_to_genes, f, ensure_ascii=False)

 
    matrix_info = {
        'dtype': 'float32',
        'total_genes': total_genes,
        'total_cells': total_cells,
        'genes_per_chunk': genes_per_chunk,
        'total_chunks': num_chunks,
        'matrix_shape': [total_genes, total_cells],
        'storage_format': 'sparse',
        'sparse_enabled': True,
        'expression_threshold': processor.config.expression_threshold
    }
    with open(os.path.join(output_dir, 'matrix_info.json'), 'w', encoding='utf-8') as f:
        json.dump(matrix_info, f, ensure_ascii=False)

    from .quadtree import FastQuadTreeNode
    tree_info = {
        'total_blocks': FastQuadTreeNode._id_counter,
        'root': quad_tree.save_node_info(blocks_dir, coordinates)
    }
    with open(os.path.join(output_dir, 'quad_tree.json'), 'w', encoding='utf-8') as f:
        json.dump(tree_info, f, ensure_ascii=False)

    manifest = {
        'formatVersion': 2,
        'datasetId': os.path.basename(os.path.abspath(output_dir)),
        'datasetName': os.path.basename(os.path.abspath(output_dir)),
        'sourceFile': os.path.basename(h5ad_filename),
        'totalCells': total_cells,
        'totalGenes': total_genes,
        'embeddings': ['umap'],
        'defaultEmbedding': 'umap',
        'files': {
            'fieldsInfo': 'fields_info.json',
            'fieldsSummary': 'fields_summary.json',
            'quadTree': 'quad_tree.json',
            'matrixInfo': 'matrix_info.json',
            'geneToChunk': 'gene_to_chunk.json',
            'chunkToGenes': 'chunk_to_genes.json'
        },
        'dirs': {
            'blocks': 'blocks',
            'cellTypes': 'cell_types',
            'fieldCatalog': 'field_catalog',
            'genes': 'genes'
        }
    }
    write_manifest_v2(output_dir, manifest)

    logger.info("Metadata saving completed")

def process_h5ad_ultra_fast(file_path, output_dir, config=None, skip_genes=False, temp_dir=None):
    from .utils import setup_logging
    
    logger = setup_logging()
    start_total = time.time()

    if temp_dir is None:
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="h5ad_proc_")

    if config is None:
        config = ProcessingConfig()
    
    logger.info(f"=" * 60)
    logger.info(f"Ultra-high-performance H5AD processing tool")
    logger.info(f"config: {config}")
    logger.info(f"input: {file_path}")
    logger.info(f"output: {output_dir}")
    logger.info(f"=" * 60)

    # ====================================================================
    logger.info(f"read: {file_path}")
    file_size_gb = os.path.getsize(file_path) / (1024**3)
    logger.info(f"file size: {file_size_gb:.2f}GB, Available memory: {config.memory_limit_gb:.2f}GB")

    adata_backed = sc.read_h5ad(file_path, backed='r')
    num_cells = adata_backed.shape[0]
    num_genes = adata_backed.shape[1]
    logger.info(f"Data size {num_cells:,} cells, {num_genes:,} genes")

    processor = FastDataProcessor(adata_backed, config)

    quad_tree, load_order = processor.build_quadtree_fast()

    processor.save_attributes_fast(output_dir, load_order)

    all_genes = adata_backed.var_names.tolist()
    total_cells = len(load_order)
    total_genes = len(all_genes)
    gene_to_chunk = {}
    chunk_to_genes = {}
    genes_per_chunk = 0
    num_chunks = 0

    if skip_genes:
        logger.info("=" * 50)
        logger.info("Skip the processing of genetic data")
        adata_backed.file.close()

        logger.info("Save metadata...")
        blocks_dir = os.path.join(output_dir, 'blocks')
        _save_metadata(
            output_dir, blocks_dir, quad_tree, processor,
            all_genes, gene_to_chunk, chunk_to_genes,
            file_path, total_genes, total_cells,
            genes_per_chunk, num_chunks,
            processor.coordinates
        )

        del processor
        gc.collect()

        total_time = time.time() - start_total
        logger.info(f"🎉 Processing completed! Time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        return total_cells, total_genes

    # ====================================================================
    logger.info("=" * 50)
    logger.info("Start reading the expression matrix...")
    t_matrix = time.time()

    expr_csc = load_expression_matrix_fast(file_path, adata_backed, num_cells, num_genes, logger)

    adata_backed.file.close()

    matrix_read_time = time.time() - t_matrix
    logger.info(f"Matrix reading completed. Time taken: {matrix_read_time:.2f} seconds")
    logger.info(f"  CSC matrix: {expr_csc.shape}, Non-zero elements: {expr_csc.nnz:,}")
    logger.info(f"  Sparsity rate: {100 * (1 - expr_csc.nnz / (expr_csc.shape[0] * expr_csc.shape[1])):.2f}%")

    csc_data = np.ascontiguousarray(expr_csc.data.astype(np.float32))
    csc_indices = np.ascontiguousarray(expr_csc.indices.astype(np.int32))
    csc_indptr = np.ascontiguousarray(expr_csc.indptr.astype(np.int64))

    del expr_csc
    gc.collect()

    # ====================================================================
    if config.genes_per_chunk > 0:
        genes_per_chunk = config.genes_per_chunk
        logger.info(f"Use the gene block size specified by the user: {genes_per_chunk}")
    else:
        genes_per_chunk = calculate_genes_per_chunk(total_cells, config.max_chunk_size_mb)
        logger.info(f"Automatically calculate the size of gene chunks: {genes_per_chunk} (based on {total_cells:,} cells)")

    num_chunks = (total_genes + genes_per_chunk - 1) // genes_per_chunk
    logger.info(f"Genetic data chunking: {num_chunks} chunks, with each chunk containing a maximum of {genes_per_chunk} genes.")

    gene_to_chunk = {}
    chunk_to_genes = {}
    coordinate_order_array = np.array(load_order, dtype=np.int32)
    original_to_quad_pos = np.full(num_cells, -1, dtype=np.int32)
    valid_mask = coordinate_order_array < num_cells
    original_to_quad_pos[coordinate_order_array[valid_mask]] = np.arange(len(load_order), dtype=np.int32)[valid_mask]

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * genes_per_chunk
        end_idx = min((chunk_idx + 1) * genes_per_chunk, total_genes)
        chunk_genes = all_genes[start_idx:end_idx]
        chunk_name = f"chunk_{chunk_idx}"
        chunk_to_genes[chunk_name] = chunk_genes
        for gene in chunk_genes:
            gene_to_chunk[gene] = chunk_name

    logger.info("=" * 50)
    logger.info("Start processing the gene expression data...")
    genes_dir = os.path.join(output_dir, 'genes')
    os.makedirs(genes_dir, exist_ok=True)

    t_genes = time.time()

    chunk_args = []
    for chunk_idx in range(num_chunks):
        gene_start = chunk_idx * genes_per_chunk
        gene_end = min((chunk_idx + 1) * genes_per_chunk, total_genes)
        chunk_args.append((
            chunk_idx, gene_start, gene_end,
            csc_data, csc_indices, csc_indptr,
            num_cells, original_to_quad_pos,
            config.expression_threshold, genes_dir, total_cells
        ))

    num_workers = min(config.max_workers, num_chunks)
    logger.info(f"Use {num_workers} parallel threads to process the genetic data...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_process_gene_chunk_worker, chunk_args))

    for result in results:
        if result[0]:
            success_count += 1
            _, cidx, ratio, size_mb = result
            if (cidx + 1) % 50 == 0 or cidx == num_chunks - 1:
                logger.info(f"  Chunk {cidx+1}/{num_chunks} completed: Compression ratio {ratio:.2f}x, {size_mb:.2f}MB")
        else:
            _, cidx, _, _, err = result
            logger.error(f"  Chunk {cidx+1} failed: {err}")

    gene_time = time.time() - t_genes
    logger.info(f"Genetic data processing completed: {success_count}/{num_chunks} chunks processed successfully, time taken: {gene_time:.2f} seconds")

    del csc_data, csc_indices, csc_indptr
    gc.collect()

    logger.info("Save metadata...")
    blocks_dir = os.path.join(output_dir, 'blocks')

    _save_metadata(
        output_dir, blocks_dir, quad_tree, processor,
        all_genes, gene_to_chunk, chunk_to_genes,
        file_path, total_genes, total_cells,
        genes_per_chunk, num_chunks,
        processor.coordinates
    )

    del processor
    gc.collect()

    total_time = time.time() - start_total
    logger.info(f"🎉 Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    return total_cells, total_genes
