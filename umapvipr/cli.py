"""
Command Line Interface Module
"""

import os
import sys
import argparse
import tempfile
import shutil
import logging
from .utils import setup_logging
from .processor import process_h5ad_ultra_fast
from .config import ProcessingConfig

def main():
    parser = argparse.ArgumentParser(
        description='Umap Visual Processor - Converts H5AD files into web-friendly formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
example:
  umapvipr -i input.h5ad -o output/
  umapvipr -i input.h5ad -o output/ --max-workers 4
  umapvipr -i input.h5ad -o output/ --skip-genes
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Enter the path of the H5AD file')
    parser.add_argument('--output', '-o', default="output", help='Output directory')
    parser.add_argument('--max-workers', '-w', type=int, default=0, help='Number of concurrent threads')
    parser.add_argument('--batch-size', type=int, default=200000, help='batch size')
    parser.add_argument('--genes-per-chunk', '-g', type=int, default=0,
                       help='The number of genes contained in each block (default: automatically calculated)')
    parser.add_argument('--max-chunk-size', '-m', type=int, default=2,
                       help='Maximum size of a single gene chunk file (MB) (default: 2MB)')
    parser.add_argument('--expression-threshold', type=float, default=0.01,
                       help='Gene expression level threshold (default: 0.01)')
    parser.add_argument('--max-points-per-node', type=int, default=60000,
                       help='Maximum number of points for a quadtree node (default: 60000)')
    parser.add_argument('--skip-genes', action='store_true',
                       help='Skip the processing of genetic data')
    parser.add_argument('--memory-limit-gb', type=float, default=0,
                       help='Memory Limit (GB) (Default: Automatic Detection)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Log level')
    parser.add_argument('--log-file', help='Log file path')
    
    args = parser.parse_args()
    
    logger = setup_logging(
        log_level=getattr(logging, args.log_level.upper()),
        log_file=args.log_file
    )
    
    config_kwargs = {
        'max_workers': args.max_workers if args.max_workers > 0 else 0,
        'batch_size': args.batch_size,
        'genes_per_chunk': args.genes_per_chunk,
        'max_chunk_size_mb': args.max_chunk_size,
        'expression_threshold': args.expression_threshold,
        'max_points_per_node': args.max_points_per_node
    }
    
    if args.memory_limit_gb > 0:
        config_kwargs['memory_limit_gb'] = args.memory_limit_gb
    
    config = ProcessingConfig(**config_kwargs)
    
    logger.info("=" * 60)
    logger.info("H5AD processor has started")
    logger.info("=" * 60)
    logger.info(f"input: {args.input}")
    logger.info(f"output: {args.output}")
    logger.info(f"Configuration:")
    logger.info(f"  Parallel threads: {config.max_workers}")
    logger.info(f"  batch size: {config.batch_size:,}")
    logger.info(f"  Gene chunk: {config.genes_per_chunk if config.genes_per_chunk > 0 else 'Automatically calculated'}")
    logger.info(f"  Expression threshold: {config.expression_threshold}")
    logger.info(f"  Quadtree node capacity: {config.max_points_per_node:,}")
    if args.skip_genes:
        logger.info("  Skip the genetic data processing: Yes")
    logger.info("=" * 60)

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="h5ad_proc_")
        
        total_cells, total_genes = process_h5ad_ultra_fast(
            args.input,
            args.output,
            config=config,
            skip_genes=args.skip_genes,
            temp_dir=temp_dir
        )
        
        logger.info("=" * 60)
        logger.info(f"✅ Processing completed!")
        logger.info(f"  Total cells: {total_cells:,}")
        logger.info(f"   Total genes: {total_genes:,}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Failure in handling: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    sys.exit(main())
