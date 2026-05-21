# Umap visual Processor

High-performance H5AD file processing tool, which converts single-cell H5AD files into a format suitable for web visualization. 
## 🚀 Key Features 
- **High-performance processing** - Optimized for large datasets of 10GB and above
- **Intelligent memory management** - Automatically detects system resources and optimizes performance
- **Sparse storage** - Supports sparse storage formats with high compression ratio
- **Quad-tree index** - Fast spatial query and loading
- **Parallel processing** - Multi-threading/multi-process acceleration
- **Web-friendly output** - Generates data formats suitable for web visualization 
## 📦 Installation 
    pip install umapvipr
### if failed, try to download the files
    cd umapvipr
    pip install -e . --no-deps 

## Usage
The uampvipr tool provides a command-line interface (CLI) and an interactive interface (e.g., jupyter notebook) for converting single-cell H5AD files into a format suitable for web visualization. Here is an overview of the available commands and their usage:

    from umapvipr import process_h5ad_ultra_fast, ProcessingConfig
    from umapvipr.utils import setup_logging
    setup_logging()
    config = ProcessingConfig(
    max_workers=4,#Number of concurrent threads
    batch_size=200000,
    genes_per_chunk=100,#The number of genes contained in each block (default: automatically calculated)
    expression_threshold=0.01 # Gene expression level threshold (default: 0.01)
    )
    process_h5ad_ultra_fast(
    'test.h5ad', # path of the h5ad file
    'test_umap',# path to the directory containing files suitable for web visualization
    config=config
    )
