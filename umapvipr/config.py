import multiprocessing
import os

class ProcessingConfig:

    def __init__(self, **kwargs):
        cpu_count = multiprocessing.cpu_count()
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            available_memory_gb = 16.0
            print("Warning: psutil is not installed. Using default memory configuration.")

        self.genes_per_chunk = kwargs.get('genes_per_chunk', 100)
        self.max_chunk_size_mb = kwargs.get('max_chunk_size_mb', 2)

        self.max_points_per_node = kwargs.get('max_points_per_node', 60000)
        self.coordinate_precision = kwargs.get('coordinate_precision', 2)

        self.expression_threshold = kwargs.get('expression_threshold', 0.01)
        self.batch_size = kwargs.get('batch_size', 200000)

        self.use_sparse_storage = kwargs.get('use_sparse_storage', True)
        self.sparse_compression = kwargs.get('sparse_compression', True)

        self.max_workers = kwargs.get('max_workers', min(cpu_count, 16))
        self.io_workers = kwargs.get('io_workers', max(4, cpu_count // 2))

        self.memory_limit_gb = kwargs.get('memory_limit_gb', 
                                          min(available_memory_gb * 0.8, 32.0))
        self.gc_frequency = kwargs.get('gc_frequency', 100000)

        self.use_vectorized_ops = kwargs.get('use_vectorized_ops', True)
        self.skip_validation = kwargs.get('skip_validation', False)
  
        self.read_batch_chunks = kwargs.get('read_batch_chunks', 5)
        
    def __str__(self):
        return f"""ProcessingConfig:
  CPU core: {multiprocessing.cpu_count()}, Work thread: {self.max_workers}
  Batch size: {self.batch_size:,}
  Quadtree node capacity: {self.max_points_per_node:,}
  Genetic segmentation: {self.genes_per_chunk} (The number of genes in each segmentation)
  Expression threshold: {self.expression_threshold}
  Memory Limitation: {self.memory_limit_gb:.1f} GB"""