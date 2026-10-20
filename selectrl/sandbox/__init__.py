"""Sandboxed execution of untrusted Python programs: persistent forked workers, a process pool, output comparison."""
from .compare import out_key, same_output
from .pool import ExecPool, default_workers, get_pool, pmap, run_prog, shutdown_pool

__all__ = ["ExecPool", "default_workers", "get_pool", "out_key", "pmap", "run_prog", "same_output", "shutdown_pool"]
