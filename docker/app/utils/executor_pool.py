"""
Executor Pool Utility

This module provides a shared thread pool executor to avoid creating
multiple executor instances across different services.
"""

import atexit
import concurrent.futures
from typing import Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)


class ExecutorPool:
    """Singleton executor pool for shared use across services"""

    _instance: Optional["ExecutorPool"] = None
    _executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_workers: int = 10):
        """
        Initialize the executor pool

        Args:
            max_workers: Maximum number of worker threads
        """
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="shared_executor"
            )
            logger.info(
                f"Created shared executor pool with {max_workers} workers"
            )

            # Register cleanup on exit
            atexit.register(self._cleanup)

    @property
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Get the shared executor instance"""
        if self._executor is None:
            self.__init__()
        return self._executor

    def submit(self, fn, *args, **kwargs):
        """Submit a task to the executor pool"""
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True):
        """Shutdown the executor pool"""
        if self._executor:
            logger.info("Shutting down shared executor pool")
            self._executor.shutdown(wait=wait)
            self._executor = None

    def _cleanup(self):
        """Cleanup function called on exit"""
        self.shutdown(wait=False)


# Global shared executor pool instance
shared_executor_pool = ExecutorPool()
