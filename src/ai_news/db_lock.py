"""Shared database locks for thread-safe concurrent access.

This module provides a single global lock that all database writes must use
to prevent SQLite "database is locked" errors during concurrent operations.
"""

import threading

# Global write lock for ALL database operations (reentrant for nested calls)
# Any code that writes to SQLite must acquire this lock first
_db_write_lock = threading.RLock()


def get_db_write_lock():
    """Get the global database write lock (reentrant).

    All SQLite write operations must use this lock to prevent concurrent writes.
    RLock allows the same thread to acquire it multiple times (nested calls).
    """
    return _db_write_lock
