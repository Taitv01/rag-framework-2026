"""
Read-write lock
===============

Many readers at once, or one writer. A waiting writer stops new readers from
entering, so a steady stream of queries cannot starve an ingestion.

Not reentrant: a thread holding the lock must not acquire it again (a reader
asking to read again while a writer waits would deadlock).

Usage:
    lock = ReadWriteLock()
    with lock.read():
        docs = store.search(query)
    with lock.write():
        store.add(chunks)
"""

import threading
from contextlib import contextmanager


class ReadWriteLock:
    """Shared reads, exclusive writes, writers first."""

    def __init__(self):
        self._condition = threading.Condition()
        self._readers = 0
        self._writing = False
        self._writers_waiting = 0

    @contextmanager
    def read(self):
        with self._condition:
            while self._writing or self._writers_waiting:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if not self._readers:
                    self._condition.notify_all()

    @contextmanager
    def write(self):
        with self._condition:
            self._writers_waiting += 1
            try:
                while self._writing or self._readers:
                    self._condition.wait()
            finally:
                self._writers_waiting -= 1
            self._writing = True
        try:
            yield
        finally:
            with self._condition:
                self._writing = False
                self._condition.notify_all()
