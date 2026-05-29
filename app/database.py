import os
import psycopg
import threading

class DatabaseManager:
    """Thread-safe Singleton Database Manager to store connection details and yield connections."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls, *args, **kwargs)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._dsn = os.getenv("DATABASE_URL")
        if not self._dsn:
            raise RuntimeError("DATABASE_URL environment variable is not set")
        self._initialized = True

    def get_connection(self, **kwargs):
        """Creates and returns a psycopg connection using the stored DSN."""
        return psycopg.connect(self._dsn, **kwargs)

def get_db_connection(**kwargs):
    """Get a database connection using the DatabaseManager singleton."""
    return DatabaseManager().get_connection(**kwargs)

