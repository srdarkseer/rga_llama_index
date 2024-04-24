import psycopg2
from src.config import db_config

class DatabaseManager:
    def __init__(self):
        self.connection = psycopg2.connect(
            dbname=db_config['database'],
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            port=db_config['port']
        )
        self.connection.autocommit = True

    def close(self):
        self.connection.close()

    def execute_query(self, query):
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            try:
                return cursor.fetchall()
            except psycopg2.ProgrammingError:
                return None  # No results to fetch
