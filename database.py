import mysql.connector
from mysql.connector import Error
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.db_config)
        except Error as e:
            print(f"Ошибка при подключении к MySQL: {e}")

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Соединение с базой данных MySQL закрыто")

    def create_tables(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS price_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp DATETIME NOT NULL,
            price DECIMAL(20, 8) NOT NULL,
            volume DECIMAL(20, 8) NOT NULL
        );
        """
        self._execute_query(create_table_query)

    def save_price_data(self, symbol: str, timestamp: datetime, price: float, volume: float):
        insert_query = """
            INSERT INTO price_history (symbol, timestamp, price, volume)
            VALUES (%s, %s, %s, %s)
        """
        self._execute_query(insert_query, (symbol, timestamp, price, volume))

    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime):
        select_query = """
            SELECT timestamp, price, volume
            FROM price_history
            WHERE symbol = %s AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
        """
        return self._fetch_data(select_query, (symbol, start_date, end_date))

    def _execute_query(self, query, params=None):
        if not self.connection or not self.connection.is_connected():
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
            self.connection.commit()
        except Error as e:
            print(f"Ошибка при выполнении запроса: {e}")

    def _fetch_data(self, query, params=None):
        if not self.connection or not self.connection.is_connected():
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                return cursor.fetchall()
        except Error as e:
            print(f"Ошибка при получении данных: {e}")
            return []