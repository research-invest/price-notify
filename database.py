import mysql.connector
from mysql.connector import Error
from datetime import datetime
import time

class DatabaseManager:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            print("Успешное подключение к базе данных MySQL")
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

    def update_price_data(self, symbol: str, timestamp: datetime, price: float, volume: float):
        update_query = """
            UPDATE price_history
            SET price = %s, volume = %s
            WHERE symbol = %s AND timestamp = %s
        """
        self._execute_query(update_query, (price, volume, symbol, timestamp))

    def _execute_query(self, query, params=None):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not self.connection or not self.connection.is_connected():
                    self.connect()
                
                with self.connection.cursor() as cursor:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                self.connection.commit()
                return
            except Error as e:
                print(f"Ошибка при выполнении запроса (попытка {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

    def _fetch_data(self, query, params=None):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not self.connection or not self.connection.is_connected():
                    self.connect()
                
                with self.connection.cursor() as cursor:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    return cursor.fetchall()
            except Error as e:
                print(f"Ошибка при получении данных (попытка {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)