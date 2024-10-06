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
            price DECIMAL(65, 10) NOT NULL,
            volume DECIMAL(20, 8) NOT NULL
        );
        """
        self._execute_query(create_table_query)

        create_table_query = """
        CREATE TABLE IF NOT EXISTS trades (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol     varchar(20) not null,
            trade_type enum ('buy', 'sell') not null,
            price      decimal(65, 10) not null,
            amount     decimal(20, 8) not null,
            timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
;
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

    def add_trade(self, trade):
        insert_query = """
        INSERT INTO trades (symbol, price, amount, trade_type)
        VALUES (%s, %s, %s, %s)
        """
        self._execute_query(insert_query, (trade['symbol'], trade['entry_price'], trade['volume'], trade['trade_type']))

    def get_all_trades(self):
        select_query = "SELECT * FROM trades ORDER BY created_at DESC"
        return self._fetch_data(select_query)

    def update_trade(self, trade):
        update_query = """
        UPDATE trades
        SET symbol = %s, entry_price = %s, volume = %s
        WHERE id = %s
        """
        self._execute_query(update_query, (trade['symbol'], trade['entry_price'], trade['volume'], trade['id']))

    def delete_trade(self, trade_id):
        delete_query = "DELETE FROM trades WHERE id = %s"
        self._execute_query(delete_query, trade_id)

    def get_trade_stats(self):
        stats_query = """
        SELECT 
            COUNT(*) as total_trades,
            SUM(volume) as total_volume,
            AVG(entry_price) as avg_entry_price,
            SUM(volume * entry_price) as total_value
        FROM trades
        """
        stats = self._fetch_data(stats_query)

        if stats['total_trades'] > 0:
            stats['avg_entry_price'] = float(stats['avg_entry_price'])
            stats['total_volume'] = float(stats['total_volume'])
            stats['total_value'] = float(stats['total_value'])
            # Здесь вы можете добавить расчет общей прибыли/убытка, если у вас есть текущие цены
            stats['total_pnl'] = 0  # Заглушка, замените на реальный расчет
        else:
            stats = {
                'total_trades': 0,
                'total_volume': 0,
                'avg_entry_price': 0,
                'total_value': 0,
                'total_pnl': 0
            }

        return stats

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