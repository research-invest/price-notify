import numpy as np
from datetime import datetime, timedelta
import bisect


class PriceAnalyzer:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.prices = {}
        self.volumes = {}
        self.timestamps = []

    def load_historical_data(self, symbols, days=30):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        all_timestamps = set()

        for symbol in symbols:
            historical_data = self.db_manager.get_historical_data(symbol, start_date, end_date)
            all_timestamps.update(row[0] for row in historical_data)

        self.timestamps = sorted(all_timestamps)

        for symbol in symbols:
            historical_data = self.db_manager.get_historical_data(symbol, start_date, end_date)
            data_dict = {row[0]: (float(row[1]), float(row[2])) for row in historical_data}

            self.prices[symbol] = []
            self.volumes[symbol] = []

            for timestamp in self.timestamps:
                if timestamp in data_dict:
                    self.prices[symbol].append(data_dict[timestamp][0])
                    self.volumes[symbol].append(data_dict[timestamp][1])
                else:
                    self.prices[symbol].append(self.prices[symbol][-1] if self.prices[symbol] else None)
                    self.volumes[symbol].append(self.volumes[symbol][-1] if self.volumes[symbol] else None)

    def analyze_prices(self, symbol: str, current_price: float) -> str:
        if len(self.prices[symbol]) < 2:
            return "Недостаточно данных для анализа."

        prices = np.array(self.prices[symbol])
        changes = np.diff(prices) / prices[:-1] * 100
        avg_change = np.mean(changes)

        last_change = (current_price - prices[-1]) / prices[-1] * 100
        volatility = np.std(changes)

        if last_change > avg_change + volatility:
            return "Сильный рост"
        elif last_change < avg_change - volatility:
            return "Сильное падение"
        elif last_change > avg_change:
            return "Умеренный рост"
        elif last_change < avg_change:
            return "Умеренное падение"
        else:
            return "Стабильность"

    def update_price(self, symbol: str, price: float, volume: float, timestamp: datetime):
        if symbol not in self.prices:
            self.prices[symbol] = []
            self.volumes[symbol] = []

        if not self.timestamps or timestamp > self.timestamps[-1]:
            self.timestamps.append(timestamp)
            self.prices[symbol].append(price)
            self.volumes[symbol].append(volume)
        else:
            # Обновляем последнюю запись, если временная метка совпадает
            if self.timestamps[-1] == timestamp:
                self.prices[symbol][-1] = price
                self.volumes[symbol][-1] = volume
            else:
                # Вставляем новую запись в правильное место
                insert_index = bisect.bisect_left(self.timestamps, timestamp)
                self.timestamps.insert(insert_index, timestamp)
                self.prices[symbol].insert(insert_index, price)
                self.volumes[symbol].insert(insert_index, volume)

        # Убедимся, что все массивы имеют одинаковую длину
        max_length = len(self.timestamps)
        self.prices[symbol] = self.prices[symbol][:max_length]
        self.volumes[symbol] = self.volumes[symbol][:max_length]
