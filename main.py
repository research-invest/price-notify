import os
import ccxt
import asyncio
from telegram import Bot
from telegram.error import TimedOut, TelegramError
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import logging
import math
from utils import format_number
from database import DatabaseManager
import traceback
import yfinance as yf
from matplotlib.dates import DateFormatter
from pycoingecko import CoinGeckoAPI
import requests
import matplotlib.dates as mdates
import random
from scipy import stats
from scipy.optimize import curve_fit
import psutil
import time
import sys
from logging.handlers import RotatingFileHandler

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Максимальный размер файла 10MB, храним 5 последних файлов
file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'crypto_analyzer.log'), 
    maxBytes=10*1024*1024, 
    backupCount=5
)
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

timezone = ZoneInfo("Europe/Moscow")


class CryptoAnalyzer:
    def __init__(self, exchange_name: str, symbols: list, telegram: dict, db_config: dict, interval: int,
                 stickers: dict, is_indexes: bool, timestamps_limit: int):
        self.dpi = 150
        self.timestamps_limit = timestamps_limit
        self.cg = CoinGeckoAPI()
        self.indices = {
            'SP500': {'values': [], 'timestamps': []},
            'Fear&Greed': {'values': [], 'timestamps': []},
            'BTC_Dominance': {'values': [], 'timestamps': []},
            'NASDAQ': {'values': [], 'timestamps': []},
            'Total_Market_Cap': {'values': [], 'timestamps': []},
            'Market_Cap_Change_24h': {'values': [], 'timestamps': []}
        }
        self.exchange = getattr(ccxt, exchange_name)()
        self.symbols = [symbol['name'] for symbol in symbols]
        self.symbol_colors = {symbol['name']: symbol['color'] for symbol in symbols}
        self.symbol_line_widths = {symbol['name']: symbol.get('line_width', 1) for symbol in symbols}
        self.bot = Bot(token=telegram['token'])
        self.chat_id = telegram['chat_id']
        self.interval = interval
        self.prices = {symbol: [] for symbol in self.symbols}
        self.volumes = {symbol: [] for symbol in self.symbols}
        self.open_interest = {symbol: [] for symbol in self.symbols}
        self.timestamps = []
        self.db_manager = DatabaseManager(db_config)
        self.db_manager.connect()
        self.db_manager.create_tables()
        self.load_historical_data()
        self.last_indices_update = datetime.now(timezone)
        self.stickers = stickers
        self.is_indexes = is_indexes
        self.trading_sessions = {
            'Азиатская': {'start': 0, 'end': 8, 'color': 'gray'},    # 00:00-08:00 UTC
            'Лондонская': {'start': 8, 'end': 16, 'color': 'gray'},  # 08:00-16:00 UTC
            'Нью-Йоркская': {'start': 13, 'end': 21, 'color': 'gray'} # 13:00-21:00 UTC
        }
        self.cpu_threshold = 70  # Порог загрузки CPU в процентах
        self.performance_log_interval = 60  # Логировать производительность каждые 60 секунд
        self.last_performance_log = time.time()

    def load_historical_data(self):
        end_date = datetime.now(timezone)
        start_date = end_date - timedelta(days=1)
        all_timestamps = set()

        # Сначала соберем все уникальные временные метки
        for symbol in self.symbols:
            historical_data = self.db_manager.get_historical_data(symbol, start_date, end_date)
            all_timestamps.update(row[0] for row in historical_data)

        # Отсортируем временные метки
        self.timestamps = sorted(all_timestamps)

        for index in self.indices:
            try:
                historical_data = self.db_manager.get_historical_data(index, start_date, end_date)
                data_dict = {row[0]: (float(row[1]), float(row[2])) for row in historical_data}

                self.indices[index]['timestamps'] = list(data_dict.keys())
                self.indices[index]['values'] = [value[0] for value in data_dict.values()]

            except Exception as e:
                print(f"Error loading data for {index}: {e}")
                self.indices[index]['timestamps'] = []
                self.indices[index]['values'] = []

        # Теперь заполним данные для каждого символа
        for symbol in self.symbols:
            historical_data = self.db_manager.get_historical_data(symbol, start_date, end_date)
            data_dict = {row[0]: (float(row[1]), float(row[2]), float(row[3])) for row in historical_data}

            self.prices[symbol] = []
            self.volumes[symbol] = []
            self.open_interest[symbol] = []

            for timestamp in self.timestamps:
                if timestamp in data_dict:
                    self.prices[symbol].append(data_dict[timestamp][0])
                    self.volumes[symbol].append(data_dict[timestamp][1])
                    self.open_interest[symbol].append(data_dict[timestamp][2])
                else:
                    # Если данных нет, используем предыдущее значение или None
                    self.prices[symbol].append(self.prices[symbol][-1] if self.prices[symbol] else None)
                    self.volumes[symbol].append(self.volumes[symbol][-1] if self.volumes[symbol] else None)
                    self.open_interest[symbol].append(self.open_interest[symbol][-1] if self.open_interest[symbol] else None)

            logger.info(f"Загружено {len(historical_data)} исторических записей для {symbol}")

    def check_cpu_usage(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        current_time = time.time()
        
        # Логируем производительность с заданным интервалом
        if current_time - self.last_performance_log >= self.performance_log_interval:
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # Память в МБ
            logger.info(f"Performance metrics - CPU: {cpu_percent}%, Memory: {memory:.1f}MB")
            self.last_performance_log = current_time

        if cpu_percent >= self.cpu_threshold:
            logger.warning(f"CPU usage is too high: {cpu_percent}%. Shutting down...")
            return False
        return True

    async def update_prices(self):
        try:
            # Проверяем CPU перед выполнением
            if not self.check_cpu_usage():
                logger.info("Gracefully shutting down due to high CPU usage")
                sys.exit(0)

            start_time = time.time()
            
            current_time = datetime.now()

            if self.is_indexes:
                # Обновляем данные S&P 500
                sp500 = yf.Ticker("^GSPC")
                latest_sp500 = float(sp500.history(period="1d")['Close'].iloc[-1])
                self.db_manager.save_price_data('SP500', current_time, latest_sp500, 0, 0)
                self.indices['SP500']['values'].append(latest_sp500)
                self.indices['SP500']['timestamps'].append(current_time)

                # Обновляем индекс страха и жадности
                fear_greed_data = requests.get('https://api.alternative.me/fng/').json()
                fear_greed_value = int(fear_greed_data['data'][0]['value'])
                self.db_manager.save_price_data('Fear&Greed', current_time, fear_greed_value,
                                                0, 0)
                self.indices['Fear&Greed']['values'].append(fear_greed_value)
                self.indices['Fear&Greed']['timestamps'].append(current_time)

                # Обновляем индекс доминирования биткоина
                bitcoin_data = self.cg.get_global()
                btc_dominance = float(bitcoin_data['market_cap_percentage']['btc'])
                self.db_manager.save_price_data('BTC_Dominance', current_time, btc_dominance,
                                                0, 0)
                self.indices['BTC_Dominance']['values'].append(btc_dominance)
                self.indices['BTC_Dominance']['timestamps'].append(current_time)

                # Добавим также общую капитализацию рынка
                total_market_cap = float(bitcoin_data['total_market_cap']['usd'])

                self.db_manager.save_price_data('Total_Market_Cap', current_time, total_market_cap,
                                                0, 0)
                self.indices['Total_Market_Cap']['values'].append(total_market_cap)
                self.indices['Total_Market_Cap']['timestamps'].append(current_time)

                # И изменение капитализации за 24 часа
                market_cap_change_24h = float(bitcoin_data['market_cap_change_percentage_24h_usd'])
                self.db_manager.save_price_data('Total_Market_Cap', current_time, market_cap_change_24h,
                                                0, 0)
                self.indices['Market_Cap_Change_24h']['values'].append(market_cap_change_24h)
                self.indices['Market_Cap_Change_24h']['timestamps'].append(current_time)

                # Обновляем NASDAQ-100
                nasdaq = yf.Ticker("^NDX")
                latest_nasdaq = float(nasdaq.history(period="1d")['Close'].iloc[-1])
                self.db_manager.save_price_data('NASDAQ', current_time, latest_nasdaq, 0, 0)
                self.indices['NASDAQ']['values'].append(latest_nasdaq)
                self.indices['NASDAQ']['timestamps'].append(current_time)

            if not self.timestamps or current_time > self.timestamps[-1]:
                self.timestamps.append(current_time)
                for symbol in self.symbols:
                    price, volume, open_interest = self.get_price_and_volume(symbol)

                    price = float(price)
                    volume = float(volume)
                    open_interest = float(open_interest) if open_interest is not None else None

                    self.prices[symbol].append(price)
                    self.volumes[symbol].append(volume)
                    self.open_interest[symbol].append(open_interest)

                    # Сохранение данных в БД
                    try:
                        self.db_manager.save_price_data(symbol, current_time, price, volume, open_interest)
                    except Exception as db_error:
                        logger.error(f"Ошибка при сохранении данных в БД: {db_error}")
            else:
                # Обновляем последнюю запись, если время совпадает
                for symbol in self.symbols:
                    price, volume, open_interest = self.get_price_and_volume(symbol)
                    price = float(price)
                    volume = float(volume)
                    open_interest = float(open_interest) if open_interest is not None else None

                    self.prices[symbol][-1] = price
                    self.volumes[symbol][-1] = volume
                    self.open_interest[symbol][-1] = open_interest
                    # Обновляем запись в БД
                    try:
                        self.db_manager.update_price_data(symbol, current_time, price, volume)
                    except Exception as db_error:
                        logger.error(f"Ошибка при обновлении данных в БД: {db_error}")

            # Тихий режим
            # if 2 <= datetime.now(timezone).hour < 6:
            #     return

            message = ''
            overall_sentiment = 0
            for symbol in self.symbols:
                formatted_price = format_number(self.prices[symbol][-1])
                formatted_volume = format_number(self.volumes[symbol][-1])
                analysis = self.analyze_prices(symbol, self.prices[symbol][-1])
                message += f"{symbol}:\nЦена: {formatted_price}\nОбъем: {formatted_volume}\nАнализ: {analysis}\n\n"

                # Простой подсчет настроения
                if "восходящий" in analysis:
                    overall_sentiment += 1
                elif "нисходящий" in analysis:
                    overall_sentiment -= 1

            if self.is_indexes:
                message += f"S&P 500: {latest_sp500:.2f}\n\n"

            if len(self.timestamps) > self.timestamps_limit:
                self.timestamps = self.timestamps[-self.timestamps_limit:]
                for symbol in self.symbols:
                    self.prices[symbol] = self.prices[symbol][-self.timestamps_limit:]
                    self.volumes[symbol] = self.volumes[symbol][-self.timestamps_limit:]
                    self.open_interest[symbol] = self.open_interest[symbol][-self.timestamps_limit:]

            await self.send_message(message)

            # Новый график: цены, объемы, OI, RCI
            price_volume_oi_rci_chart = self.create_price_volume_oi_rci_chart()
            await self.send_chart(price_volume_oi_rci_chart)

            # Отправляем график индексов раз в час
            current_time = datetime.now(timezone)
            if self.is_indexes and current_time.minute == 0 and (
                    current_time - self.last_indices_update).total_seconds() >= 3600:
                indices_chart = self.create_indices_chart()
                await self.send_chart(indices_chart)
                self.last_indices_update = current_time.replace(minute=0, second=0, microsecond=0)

            # Отправка стикера на основе общего настроения
            if overall_sentiment > 0:
                await self.send_sticker(random.choice(self.stickers['positive']))
            elif overall_sentiment < 0:
                await self.send_sticker(random.choice(self.stickers['negative']))

            # Логируем время выполнения update_prices
            execution_time = time.time() - start_time
            logger.info(f"update_prices execution time: {execution_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error in update_prices: {e}\n{traceback.format_exc()}")

    def analyze_prices(self, symbol: str, current_price: float) -> str:
        if len(self.prices[symbol]) < 2:
            return "Недостаточно данных для анализа."

        prices = np.array(self.prices[symbol], dtype=float)

        # Расчет процентного изменения
        changes = np.diff(prices) / prices[:-1] * 100
        avg_change = np.mean(changes)

        # Добавим вывод для отладки
        # print(f"Среднее изменение для {symbol}: {avg_change:.4f}%")

        # Определение тренда
        trend = "восходящий" if avg_change > 0.05 else "нисходящий" if avg_change < -0.05 else "боковой"

        # Добавим проверку на более длительный период (если есть достаточно данных)
        if len(prices) >= 10:
            long_term_change = (prices[-1] - prices[-10]) / prices[-10] * 100
            if long_term_change > 0.5:
                trend = "восходящий"
            elif long_term_change < -0.5:
                trend = "нисходящий"

        # Расчет волатильности (стандартное отклонение изменений)
        volatility = np.std(changes)

        # Расчет скользящих средних
        ma5 = np.mean(prices[-min(5, len(prices)):])
        ma10 = np.mean(prices[-min(10, len(prices)):])

        # Формирование анализа и рекомендаций
        analysis = f"Тренд: {trend}. "
        analysis += f"Среднее изменение за последние периоды: {avg_change:.2f}%. "
        analysis += f"Волатильность: {volatility:.2f}%. "

        if current_price > ma5 > ma10:
            analysis += "Цена выше MA5 и MA10. Возможен продолжающийся рост. 🚀"
        elif current_price < ma5 < ma10:
            analysis += "Цена ниже MA5 и MA10. Возможно продолжение снижения. 💩 "
        elif ma5 > current_price > ma10:
            analysis += "Цена между MA5 и MA10. Возможна консолидация. "

        if volatility > 2:
            analysis += "Высокая волатильность. Будьте осторожны. 😵‍💫"
        elif volatility < 0.5:
            analysis += "Низкая волатильность. Возможен скорый всплеск активности. 🥳"

        if trend == "восходящий" and current_price > ma10:
            analysis += "Рекомендация: Рассмотрите покупку или удержание позиции. 😎"
        elif trend == "нисходящий" and current_price < ma10:
            analysis += "Рекомендация: Рассмотрите продажу или сокращение позиции. 😭"
        else:
            analysis += "Рекомендация: Наблюдайте за развитием ситуации. 🤓"

        return analysis

    def get_price_and_volume(self, symbol: str):
        ticker = {
            'last': 0,
            'quoteVolume': 0,
        }
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            open_interest = self.exchange.fetch_open_interest(symbol)['openInterestAmount']
            if open_interest is None:
                logger.warning(f"Open interest is None for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to fetch open interest for {symbol}: {e}")
            open_interest = 0
        return ticker['last'], ticker['quoteVolume'], open_interest

    def create_price_volume_oi_rci_chart(self):
        start_time = time.time()
        try:
            if len(self.timestamps) < 2:
                return None

            fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)  # 4 графика: цена, объем, OI, RCI

            # 1. Цены
            for symbol in self.symbols:
                prices = np.array(self.prices[symbol])
                if len(prices) != len(self.timestamps) or prices[0] is None:
                    continue
                norm_prices = (prices - prices[0]) / prices[0] * 100
                axes[0].plot(self.timestamps, norm_prices, color=self.symbol_colors[symbol], label=symbol)
            axes[0].set_ylabel('Изм. цены, %')
            axes[0].legend()
            axes[0].grid(True)

            # 2. Объемы
            for symbol in self.symbols:
                volumes = np.array(self.volumes[symbol])
                if len(volumes) != len(self.timestamps) or volumes[0] is None:
                    continue
                norm_volumes = (volumes - volumes[0]) / volumes[0] * 100
                axes[1].plot(self.timestamps, norm_volumes, color=self.symbol_colors[symbol], label=symbol)
            axes[1].set_ylabel('Изм. объема, %')
            axes[1].legend()
            axes[1].grid(True)

            # 3. Open Interest
            for symbol in self.symbols:
                oi = np.array(self.open_interest[symbol])
                valid_indices = [i for i, x in enumerate(oi) if x is not None]
                if not valid_indices:
                    continue
                oi = oi[valid_indices]
                valid_timestamps = [self.timestamps[i] for i in valid_indices]
                if len(oi) == 0 or oi[0] is None:
                    continue
                norm_oi = (oi - oi[0]) / oi[0] * 100
                axes[2].plot(valid_timestamps, norm_oi, color=self.symbol_colors[symbol], label=symbol)
            axes[2].set_ylabel('Изм. OI, %')
            axes[2].legend()
            axes[2].grid(True)

            # 4. RCI (Rank Correlation Index)
            def calc_rci(prices, period=9):
                if len(prices) < period:
                    return [None] * len(prices)
                rci = [None] * (period - 1)
                for i in range(period - 1, len(prices)):
                    window = prices[i - period + 1:i + 1]
                    rank_price = np.argsort(np.argsort(window))
                    rank_time = np.arange(period)
                    diff = rank_price - rank_time
                    rci_val = (1 - 6 * np.sum(diff ** 2) / (period * (period ** 2 - 1))) * 100
                    rci.append(rci_val)
                return rci

            for symbol in self.symbols:
                if symbol not in ['BTC/USDT', 'ETH/USDT']:
                    continue
                prices = np.array(self.prices[symbol])
                if len(prices) != len(self.timestamps) or prices[0] is None:
                    continue
                rci = calc_rci(prices)
                axes[3].plot(self.timestamps, rci, color=self.symbol_colors[symbol], label=symbol)
            axes[3].set_ylabel('RCI')
            axes[3].legend()
            axes[3].grid(True)

            axes[-1].set_xlabel('Время')
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            caption = f"Анализ: цена, объем, OI, RCI (BTC/ETH) за период {self.interval}s на {datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S')}"
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi)
            buf.seek(0)
            plt.close()
            return buf, caption
        finally:
            logger.info(f"Chart creation time: {time.time() - start_time:.2f} seconds")

    def create_indices_chart(self):
        start_time = time.time()
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

            colors = ['#2C3E50', '#8E44AD', '#F39C12', '#16A085', '#27AE60', '#C0392B']
            index_lines = {}

            for i, (index, color) in enumerate(zip(self.indices, colors)):
                values = np.array(self.indices[index]['values'])
                timestamps = self.indices[index]['timestamps']

                if len(values) > 0 and len(timestamps) > 0:
                    if index in ['Total_Market_Cap', 'Market_Cap_Change_24h']:
                        continue
                        # line, = ax.plot(timestamps, values, color=color, label=index)
                    else:
                        initial_value = values[0]
                        normalized_values = (values - initial_value) / initial_value * 100
                        line, = ax.plot(timestamps, normalized_values, color=color, label=index)

                    index_lines[index] = line

                    # Добавляем аннотации
                    if len(values) > 1:
                        ax.annotate(f'{values[-1]:.2f}',
                                    (timestamps[-1],
                                     values[-1] if index in ['Total_Market_Cap', 'Market_Cap_Change_24h'] else
                                     normalized_values[-1]),
                                    textcoords="offset points",
                                    xytext=(0, 5),
                                    ha='center',
                                    fontsize=8,
                                    color=color)
                else:
                    print(f"No data available for {index}")

            ax.legend(index_lines.values(), index_lines.keys(), loc='upper left')
            ax.set_xlabel('Время')
            ax.set_ylabel('Значение / Процентное изменение')
            ax.set_title('Изменение индексов')
            ax.grid(True)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi)
            buf.seek(0)
            plt.close(fig)

            return buf, 'Изменение индексов #indexes'

        finally:
            # Логируем время создания графика индексов
            execution_time = time.time() - start_time
            logger.info(f"Indices chart creation time: {execution_time:.2f} seconds")

    async def send_message(self, message: str, max_retries=3):
        for attempt in range(max_retries):
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=message)
                return
            except TimedOut:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Failed to send message after {max_retries} attempts due to timeout")
            except TelegramError as e:
                logger.error(f"Telegram error occurred: {e}")
                return

    async def send_chart(self, chart_info, max_retries=3):
        if not chart_info:
            return
        chart, caption = chart_info
        for attempt in range(max_retries):
            try:
                await self.bot.send_photo(chat_id=self.chat_id, photo=chart, caption=caption)
                return
            except TimedOut:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Failed to send chart after {max_retries} attempts due to timeout")
            except TelegramError as e:
                logger.error(f"Telegram error occurred: {e}")
                return

    async def run(self, interval: int = 300):
        while True:
            try:
                # Проверяем CPU перед каждой итерацией
                if not self.check_cpu_usage():
                    logger.info("Shutting down due to high CPU usage")
                    sys.exit(0)
                    
                await self.update_prices()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"An error occurred in run: {e}")
                await asyncio.sleep(10)

    async def send_sticker(self, sticker_id: str, max_retries=3):
        for attempt in range(max_retries):
            try:
                await self.bot.send_sticker(chat_id=self.chat_id, sticker=sticker_id)
                return
            except TimedOut:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Не удалось отправить стике после {max_retries} попыток из-за таймаута")
            except TelegramError as e:
                logger.error(f"Произошла ошибка Telegram: {e}")
                return


async def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    analyzer = CryptoAnalyzer(
        exchange_name=config['exchange_name'],
        symbols=config['symbols'],
        telegram=config['telegram'],
        db_config=config['db'],
        interval=config['update_interval'],
        stickers=config['stickers'],
        is_indexes=config['is_indexes'],
        timestamps_limit=config['timestamps_limit'],
    )

    await analyzer.run(interval=config['update_interval'])


if __name__ == "__main__":
    asyncio.run(main())
