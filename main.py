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

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

timezone = ZoneInfo("Europe/Moscow")


class CryptoAnalyzer:
    def __init__(self, exchange_name: str, symbols: list, colors: list, telegram_token: str, chat_id: str,
                 db_config: dict, interval: int):
        self.dpi = 100
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
        self.symbols = symbols
        self.bot = Bot(token=telegram_token)
        self.chat_id = chat_id
        self.interval = interval
        self.prices = {symbol: [] for symbol in symbols}
        self.volumes = {symbol: [] for symbol in symbols}
        self.timestamps = []
        self.colors = colors
        # self.colors = plt.cm.rainbow(np.linspace(0, 1, len(symbols)))
        self.db_manager = DatabaseManager(db_config)
        self.db_manager.connect()
        self.db_manager.create_tables()
        self.load_historical_data()
        self.index_lines = {}

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
            data_dict = {row[0]: (float(row[1]), float(row[2])) for row in historical_data}

            self.prices[symbol] = []
            self.volumes[symbol] = []

            for timestamp in self.timestamps:
                if timestamp in data_dict:
                    self.prices[symbol].append(data_dict[timestamp][0])
                    self.volumes[symbol].append(data_dict[timestamp][1])
                else:
                    # Если данных нет, используем предыдущее значение или None
                    self.prices[symbol].append(self.prices[symbol][-1] if self.prices[symbol] else None)
                    self.volumes[symbol].append(self.volumes[symbol][-1] if self.volumes[symbol] else None)

            logger.info(f"Загружено {len(historical_data)} исторических записей для {symbol}")

        # Загружаем исторические данные S&P 500
        # sp500_data = self.db_manager.get_historical_data('SP500', start_date, end_date)
        # self.sp500_prices = [float(price) for _, price, _ in sp500_data]
        # self.sp500_timestamps = [timestamp for timestamp, _, _ in sp500_data]

    async def update_prices(self):
        try:
            current_time = datetime.now()

            # Обновляем данные S&P 500
            sp500 = yf.Ticker("^GSPC")
            latest_sp500 = float(sp500.history(period="1d")['Close'].iloc[-1])
            self.db_manager.save_price_data('SP500', current_time, latest_sp500, 0)  # объем 0 для S&P 500
            self.indices['SP500']['values'].append(latest_sp500)
            self.indices['SP500']['timestamps'].append(current_time)

            # Обновляем индекс страха и жадности
            fear_greed_data = requests.get('https://api.alternative.me/fng/').json()
            fear_greed_value = int(fear_greed_data['data'][0]['value'])
            self.db_manager.save_price_data('Fear&Greed', current_time, fear_greed_value, 0)  # объем 0 для Fear&Greed
            self.indices['Fear&Greed']['values'].append(fear_greed_value)
            self.indices['Fear&Greed']['timestamps'].append(current_time)

            # Обновляем индекс доминирования биткоина
            bitcoin_data = self.cg.get_global()
            btc_dominance = float(bitcoin_data['market_cap_percentage']['btc'])
            self.db_manager.save_price_data('BTC_Dominance', current_time, btc_dominance,
                                            0)  # объем 0 для BTC_Dominance
            self.indices['BTC_Dominance']['values'].append(btc_dominance)
            self.indices['BTC_Dominance']['timestamps'].append(current_time)

            # Добавим также общую капитализацию рынка
            total_market_cap = float(bitcoin_data['total_market_cap']['usd'])

            self.db_manager.save_price_data('Total_Market_Cap', current_time, total_market_cap,
                                            0)  # объем 0 для Total_Market_Cap
            self.indices['Total_Market_Cap']['values'].append(total_market_cap)
            self.indices['Total_Market_Cap']['timestamps'].append(current_time)

            # И изменение капитализации за 24 часа
            market_cap_change_24h = float(bitcoin_data['market_cap_change_percentage_24h_usd'])
            self.db_manager.save_price_data('Total_Market_Cap', current_time, market_cap_change_24h,
                                            0)  # объем 0 для Market_Cap_Change_24h
            self.indices['Market_Cap_Change_24h']['values'].append(market_cap_change_24h)
            self.indices['Market_Cap_Change_24h']['timestamps'].append(current_time)

            # Обновляем NASDAQ-100
            nasdaq = yf.Ticker("^NDX")
            latest_nasdaq = float(nasdaq.history(period="1d")['Close'].iloc[-1])
            self.db_manager.save_price_data('NASDAQ', current_time, latest_nasdaq, 0)  # объем 0 для NASDAQ
            self.indices['NASDAQ']['values'].append(latest_nasdaq)
            self.indices['NASDAQ']['timestamps'].append(current_time)

            if not self.timestamps or current_time > self.timestamps[-1]:
                self.timestamps.append(current_time)
                for symbol in self.symbols:
                    price, volume = self.get_price_and_volume(symbol)
                    price = float(price)
                    volume = float(volume)

                    self.prices[symbol].append(price)
                    self.volumes[symbol].append(volume)

                    # Сохранение данных в БД
                    try:
                        self.db_manager.save_price_data(symbol, current_time, price, volume)
                    except Exception as db_error:
                        logger.error(f"Ошибка при сохранении данных в БД: {db_error}")
            else:
                # Обновля��м последнюю запись, если время совпадает
                for symbol in self.symbols:
                    price, volume = self.get_price_and_volume(symbol)
                    price = float(price)
                    volume = float(volume)

                    self.prices[symbol][-1] = price
                    self.volumes[symbol][-1] = volume
                    # Обновляем запись в БД
                    try:
                        self.db_manager.update_price_data(symbol, current_time, price, volume)
                    except Exception as db_error:
                        logger.error(f"Ошибка при обновлении данных в БД: {db_error}")

            # Тихий режим
            # if 2 <= datetime.now(timezone).hour < 6:
            #     return

            message = ''
            for symbol in self.symbols:
                formatted_price = format_number(self.prices[symbol][-1])
                formatted_volume = format_number(self.volumes[symbol][-1])
                analysis = self.analyze_prices(symbol, self.prices[symbol][-1])
                message += f"{symbol}:\nЦена: {formatted_price}\nОбъем: {formatted_volume}\nАнализ: {analysis}\n\n"

            message += f"S&P 500: {latest_sp500:.2f}\n\n"

            if len(self.timestamps) > 100:
                self.timestamps = self.timestamps[-100:]
                for symbol in self.symbols:
                    self.prices[symbol] = self.prices[symbol][-100:]
                    self.volumes[symbol] = self.volumes[symbol][-100:]

            await self.send_message(message)

            price_volume_chart = self.create_price_volume_chart()
            await self.send_chart(price_volume_chart)

        except Exception as e:
            logger.error(f"Error in update_prices: {e}\n{traceback.format_exc()}")

    def analyze_prices(self, symbol: str, current_price: float) -> str:
        if len(self.prices[symbol]) < 2:
            return "Недостаточно данных для анализа."

        prices = np.array(self.prices[symbol], dtype=float)

        # Расчет процентного изменения
        changes = np.diff(prices) / prices[:-1] * 100
        avg_change = np.mean(changes)

        # Определение тренда
        trend = "восходящий" if avg_change > 0.5 else "нисходящий" if avg_change < -0.5 else "боковой"

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
            analysis += "Цена выше MA5 и MA10. Возможен продолжающийся рост. "
        elif current_price < ma5 < ma10:
            analysis += "Цена ниже MA5 и MA10. Возможно продолжение снижения. "
        elif ma5 > current_price > ma10:
            analysis += "Цена между MA5 и MA10. Возможна консолидация. "

        if volatility > 2:
            analysis += "Высокая волатильность. Будьте осторожны. "
        elif volatility < 0.5:
            analysis += "Низкая волатильность. Возможен скорый всплеск активности. "

        if trend == "восходящий" and current_price > ma10:
            analysis += "Рекомендация: Рассмотрите покупку или удержание позиции. "
        elif trend == "нисходящий" and current_price < ma10:
            analysis += "Рекомендация: Рассмотрите продажу или сокращение позиции. "
        else:
            analysis += "Рекомендация: Наблюдайте за развитием ситуации. "

        return analysis

    def get_price_and_volume(self, symbol: str):
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last'], ticker['quoteVolume']

    def create_price_volume_chart(self):
        if len(self.timestamps) < 2:
            print("Not enough data to create chart")
            return None

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True, constrained_layout=True)
        formatted_date_time = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")

        caption = f"Анализ цен и объемов торгов за период {self.interval}s на {formatted_date_time}"

        fig.suptitle(caption, fontsize=13)

        # Интервал аннотаций
        total_points = len(self.timestamps)
        points_per_annotation = max(1, int(total_points / 10))  # аннотация на каждой 10-й точке
        min_interval_seconds = 300  # Минимальный интервал в секундах между аннотациями
        annotation_interval = max(points_per_annotation, int(min_interval_seconds / self.interval))

        for i, symbol in enumerate(self.symbols):
            if len(self.prices[symbol]) != len(self.timestamps):
                logger.warning(f"Несоответствие данных для {symbol}. Пропуск построения графика.")
                continue

            color = self.colors[i]

            # Нормализация цен
            prices = np.array(self.prices[symbol])
            initial_price = prices[0]
            if initial_price is None:
                continue

            normalized_prices = (prices - initial_price) / initial_price * 100

            # График цены
            linewidth = 2.3 if i == 0 else 1  # Делаем линию Bitcoin толще
            linestyle = '-'  # if i == 0 else '--'
            line, = ax1.plot(self.timestamps, normalized_prices, color=color, label=f'{symbol} Цена',
                             linewidth=linewidth, linestyle=linestyle)

            # Аннотации для цен
            for j, (timestamp, norm_price, price) in enumerate(zip(self.timestamps, normalized_prices, prices)):
                if j % annotation_interval == 0 or j == len(
                        prices) - 1:  # Аннотируем каждую 3-ю точку и последнюю %j % 3 == 0 or
                    ax1.annotate(f'{format_number(price)}',
                                 xy=(timestamp, norm_price),
                                 xytext=(0, 5), textcoords='offset points',
                                 ha='center', va='bottom', color=color,
                                 fontsize=8, rotation=45)

            # Нормализация объемов
            volumes = np.array(self.volumes[symbol])
            initial_volume = volumes[0]
            normalized_volumes = (volumes - initial_volume) / initial_volume * 100

            # График объема
            ax2.plot(self.timestamps, normalized_volumes, color=color, label=f'{symbol} Объем',
                     linewidth=linewidth, linestyle=linestyle)

            # Аннотации для объемов
            for j, (timestamp, norm_volume, volume) in enumerate(zip(self.timestamps, normalized_volumes, volumes)):
                if j % annotation_interval == 0 or j == len(
                        volumes) - 1:  # Аннотируем каждую 3-ю точку и последнюю j % 3 == 0 or
                    ax2.annotate(f'{format_number(volume)}',
                                 xy=(timestamp, norm_volume),
                                 xytext=(0, 5), textcoords='offset points',
                                 ha='center', va='bottom', color=color,
                                 fontsize=8, rotation=45)

        colors = ['#2C3E50', '#8E44AD', '#F39C12', '#16A085', '#27AE60', '#C0392B']
        for i, (index, color) in enumerate(zip(self.indices, colors)):
            values = np.array(self.indices[index]['values'])
            timestamps = self.indices[index]['timestamps']

            if len(values) > 0 and len(timestamps) > 0:
                if index in ['Total_Market_Cap', 'Market_Cap_Change_24h']:
                    # Для этих индексов мы не нормализуем значения
                    line, = ax3.plot(timestamps, values, color=color, label=index)
                else:
                    # Для остальных индексов нормализуем значения
                    initial_value = values[0]
                    normalized_values = (values - initial_value) / initial_value * 100
                    line, = ax3.plot(timestamps, normalized_values, color=color, label=index)
                
                self.index_lines[index] = line

                # Добавляем аннотации
                if len(values) > 1:
                    ax3.annotate(f'{values[0]:.2f}', 
                                 (timestamps[0], values[0] if index in ['Total_Market_Cap', 'Market_Cap_Change_24h'] else normalized_values[0]),
                                 textcoords="offset points", 
                                 xytext=(0,10 + i*20), 
                                 ha='center',
                                 fontsize=8,
                                 color=color)
                    
                    ax3.annotate(f'{values[-1]:.2f}', 
                                 (timestamps[-1], values[-1] if index in ['Total_Market_Cap', 'Market_Cap_Change_24h'] else normalized_values[-1]),
                                 textcoords="offset points", 
                                 xytext=(0,10 + i*20), 
                                 ha='center',
                                 fontsize=8,
                                 color=color)
            else:
                print(f"No data available for {index}")

        ax1.set_ylabel('Процентное изменение цены')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
        ax1.axhline(y=0, color='gray', linestyle='--')

        ax2.set_xlabel('Время')
        ax2.set_ylabel('Процентное изменение объема')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
        ax2.axhline(y=0, color='gray', linestyle='--')

        ax3.set_xlabel('Время')
        ax3.set_ylabel('Процентное изменение')
        ax3.legend(self.index_lines.values(), self.index_lines.keys(), loc='upper left')
        ax3.grid(True)
        ax3.title.set_text('Изменение индекса S&P 500')

        # Настройка форматирования оси X для всех подграфиков
        for ax in (ax1, ax2, ax3):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Настройка общей оси X
        fig.align_xlabels()

        if not os.path.exists('render'):
            os.makedirs('render')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi)  # Увеличиваем DPI для лучшего качества
        plt.savefig('render/graph.png', format='png', dpi=self.dpi)
        buf.seek(0)
        plt.close()

        return buf, caption

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
                await self.update_prices()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"An error occurred in run: {e}")
                await asyncio.sleep(10)  # Ждем 10 секунд перед повторной попыткой в случае ошибки


async def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    analyzer = CryptoAnalyzer(
        exchange_name=config['exchange_name'],
        symbols=config['symbols'],
        colors=config['colors'],
        telegram_token=config['telegram_token'],
        chat_id=config['chat_id'],
        db_config=config['db'],
        interval=config['update_interval'],
    )
    await analyzer.run(interval=config['update_interval'])


if __name__ == "__main__":
    asyncio.run(main())