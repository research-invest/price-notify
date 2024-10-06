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

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

timezone = ZoneInfo("Europe/Moscow")


class CryptoAnalyzer:
    def __init__(self, exchange_name: str, symbols: list, telegram: dict, db_config: dict, interval: int,
                 stickers: dict):
        self.dpi = 140
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
        self.timestamps = []
        self.db_manager = DatabaseManager(db_config)
        self.db_manager.connect()
        self.db_manager.create_tables()
        self.load_historical_data()
        self.last_indices_update = datetime.now(timezone)
        self.stickers = stickers

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
                # Обновляем последнюю запись, если время совпадает
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

            message += f"S&P 500: {latest_sp500:.2f}\n\n"

            if len(self.timestamps) > 100:
                self.timestamps = self.timestamps[-100:]
                for symbol in self.symbols:
                    self.prices[symbol] = self.prices[symbol][-100:]
                    self.volumes[symbol] = self.volumes[symbol][-100:]

            await self.send_message(message)

            price_volume_chart = self.create_price_volume_chart()
            await self.send_chart(price_volume_chart)

            # Отправляем график индексов раз в час
            current_time = datetime.now(timezone)
            if current_time.minute == 0 and (current_time - self.last_indices_update).total_seconds() >= 3600:
                indices_chart = self.create_indices_chart()
                await self.send_chart(indices_chart)
                self.last_indices_update = current_time.replace(minute=0, second=0, microsecond=0)

            # Отправка стикера на основе общего настроения
            if overall_sentiment > 0:
                await self.send_sticker(random.choice(self.stickers['positive']))
            elif overall_sentiment < 0:
                await self.send_sticker(random.choice(self.stickers['negative']))


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

        # Добавим проверку на боле длительный период (если есть достаточно данных)
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
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last'], ticker['quoteVolume']

    def create_price_volume_chart(self):
        if len(self.timestamps) < 2:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        formatted_date_time = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")

        caption = f"Анализ цен и объемов торгов за период {self.interval}s на {formatted_date_time}"

        fig.suptitle(caption, fontsize=18)

        # Функция для полиномиальной аппроксимации
        def poly_func(x, a, b, c):
            return a * x ** 2 + b * x + c

        # Интервал аннотаций
        total_points = len(self.timestamps)
        points_per_annotation = max(1, int(total_points / 10))
        min_interval_seconds = 300
        annotation_interval = max(points_per_annotation, int(min_interval_seconds / self.interval))

        # Создаем пустые списки для хранения линий легенды
        price_lines = []
        volume_lines = []
        price_labels = []
        volume_labels = []

        alpha = 0.35

        def add_slope_annotation(ax, x, y, slope, color):
            angle = math.degrees(math.atan(slope))
            ax.annotate(f'{angle:.1f}°', 
                        xy=(x[-1], y[-1]), 
                        xytext=(5, 0), 
                        textcoords='offset points',
                        color=color,
                        fontsize=10,
                        ha='left', 
                        va='center')

        for symbol in self.symbols:
            if len(self.prices[symbol]) != len(self.timestamps):
                logger.warning(f"Несоответствие данных для {symbol}. Пропуск построения графика.")
                continue

            color = self.symbol_colors[symbol]
            line_width = self.symbol_line_widths[symbol]

            # Нормализация цен
            prices = np.array(self.prices[symbol])
            initial_price = prices[0]
            if initial_price is None:
                continue

            normalized_prices = (prices - initial_price) / initial_price * 100

            # График цены
            price_line, = ax1.plot(self.timestamps, normalized_prices, color=color, label=f'{symbol} Цена',
                                   linewidth=line_width, linestyle='-')
            price_lines.append(price_line)
            price_labels.append(f'{symbol} Цена')

            # Добавляем линейную регрессию для цен
            x = mdates.date2num(self.timestamps)
            x_scaled = (x - x[0]) / (x[-1] - x[0])  # Нормализуем x к диапазону [0, 1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_scaled, normalized_prices)
            line = slope * x_scaled + intercept
            reg_line, = ax1.plot(self.timestamps, line, color=color, linestyle='--', linewidth=1.5, alpha=alpha)
            add_slope_annotation(ax1, x, line, slope, color)
            
            # Добавляем полиномиальную аппроксимацию для цен
            try:
                popt, _ = curve_fit(poly_func, x_scaled, normalized_prices)
                x_line = np.linspace(0, 1, 100)
                y_line = poly_func(x_line, *popt)
                approx_line, = ax1.plot(mdates.num2date(x[0] + x_line * (x[-1] - x[0])), y_line, color=color, linestyle=':', linewidth=2, alpha=alpha)
                # Вычисляем наклон в последней точке полиномиальной аппроксимации
                poly_slope = 2 * popt[0] * x_line[-1] + popt[1]
                add_slope_annotation(ax1, x, y_line, poly_slope, color)
            except Exception as e:
                print(f"Ошибка при расчете полиномиальной аппроксимации для {symbol} (цены): {e}")
                approx_line = None

            # Аннотации для цен
            for j, (timestamp, norm_price, price) in enumerate(zip(self.timestamps, normalized_prices, prices)):
                if j % annotation_interval == 0 or j == len(prices) - 1:
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
            volume_line, = ax2.plot(self.timestamps, normalized_volumes, color=color, label=f'{symbol} Объем',
                                    linewidth=line_width, linestyle='-')
            volume_lines.append(volume_line)
            volume_labels.append(f'{symbol} Объем')

            # Добавляем линейную регрессию для объемов
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_scaled, normalized_volumes)
            line = slope * x_scaled + intercept
            ax2.plot(self.timestamps, line, color=color, linestyle='--', linewidth=1.5, alpha=alpha)
            add_slope_annotation(ax2, x, line, slope, color)

            # Добавляем полиномиальную аппроксимацию для объемов
            try:
                popt, _ = curve_fit(poly_func, x_scaled, normalized_volumes)
                y_line = poly_func(x_line, *popt)
                ax2.plot(mdates.num2date(x[0] + x_line * (x[-1] - x[0])), y_line, color=color, linestyle=':', linewidth=2, alpha=alpha)
                # Вычисляем наклон в последней точке полиномиальной аппроксимации
                poly_slope = 2 * popt[0] * x_line[-1] + popt[1]
                add_slope_annotation(ax2, x, y_line, poly_slope, color)
            except Exception as e:
                print(f"Ошибка при расчете полиномиальной аппроксимации для {symbol} (объемы): {e}")

            # Аннотации для объемов
            for j, (timestamp, norm_volume, volume) in enumerate(zip(self.timestamps, normalized_volumes, volumes)):
                if j % annotation_interval == 0 or j == len(volumes) - 1:
                    ax2.annotate(f'{format_number(volume)}',
                                 xy=(timestamp, norm_volume),
                                 xytext=(0, 5), textcoords='offset points',
                                 ha='center', va='bottom', color=color,
                                 fontsize=8, rotation=45)

        # Добавляем линии регрессии и аппроксимации в легенду только один раз
        price_lines.extend([ax1.plot([], [], color='gray', linestyle='--', linewidth=1.5)[0],
                            ax1.plot([], [], color='gray', linestyle=':', linewidth=1)[0]])
        price_labels.extend(['Линейная регрессия', 'Полиномиальная аппроксимация'])
        
        # Для объемов добавляем только метки, так как линии уже добавлены
        volume_labels.extend(['Линейная регрессия', 'Полиномиальная аппроксимация'])

        # Устанавливаем легенды
        ax1.legend(price_lines, price_labels, loc='upper left', fontsize=8)
        ax2.legend(volume_lines + [ax2.plot([], [], color='gray', linestyle='--', linewidth=1.5)[0],
                                   ax2.plot([], [], color='gray', linestyle=':', linewidth=1)[0]], 
                   volume_labels, loc='upper left', fontsize=8)

        ax1.set_ylabel('Процентное изменение цены')
        ax1.grid(True)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
        ax1.axhline(y=0, color='gray', linestyle='--')

        ax2.set_xlabel('Время')
        ax2.set_ylabel('Процентное изменение объема')
        ax2.grid(True)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
        ax2.axhline(y=0, color='gray', linestyle='--')

        plt.gcf().autofmt_xdate()
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if not os.path.exists('render'):
            os.makedirs('render')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi)
        plt.savefig('render/graph.png', format='png', dpi=self.dpi)
        buf.seek(0)
        plt.close()

        return buf, caption

    def create_indices_chart(self):
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
        stickers=config['stickers']
    )

    await analyzer.run(interval=config['update_interval'])


if __name__ == "__main__":
    asyncio.run(main())
