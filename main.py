import os
import asyncio
from telegram import Bot
from telegram.error import TimedOut, TelegramError
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import json
import logging
import math
from utils import format_number
import traceback
from matplotlib.dates import DateFormatter
from pycoingecko import CoinGeckoAPI
import requests
import matplotlib.dates as mdates
import random
from scipy import stats
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# timezone = ZoneInfo("Europe/Moscow")
timezone = ZoneInfo("UTC")


class CryptoAnalyzer:
    def __init__(self, symbols: list, telegram: dict, intervals: dict,
                 stickers: dict, tickers_api_url: str):
        self.dpi = 140
        self.symbols = [symbol['name'] for symbol in symbols]
        self.symbol_colors = {symbol['name']: symbol['color'] for symbol in symbols}
        self.symbol_line_widths = {symbol['name']: symbol.get('line_width', 1) for symbol in symbols}
        self.bot = Bot(token=telegram['token'])
        self.chat_id = telegram['chat_id']
        self.intervals = intervals
        self.prices = {symbol: [] for symbol in self.symbols}
        self.volumes = {symbol: [] for symbol in self.symbols}
        self.timestamps = []
        self.stickers = stickers
        self.tickers_api_url = tickers_api_url
        self.last_update = {interval: datetime.now(timezone) for interval in intervals}

    def get_price_and_volume_data(self, symbol: str, interval: int):
        try:
            url = f"{self.tickers_api_url}/tickers?symbol={symbol}&interval={interval}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise ValueError(f"Нет данных для символа {symbol}")
            
            return data
        except requests.RequestException as e:
            logging.error(f"Ошибка при получении данных для {symbol}: {e}")
            return None
        except (KeyError, IndexError) as e:
            logging.error(f"Ошибка в формате данных для {symbol}: {e}")
            return None

    async def update_prices(self):
        try:
            current_time = datetime.now(timezone)

            for interval_name, interval_seconds in self.intervals.items():
                if current_time - self.last_update[interval_name] >= timedelta(seconds=interval_seconds):
                    logger.info(f"Обновление для интервала {interval_name}")
                    
                    all_data = {}
                    message = ''
                    for symbol in self.symbols:
                        data = self.get_price_and_volume_data(symbol, interval_seconds)
                        if data:
                            all_data[symbol] = data
                            self.prices[symbol] = [item['last_price'] for item in data]
                            self.volumes[symbol] = [item['volume'] for item in data]
                            self.timestamps = [datetime.fromisoformat(item['timestamp'].rstrip('Z')).replace(tzinfo=timezone) for item in data]

                            current_price = self.prices[symbol][-1]
                            formatted_price = format_number(current_price)
                            formatted_volume = format_number(self.volumes[symbol][-1])
                            analysis = self.analyze_prices(symbol, current_price)
                            message += f"{symbol}:\nЦена: {formatted_price}\nОбъем: {formatted_volume}\nАнализ: {analysis}\n\n"
                        else:
                            logger.warning(f"Не удалось получить данные для {symbol}")

                    if all_data:
                        await self.send_message(message)

                        price_volume_chart = self.create_price_volume_chart(interval_name, interval_seconds)
                        await self.send_chart(price_volume_chart)

                        # Отправляем стикер
                        await self.send_sticker(self.choose_sticker(all_data))
                    
                    self.last_update[interval_name] = current_time

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

    def create_price_volume_chart(self, interval_name, interval_seconds):
        if len(self.timestamps) < 2:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        formatted_date_time = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")

        caption = f"Анализ цен и объемов торгов за период {interval_name} на {formatted_date_time} #{interval_name}"

        fig.suptitle(caption, fontsize=18)

        # Функция для полиномиальной аппроксимации
        def poly_func(x, a, b, c):
            return a * x ** 2 + b * x + c

        # Интервал аннотаций
        total_points = len(self.timestamps)
        points_per_annotation = max(1, int(total_points / 10))
        min_interval_seconds = 300
        annotation_interval = max(points_per_annotation, int(min_interval_seconds / interval_seconds))

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
                approx_line, = ax1.plot(mdates.num2date(x[0] + x_line * (x[-1] - x[0])), y_line, color=color,
                                        linestyle=':', linewidth=2, alpha=alpha)
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
                ax2.plot(mdates.num2date(x[0] + x_line * (x[-1] - x[0])), y_line, color=color, linestyle=':',
                         linewidth=2, alpha=alpha)
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

        # Добавляем линии регрессии и аппроксимации в лег��нду только один раз
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

        # Добавляем линию доминации ETH/BTC на график цен
        if 'ETH/USDT' in self.prices and 'BTC/USDT' in self.prices:
            eth_prices = np.array(self.prices['ETH/USDT'])
            btc_prices = np.array(self.prices['BTC/USDT'])
            
            # Вычисляем относительную силу ETH к BTC
            eth_normalized = (eth_prices - eth_prices[0]) / eth_prices[0] * 100
            btc_normalized = (btc_prices - btc_prices[0]) / btc_prices[0] * 100
            dominance = eth_normalized - btc_normalized
            
            # Добавляем линию доминации на график цен
            dom_line, = ax1.plot(self.timestamps, dominance, color='purple', 
                                label='ETH/BTC Доминация', linewidth=1.5, 
                                linestyle='--', alpha=0.8)
            
            # Добавляем аннотации для значимых точек доминации
            for j, (timestamp, dom) in enumerate(zip(self.timestamps, dominance)):
                if j % annotation_interval == 0 or j == len(dominance) - 1:
                    ax1.annotate(f'D:{dom:.1f}%',
                                xy=(timestamp, dom),
                                xytext=(0, -10), textcoords='offset points',
                                ha='center', va='top', color='purple',
                                fontsize=8, rotation=45)

            # Добавляем индикатор текущей доминации в легенду
            if dominance[-1] > 0:
                dom_status = f'ETH/BTC: +{dominance[-1]:.1f}% (ETH сильнее)'
            else:
                dom_status = f'ETH/BTC: {dominance[-1]:.1f}% (BTC сильнее)'
            
            # Обновляем списки для легенды
            price_lines.append(dom_line)
            price_labels.append(dom_status)

        # Обновляем легенду с новыми элементами
        ax1.legend(price_lines, price_labels, loc='upper left', fontsize=8)

        # Перемещаем код сохранения графика в конец
        if not os.path.exists('render'):
            os.makedirs('render')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi)
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

    def choose_sticker(self, all_data):
        # Простая логика выбора стикера на основе изменения цены
        total_change = 0
        for symbol, data in all_data.items():
            if len(data) >= 2:
                change = (data[-1]['last_price'] - data[0]['last_price']) / data[0]['last_price']
                total_change += change

        if total_change > 0:
            return random.choice(self.stickers['positive'])
        else:
            return random.choice(self.stickers['negative'])

    async def send_sticker(self, sticker_id):
        try:
            await self.bot.send_sticker(chat_id=self.chat_id, sticker=sticker_id)
        except Exception as e:
            logger.error(f"Ошибка при отправке стикера: {e}")

    async def run(self):
        while True:
            await self.update_prices()
            await asyncio.sleep(min(self.intervals.values()))  # Ждем минимальный интервал перед следующим обновлением


async def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    analyzer = CryptoAnalyzer(
        symbols=config['symbols'],
        telegram=config['telegram'],
        intervals=config['intervals'],
        stickers=config['stickers'],
        tickers_api_url=config['tickers_api_url']
    )

    await analyzer.run()

if __name__ == "__main__":
    asyncio.run(main())
