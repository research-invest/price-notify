import ccxt
import asyncio
from telegram import Bot
from telegram.error import TimedOut, TelegramError
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import math
from database import DatabaseManager

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_number(number):
    if number >= 1 or number == 0:
        return f"{number:,.2f}".replace(',', ' ')
    elif number >= 0.0001:
        return f"{number:.4f}"
    else:
        if number > 0:
            significant_digits = -math.floor(math.log10(number)) + 2
            format_string = f"{{:.{significant_digits}f}}"
            return format_string.format(number)
        else:
            return "0.00"  # Для отрицательных чисел или нуля


class CryptoAnalyzer:
    def __init__(self, exchange_name: str, symbols: list, telegram_token: str, chat_id: str, db_config: dict):
        self.exchange = getattr(ccxt, exchange_name)()
        self.symbols = symbols
        self.bot = Bot(token=telegram_token)
        self.chat_id = chat_id
        self.prices = {symbol: [] for symbol in symbols}
        self.volumes = {symbol: [] for symbol in symbols}
        self.timestamps = []
        self.colors = plt.cm.rainbow(np.linspace(0, 1, len(symbols)))
        self.db_manager = DatabaseManager(db_config)
        self.db_manager.connect()
        self.db_manager.create_tables()
        self.load_historical_data()

    def load_historical_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        all_timestamps = set()
        
        # Сначала соберем все уникальные временные метки
        for symbol in self.symbols:
            historical_data = self.db_manager.get_historical_data(symbol, start_date, end_date)
            all_timestamps.update(row[0] for row in historical_data)
        
        # Отсортируем временные метки
        self.timestamps = sorted(all_timestamps)
        
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

            message = ''
            for symbol in self.symbols:
                formatted_price = format_number(self.prices[symbol][-1])
                formatted_volume = format_number(self.volumes[symbol][-1])
                analysis = self.analyze_prices(symbol, self.prices[symbol][-1])
                message += f"{symbol}:\nЦена: {formatted_price}\nОбъем: {formatted_volume}\nАнализ: {analysis}\n\n"

            if len(self.timestamps) > 100:
                self.timestamps = self.timestamps[-100:]
                for symbol in self.symbols:
                    self.prices[symbol] = self.prices[symbol][-100:]
                    self.volumes[symbol] = self.volumes[symbol][-100:]

            await self.send_message(message)

            price_volume_chart = self.create_price_volume_chart()
            await self.send_chart(price_volume_chart)

        except Exception as e:
            logger.error(f"An error occurred in update_prices: {e}")

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
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        for i, symbol in enumerate(self.symbols):
            if len(self.prices[symbol]) != len(self.timestamps):
                logger.warning(f"Несоответствие данных для {symbol}. Пропуск построения графика.")
                continue

            color = self.colors[i]
            
            # Нормализация цен
            prices = np.array(self.prices[symbol])
            initial_price = prices[0]
            normalized_prices = (prices - initial_price) / initial_price * 100

            # График цены
            ax1.plot(self.timestamps, normalized_prices, color=color, label=f'{symbol} Цена')
            
            # Нормализация объемов
            volumes = np.array(self.volumes[symbol])
            initial_volume = volumes[0]
            normalized_volumes = (volumes - initial_volume) / initial_volume * 100
            
            # График объема
            ax2.plot(self.timestamps, normalized_volumes, color=color, label=f'{symbol} Объем')

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

        plt.gcf().autofmt_xdate()
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)  # Увеличиваем DPI для лучшего качества
        buf.seek(0)
        plt.close()

        return buf

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

    async def send_chart(self, chart, max_retries=3):
        if not chart:
            return
        for attempt in range(max_retries):
            try:
                await self.bot.send_photo(chat_id=self.chat_id, photo=chart)
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
        telegram_token=config['telegram_token'],
        chat_id=config['chat_id'],
         db_config=config['db']
    )
    await analyzer.run(interval=config['update_interval'])


if __name__ == "__main__":
    asyncio.run(main())
