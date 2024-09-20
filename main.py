import ccxt
import asyncio
from telegram import Bot
from telegram.error import TimedOut, TelegramError
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from datetime import datetime
import json
import logging
import math

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
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
    def __init__(self, exchange_name: str, symbols: list, telegram_token: str, chat_id: str):
        self.exchange = getattr(ccxt, exchange_name)()
        self.symbols = symbols
        self.bot = Bot(token=telegram_token)
        self.chat_id = chat_id
        self.prices = {symbol: [] for symbol in symbols}
        self.volumes = {symbol: [] for symbol in symbols}
        self.timestamps = []
        self.colors = plt.cm.rainbow(np.linspace(0, 1, len(symbols)))
        self.prev_prices = {symbol: None for symbol in symbols}

    def analyze_prices(self, symbol: str, current_price: float) -> str:
        if self.prev_prices[symbol] is None:
            self.prev_prices[symbol] = current_price
            return "Недостаточно данных для анализа."

        price_change = (current_price - self.prev_prices[symbol]) / self.prev_prices[symbol]

        if price_change > 0.02:
            analysis = "Цена значительно выросла! Рекомендация: Рассмотрите продажу."
        elif price_change < -0.02:
            analysis = "Цена значительно упала! Рекомендация: Рассмотрите покупку."
        elif 0.005 < price_change <= 0.02:
            analysis = "Наблюдается умеренный рост. Следите за трендом."
        elif -0.02 <= price_change < -0.005:
            analysis = "Наблюдается умеренное снижение. Следите за трендом."
        else:
            analysis = "Цена стабильна."

        self.prev_prices[symbol] = current_price
        return analysis

    def get_price_and_volume(self, symbol: str):
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last'], ticker['quoteVolume']

    async def update_prices(self):
        try:
            current_time = datetime.now()
            self.timestamps.append(current_time)

            message = ""
            for symbol in self.symbols:
                price, volume = self.get_price_and_volume(symbol)
                self.prices[symbol].append(price)
                self.volumes[symbol].append(volume)

                formatted_price = format_number(price)
                formatted_volume = format_number(volume)
                analysis = self.analyze_prices(symbol, price)
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

    def create_price_volume_chart(self):
        if len(self.timestamps) < 2:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        for i, symbol in enumerate(self.symbols):
            color = self.colors[i]
            
            # Нормализация цен
            prices = np.array(self.prices[symbol])
            initial_price = prices[0]
            normalized_prices = (prices - initial_price) / initial_price * 100

            # График цены
            line, = ax1.plot(self.timestamps, normalized_prices, color=color, label=f'{symbol} Цена')
            
            # Аннотации для цен
            for j, (timestamp, norm_price, price) in enumerate(zip(self.timestamps, normalized_prices, prices)):
                if j % 3 == 0 or j == len(prices) - 1:  # Аннотируем каждую 3-ю точку и последнюю
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
            ax2.plot(self.timestamps, normalized_volumes, color=color, label=f'{symbol} Объем')
            
            # Аннотации для объемов
            for j, (timestamp, norm_volume, volume) in enumerate(zip(self.timestamps, normalized_volumes, volumes)):
                if j % 3 == 0 or j == len(volumes) - 1:  # Аннотируем каждую 3-ю точку и последнюю
                    ax2.annotate(f'{format_number(volume)}', 
                                 xy=(timestamp, norm_volume),
                                 xytext=(0, 5), textcoords='offset points',
                                 ha='center', va='bottom', color=color,
                                 fontsize=8, rotation=45)

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
    )
    await analyzer.run(interval=config['update_interval'])


if __name__ == "__main__":
    asyncio.run(main())
