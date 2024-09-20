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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    def __init__(self, exchange_name: str, symbols: list, telegram_token: str, chat_id: str, correlation_interval: int):
        self.exchange = getattr(ccxt, exchange_name)()
        self.symbols = symbols
        self.bot = Bot(token=telegram_token)
        self.chat_id = chat_id
        self.correlation_interval = correlation_interval
        self.prices = {symbol: [] for symbol in symbols}
        self.volumes = {symbol: [] for symbol in symbols}
        self.timestamps = []
        self.colors = plt.cm.rainbow(np.linspace(0, 1, len(symbols)))

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
                message += f"{symbol}:\nЦена: {formatted_price}\nОбъем: {formatted_volume}\n\n"

            if len(self.timestamps) > 100:
                self.timestamps = self.timestamps[-100:]
                for symbol in self.symbols:
                    self.prices[symbol] = self.prices[symbol][-100:]
                    self.volumes[symbol] = self.volumes[symbol][-100:]

            await self.send_message(message)

            price_volume_chart = self.create_price_volume_chart()
            await self.send_chart(price_volume_chart)

            if len(self.timestamps) % self.correlation_interval == 0:
                correlation_chart = self.create_correlation_chart()
                await self.send_chart(correlation_chart)

        except Exception as e:
            logger.error(f"An error occurred in update_prices: {e}")

    def create_price_volume_chart(self):
        if len(self.timestamps) < 2:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        for i, symbol in enumerate(self.symbols):
            color = self.colors[i]
            ax1.plot(self.timestamps, self.prices[symbol], color=color, label=f'{symbol} Цена')
            ax2.plot(self.timestamps, self.volumes[symbol], color=color, label=f'{symbol} Объем')

        ax1.set_ylabel('Цена')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number(x)))

        ax2.set_xlabel('Время')
        ax2.set_ylabel('Объем')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number(x)))

        plt.gcf().autofmt_xdate()
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return buf

    def create_correlation_chart(self):
        if len(self.timestamps) < 2:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, symbol in enumerate(self.symbols):
            prices = np.array(self.prices[symbol])
            initial_price = prices[0]
            normalized_prices = (prices - initial_price) / initial_price * 100  # процентное изменение
            line, = ax.plot(self.timestamps, normalized_prices, color=self.colors[i], label=symbol)
            
            # Добавляем аннотацию с текущей ценой
            current_price = prices[-1]
            ax.annotate(f'{symbol}: {current_price:.2f}', 
                        xy=(self.timestamps[-1], normalized_prices[-1]),
                        xytext=(5, 0), textcoords='offset points',
                        ha='left', va='center', color=line.get_color())

        ax.set_xlabel('Время')
        ax.set_ylabel('Процентное изменение цены')
        ax.legend(loc='upper left')
        ax.grid(True)

        plt.title('Сравнение изменения цен криптовалют')
        plt.gcf().autofmt_xdate()
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        ax.axhline(y=0, color='gray', linestyle='--')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
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
        correlation_interval=config['correlation_interval']
    )
    await analyzer.run(interval=config['update_interval'])


if __name__ == "__main__":
    asyncio.run(main())
