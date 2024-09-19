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
# from pprint import pprint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CryptoAnalyzer:
    def __init__(self, exchange_name: str, symbol: str, telegram_token: str, chat_id: str):
        self.exchange = getattr(ccxt, exchange_name)()
        self.symbol = symbol
        self.bot = Bot(token=telegram_token)
        self.chat_id = chat_id
        self.prev_crypto_price, self.prev_volume = self.get_price_and_volume(self.symbol)
        self.prev_btc_price, _ = self.get_price_and_volume('BTC/USDT')
        self.crypto_prices = []
        self.btc_prices = []
        self.volumes = []
        self.timestamps = []

    def get_price_and_volume(self, symbol: str):
        ticker = self.exchange.fetch_ticker(symbol)
        # print(f"Ticker for {symbol}:")
        # pprint(ticker)

        return ticker['last'], ticker['quoteVolume']

    def analyze_prices(self, crypto_price: float, btc_price: float) -> str:
        crypto_change = (crypto_price - self.prev_crypto_price) / self.prev_crypto_price
        btc_change = (btc_price - self.prev_btc_price) / self.prev_btc_price

        if crypto_change > 0.02:
            return "Цена выросла! Надо продавать."
        elif crypto_change < -0.02:
            return "Цена упала! Надо покупать."
        elif abs(crypto_change - btc_change) > 0.01:
            return "Внимание! Цена движется иначе, чем Bitcoin."
        else:
            return "Значительных изменений нет."

    def create_correlation_chart(self):
        if len(self.crypto_prices) < 2:
            return None

        norm_crypto = (np.array(self.crypto_prices) - np.mean(self.crypto_prices)) / np.std(self.crypto_prices)
        norm_btc = (np.array(self.btc_prices) - np.mean(self.btc_prices)) / np.std(self.btc_prices)

        plt.figure(figsize=(10, 6))
        plt.plot(self.timestamps, norm_crypto, label=self.symbol)
        plt.plot(self.timestamps, norm_btc, label='BTC/USDT')
        plt.title(f'Корреляция {self.symbol} и BTC/USDT')
        plt.xlabel('Время')
        plt.ylabel('Нормализованная цена')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return buf

    def create_price_volume_chart(self):
        if len(self.crypto_prices) < 2:
            return None

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # цена
        ax1.set_xlabel('Время')
        ax1.set_ylabel('Цена', color='tab:blue')
        ax1.plot(self.timestamps, self.crypto_prices, color='tab:blue', label='Цена')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # объем
        ax2 = ax1.twinx()
        ax2.set_ylabel('Объем', color='tab:orange')
        ax2.plot(self.timestamps, self.volumes, color='tab:orange', label='Объем')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Настройка графика
        plt.title(f'График котировок и объема торгов {self.symbol}')
        fig.tight_layout()
        plt.gcf().autofmt_xdate()
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # Добавление легенды
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Сохранение графика
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
                    await asyncio.sleep(1)  # Ждем секунду перед повторной попыткой
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
                    await asyncio.sleep(1)  # Ждем секунду перед повторной попыткой
                else:
                    logger.error(f"Failed to send chart after {max_retries} attempts due to timeout")
            except TelegramError as e:
                logger.error(f"Telegram error occurred: {e}")
                return

    async def update_prices(self):
        try:
            crypto_price, volume = self.get_price_and_volume(self.symbol)
            btc_price, _ = self.get_price_and_volume('BTC/USDT')
            analysis = self.analyze_prices(crypto_price, btc_price)

            current_time = datetime.now()
            self.crypto_prices.append(crypto_price)
            self.btc_prices.append(btc_price)
            self.volumes.append(volume)
            self.timestamps.append(current_time)

            if len(self.crypto_prices) > 100:
                self.crypto_prices = self.crypto_prices[-100:]
                self.btc_prices = self.btc_prices[-100:]
                self.volumes = self.volumes[-100:]
                self.timestamps = self.timestamps[-100:]

            volume_change = ((volume - self.prev_volume) / self.prev_volume) * 100 if self.prev_volume else 0
            volume_str = f"{volume:,.0f}" if volume >= 1_000_000 else f"{volume:,.2f}"

            message = f"Цена {self.symbol}: {crypto_price:,.2f}\n"
            message += f"Объем: {volume_str} (изменение: {volume_change:+.2f}%)\n"
            message += f"BTC: {btc_price:,.2f}\n{analysis}"

            await self.send_message(message)

            price_volume_chart = self.create_price_volume_chart()
            await self.send_chart(price_volume_chart)

            if len(self.crypto_prices) % 5 == 0: #Каждые 5 минут шлем корреляцию
                correlation_chart = self.create_correlation_chart()
                await self.send_chart(correlation_chart)

            self.prev_crypto_price = crypto_price
            self.prev_btc_price = btc_price
            self.prev_volume = volume
        except Exception as e:
            logger.error(f"An error occurred in update_prices: {e}")

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
        symbol=config['symbol'],
        telegram_token=config['telegram_token'],
        chat_id=config['chat_id']
    )
    await analyzer.run(interval=config['update_interval'])


if __name__ == "__main__":
    asyncio.run(main())
