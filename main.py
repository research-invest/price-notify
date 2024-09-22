import logging

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from telegram import Update

from database_manager import DatabaseManager
from price_analyzer import PriceAnalyzer
from trade_manager import TradeManager
from chart_creator import ChartCreator
from telegram_bot import TelegramBot
from config import CONFIG
import asyncio
import ccxt
from datetime import datetime
from telegram.ext import ApplicationBuilder, Application
from utils import format_number

logger = logging.getLogger(__name__)

class CryptoApp:
    def __init__(self):
        self.exchange = getattr(ccxt, CONFIG['exchange_name'])()
        self.symbols = CONFIG['symbols']
        self.db_manager = DatabaseManager(CONFIG['db'])
        self.price_analyzer = PriceAnalyzer(self.db_manager)
        self.trade_manager = TradeManager(self.db_manager)
        self.chart_creator = ChartCreator(self.price_analyzer, self.trade_manager)
        self.telegram_bot = TelegramBot(
            CONFIG['telegram_token'],
            CONFIG['chat_id'],
            self.trade_manager,
            self.price_analyzer,
            self.chart_creator
        )

    async def update_prices(self):
        while True:
            try:
                message = ""
                for symbol in self.symbols:
                    ticker = self.exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    volume = ticker['quoteVolume']
                    timestamp = datetime.fromtimestamp(ticker['timestamp'] / 1000)
                    self.price_analyzer.update_price(symbol, price, volume, timestamp)
                    self.db_manager.save_price_data(symbol, timestamp, price, volume)
                    
                    analysis = self.price_analyzer.analyze_prices(symbol, price)
                    message += f"{symbol}:\nЦена: {format_number(price)}\nОбъем: {format_number(volume)}\nАнализ: {analysis}\n\n"

                await self.telegram_bot.send_message(message)
                chart = self.chart_creator.create_price_volume_chart()
                if chart:
                    await self.telegram_bot.send_photo(chart)
                else:
                    logger.warning("Failed to create chart")

            except Exception as e:
                logger.error(f"Error in update_prices: {e}")

            await asyncio.sleep(CONFIG['update_interval'])

    async def start(self):
        self.db_manager.connect()
        self.db_manager.create_tables()
        self.price_analyzer.load_historical_data(self.symbols)
        self.trade_manager.load_trades(self.symbols, self.price_analyzer.timestamps[0],
                                       self.price_analyzer.timestamps[-1])

        application = ApplicationBuilder().token(CONFIG['telegram_token']).build()
        self.telegram_bot.set_application(application)

        # Запускаем обновление цен и Telegram бота асинхронно
        async def run_bot():
            await application.initialize()
            await application.start()
            await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)

        update_task = asyncio.create_task(self.update_prices())
        bot_task = asyncio.create_task(run_bot())

        try:
            # Ждем, пока одна из задач не завершится или не будет отменена
            await asyncio.gather(update_task, bot_task)
        except asyncio.CancelledError:
            logger.info("Tasks were cancelled")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            # Отменяем задачи, если они еще выполняются
            update_task.cancel()
            bot_task.cancel()

            # Ждем завершения задач
            await asyncio.gather(update_task, bot_task, return_exceptions=True)

            # Останавливаем бота
            await application.stop()
            await application.shutdown()


if __name__ == "__main__":
    app = CryptoApp()
    asyncio.run(app.start())
