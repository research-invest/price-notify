import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError
from utils import format_number

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, token, chat_id, trade_manager, price_analyzer, chart_creator):
        self.chat_id = chat_id
        self.trade_manager = trade_manager
        self.price_analyzer = price_analyzer
        self.chart_creator = chart_creator
        self.token = token
        self.application = None

    def set_application(self, application: Application):
        self.application = application
        self.setup_handlers()

    # buy - покупка
    # sell - продажа
    # pnl - PNL
    # chart - диаграмма

    def setup_handlers(self):
        self.application.add_handler(CommandHandler("buy", self.handle_buy))
        self.application.add_handler(CommandHandler("sell", self.handle_sell))
        self.application.add_handler(CommandHandler("pnl", self.handle_pnl))
        self.application.add_handler(CommandHandler("chart", self.handle_chart))

    async def send_message(self, message):
        await self.application.bot.send_message(chat_id=self.chat_id, text=message)

    async def send_photo(self, photo):
        if not self.application:
            logger.error("Application is not initialized")
            return

        try:
            await self.application.bot.send_photo(chat_id=self.chat_id, photo=photo)
        except TelegramError as e:
            logger.error(f"Failed to send photo: {e}")
        except Exception as e:
            logger.error(f"Unexpected error when sending photo: {e}")

    async def handle_buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            _, symbol, amount, price = update.message.text.split()
            amount = float(amount)
            price = float(price)
            self.trade_manager.add_trade(symbol, 'buy', price, amount)
            await update.message.reply_text(f"Покупка зарегистрирована: {symbol}, {format_number(amount)} по цене {format_number(price)}")
        except ValueError:
            await update.message.reply_text("Неверный формат. Используйте: /buy SYMBOL AMOUNT PRICE")

    async def handle_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            _, symbol, amount, price = update.message.text.split()
            amount = float(amount)
            price = float(price)
            self.trade_manager.add_trade(symbol, 'sell', price, amount)
            await update.message.reply_text(f"Продажа зарегистрирована: {symbol}, {format_number(amount)} по цене {format_number(price)}")
        except ValueError:
            await update.message.reply_text("Неверный формат. Используйте: /sell SYMBOL AMOUNT PRICE")

    async def handle_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        current_prices = {symbol: prices[-1] for symbol, prices in self.price_analyzer.prices.items()}
        pnl_report = self.trade_manager.calculate_pnl(current_prices)
        await update.message.reply_text(pnl_report)

    async def handle_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chart = self.chart_creator.create_price_volume_chart()
        if chart:
            await update.message.reply_photo(photo=chart)
        else:
            await update.message.reply_text("Недостаточно данных для построения графика.")
