import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from database import DatabaseManager
import json
import asyncio
import signal

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeManager:
    def __init__(self, database: DatabaseManager):
        self.db = database

    def add_trade(self, trade):
        return self.db.add_trade(trade)

    def get_all_trades(self):
        return self.db.get_all_trades()

    def update_trade(self, trade):
        return self.db.update_trade(trade)

    def delete_trade(self, trade_id):
        return self.db.delete_trade(trade_id)

    def get_trade_stats(self):
        return self.db.get_trade_stats()


class TelegramBot:
    def __init__(self, telegram: dict, trade_manager: TradeManager, environment: str, ):
        self.token = telegram['token']
        self.webhook_url = telegram['webhook_url']
        self.environment = environment
        self.trade_manager = trade_manager

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        keyboard = [
            [InlineKeyboardButton("Купить", callback_data='buy'),
             InlineKeyboardButton("Продать", callback_data='sell')],
            [InlineKeyboardButton("Прибыль/Убыток", callback_data='pnl'),
             InlineKeyboardButton("График", callback_data='chart')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text('Выберите действие:', reply_markup=reply_markup)

    async def buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Пожалуйста, введите данные покупки в формате: цена входа, объем")
        context.user_data['next_action'] = 'buy'

    async def sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Пожалуйста, введите данные продажи в формате: цена выхода, объем")
        context.user_data['next_action'] = 'sell'

    async def pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        stats = self.trade_manager.get_trade_stats()
        if stats:
            message = f"Статистика:\nВсего сделок: {stats['total_trades']}\n"
            message += f"Общий объем: {stats['total_volume']}\n"
            message += f"Средняя цена входа: {stats['avg_entry_price']:.2f}\n"
            message += f"Общая прибыль/убыток: {stats['total_pnl']:.2f}"
        else:
            message = "Нет данных для статистики."
        await update.message.reply_text(message)

    async def chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Здесь должна быть логика создания и отправки графика
        await update.message.reply_text("Функция графика пока не реализована.")

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()

        if query.data == 'buy':
            await self.buy(update, context)
        elif query.data == 'sell':
            await self.sell(update, context)
        elif query.data == 'pnl':
            await self.pnl(update, context)
        elif query.data == 'chart':
            await self.chart(update, context)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        next_action = context.user_data.get('next_action')
        if next_action in ['buy', 'sell']:
            try:
                entry_price, volume = update.message.text.split(',')
                trade = {
                    'symbol': 'TAO/USDT',
                    'trade_type': next_action,
                    'entry_price': float(entry_price.strip()),
                    'volume': float(volume.strip())
                }
                self.trade_manager.add_trade(trade)
                await update.message.reply_text("Сделка успешно добавлена!")
            except ValueError:
                await update.message.reply_text(
                    "Неверный формат. Пожалуйста, используйте формат: цена входа, объем")
            finally:
                context.user_data['next_action'] = None
        elif 'editing_trade' in context.user_data:
            try:
                symbol, entry_price, volume = update.message.text.split(',')
                trade = {
                    'id': context.user_data['editing_trade'],
                    'symbol': symbol.strip(),
                    'entry_price': float(entry_price.strip()),
                    'volume': float(volume.strip())
                }
                self.trade_manager.update_trade(trade)
                await update.message.reply_text("Сделка успешно обновлена!")
            except ValueError:
                await update.message.reply_text(
                    "Неверный формат. Пожалуйста, используйте формат: символ, цена входа, объем")
            finally:
                del context.user_data['editing_trade']
        else:
            await update.message.reply_text("Я не понимаю. Пожалуйста, используйте команду /start для начала работы.")

    async def run(self):
        self.application = Application.builder().token(self.token).build()

        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("buy", self.buy))
        self.application.add_handler(CommandHandler("sell", self.sell))
        self.application.add_handler(CommandHandler("pnl", self.pnl))
        self.application.add_handler(CommandHandler("chart", self.chart))
        self.application.add_handler(CallbackQueryHandler(self.button))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        await self.application.initialize()
        await self.application.start()
        
        if self.environment == 'production':
            # Настройка для webhook
            await self.application.bot.set_webhook(url=f"{self.webhook_url}/{self.token}")
            await self.application.update_queue.put(Update.de_json({"update_id": -1}, self.application.bot))
        else:
            # Использование polling для локальной разработки
            await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)

    async def stop(self):
        await self.application.stop()
        await self.application.shutdown()


async def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    db_manager = DatabaseManager(config['db'])
    db_manager.connect()
    db_manager.create_tables()

    trade_manager = TradeManager(db_manager)
    bot = TelegramBot(config['telegram'], trade_manager, config['environment'])

    # Настройка обработчика сигналов для корректного завершения
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))

    try:
        await bot.run()
        # Держим программу запущенной
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await bot.stop()
        db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
