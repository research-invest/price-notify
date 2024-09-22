from datetime import datetime
from utils import format_number

class TradeManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.trades = {}

    def load_trades(self, symbols, start_date, end_date):
        for symbol in symbols:
            trades = self.db_manager.get_trades(symbol, start_date, end_date)
            self.trades[symbol] = [(trade[0], float(trade[1]), float(trade[2]), trade[3]) for trade in trades]

    def add_trade(self, symbol: str, trade_type: str, price: float, amount: float):
        timestamp = datetime.now()
        self.db_manager.add_trade(symbol, trade_type, price, amount, timestamp)
        if symbol not in self.trades:
            self.trades[symbol] = []
        self.trades[symbol].append((trade_type, price, amount, timestamp))

    def calculate_pnl(self, current_prices):
        pnl_report = "PNL отчет:\n"
        for symbol, trades in self.trades.items():
            symbol_pnl = 0
            total_bought = 0
            total_sold = 0
            for trade_type, price, amount, _ in trades:
                if trade_type == 'buy':
                    symbol_pnl -= price * amount
                    total_bought += amount
                else:  # sell
                    symbol_pnl += price * amount
                    total_sold += amount
            
            current_price = current_prices.get(symbol, 0)
            unrealized_pnl = (total_bought - total_sold) * current_price
            total_pnl = symbol_pnl + unrealized_pnl
            
            pnl_report += f"{symbol}:\n"
            pnl_report += f"  Реализованный PNL: {format_number(symbol_pnl)}\n"
            pnl_report += f"  Нереализованный PNL: {format_number(unrealized_pnl)}\n"
            pnl_report += f"  Общий PNL: {format_number(total_pnl)}\n\n"
        
        return pnl_report
