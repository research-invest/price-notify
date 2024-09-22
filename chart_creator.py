import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from config import CONFIG
import logging
from utils import format_number

logger = logging.getLogger(__name__)


class ChartCreator:
    def __init__(self, price_analyzer, trade_manager):
        self.price_analyzer = price_analyzer
        self.trade_manager = trade_manager
        self.colors = None

    def create_price_volume_chart(self):
        if len(self.price_analyzer.timestamps) < 2:
            logger.warning("Недостаточно данных для построения графика")
            return None

        # Инициализируем цвета здесь, когда мы уверены, что у нас есть данные
        if self.colors is None or len(self.colors) != len(self.price_analyzer.prices):
            self.colors = plt.cm.rainbow(np.linspace(0, 1, len(self.price_analyzer.prices)))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        for i, symbol in enumerate(self.price_analyzer.prices.keys()):
            color = self.colors[i]

            prices = np.array(self.price_analyzer.prices[symbol])
            if len(prices) == 0:
                logger.warning(f"Нет данных о ценах для символа {symbol}")
                continue  # Пропускаем символы без данных

            # Убедимся, что длины массивов совпадают
            min_length = min(len(self.price_analyzer.timestamps), len(prices))
            timestamps = self.price_analyzer.timestamps[:min_length]
            prices = prices[:min_length]

            initial_price = prices[0]
            normalized_prices = (prices - initial_price) / initial_price * 100

            # График цены
            linewidth = 2 if i == 0 else 1  # Делаем линию Bitcoin толще
            linestyle = '-'  # if i == 0 else '--'
            line, = ax1.plot(timestamps, normalized_prices, color=color, label=f'{symbol} Цена',
                             linewidth=linewidth, linestyle=linestyle)

            # Добавляем аннотации
            for j, (timestamp, price) in enumerate(zip(timestamps, normalized_prices)):
                if j % CONFIG['annotation_interval'] == 0 or j == len(prices) - 1:
                    ax1.annotate(f'{format_number(price)}%', (timestamp, price), textcoords="offset points",
                                 xytext=(0, 10), ha='center', fontsize=8, color=color)

            volumes = np.array(self.price_analyzer.volumes[symbol][:min_length])
            if len(volumes) == 0:
                logger.warning(f"Нет данных об объемах для символа {symbol}")
                continue  # Пропускаем символы без данных

            initial_volume = volumes[0]
            normalized_volumes = (volumes - initial_volume) / initial_volume * 100

            ax2.plot(timestamps, normalized_volumes, color=color, label=f'{symbol} Объем', linewidth=linewidth,
                     linestyle=linestyle)

            # Аннотации для объемов
            for j, (timestamp, norm_volume, volume) in enumerate(zip(timestamps, normalized_volumes, volumes)):
                if j % CONFIG['annotation_interval'] == 0 or j == len(volumes) - 1:
                    ax2.annotate(f'{volume}',
                                 xy=(timestamp, norm_volume),
                                 xytext=(0, 5), textcoords='offset points',
                                 ha='center', va='bottom', color=color,
                                 fontsize=8, rotation=45)

            for trade_type, price, amount, trade_timestamp in self.trade_manager.trades.get(symbol, []):
                if trade_timestamp in timestamps:
                    if trade_type == 'buy':
                        ax1.scatter(trade_timestamp, price, color='green', marker='^', s=100)
                    else:  # sell
                        ax1.scatter(trade_timestamp, price, color='red', marker='v', s=100)

        ax1.set_ylabel('Процентное изменение цены')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Процентное изменение объема')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        plt.close()

        return buf
