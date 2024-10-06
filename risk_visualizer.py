import matplotlib.pyplot as plt
import numpy as np


class RiskVisualizer:
    def __init__(self, initial_price, initial_investment, leverage, target_profit_percentage):
        self.initial_price = initial_price
        self.initial_investment = initial_investment
        self.leverage = leverage
        self.target_profit_percentage = target_profit_percentage
        self.position_size = initial_investment * leverage
        self.initial_quantity = self.position_size / initial_price

    def calculate_price_for_profit(self, profit_percentage):
        profit = self.initial_investment * profit_percentage / 100
        price_change = profit / (self.initial_quantity * self.leverage)
        return self.initial_price + price_change

    def visualize_risk(self):
        price_range = np.linspace(self.initial_price * 0.94, self.initial_price * 1.06, 100)
        
        # Точки для докупа при падении цены (шаг 10% прибыли)
        down_prices = [self.calculate_price_for_profit(-i*10) for i in range(1, 4)]
        down_investments = [self.initial_investment * (1 + i*0.5) for i in range(3)]
        
        # Точки для продажи при росте цены (шаг 10% прибыли)
        up_prices = [self.calculate_price_for_profit(i*10) for i in range(1, 4)]
        up_investments = [self.initial_investment * (1 - i*0.2) for i in range(3)]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Диагональная линия цены
        ax.plot(price_range, price_range, color='black', linestyle='-', linewidth=2, label='Цена')

        # График для падения цены
        profit_down = (price_range - self.initial_price) * self.initial_quantity * self.leverage
        ax.plot(price_range, profit_down + price_range, label='Прибыль/Убыток (падение)', color='red')
        for price, inv in zip(down_prices, down_investments):
            profit = (price - self.initial_price) * self.initial_quantity * self.leverage
            ax.plot(price, profit + price, 'ro', markersize=10)
            ax.vlines(price, price, profit + price, colors='red', linestyles='dotted')
            profit_label = "Прибыль" if profit >= 0 else "Убыток"
            ax.annotate(f'Докуп: {inv:.2f}$\nЦена: {price:.2f}\n{profit_label}: {abs(profit):.2f}$', 
                        (price, profit + price), xytext=(5, 5), textcoords='offset points')

        # График для роста цены
        profit_up = (price_range - self.initial_price) * self.initial_quantity * self.leverage
        ax.plot(price_range, profit_up + price_range, label='Прибыль/Убыток (рост)', color='green')
        for price, inv in zip(up_prices, up_investments):
            profit = (price - self.initial_price) * self.initial_quantity * self.leverage
            ax.plot(price, profit + price, 'go', markersize=10)
            ax.vlines(price, price, profit + price, colors='green', linestyles='dotted')
            profit_label = "Прибыль" if profit >= 0 else "Убыток"
            ax.annotate(f'Продажа: {self.initial_investment - inv:.2f}$\nЦена: {price:.2f}\n{profit_label}: {abs(profit):.2f}$', 
                        (price, profit + price), xytext=(5, 5), textcoords='offset points')

        # Точка безубыточности
        ax.axvline(x=self.initial_price, color='blue', linestyle='--', label='Начальная цена')

        # Целевая прибыль
        target_price_up = self.calculate_price_for_profit(self.target_profit_percentage)
        ax.axvline(x=target_price_up, color='purple', linestyle=':', 
                   label=f'Целевая прибыль {self.target_profit_percentage}% (цена: {target_price_up:.2f})')

        ax.set_xlabel('Цена')
        ax.set_ylabel('Прибыль/Убыток + Цена')
        ax.set_title('Визуализация рисков и стратегии усреднения/частичной продажи')
        ax.legend()
        ax.grid(True)
        plt.show()


if __name__ == "__main__":
    visualizer = RiskVisualizer(initial_price=500, initial_investment=1000, leverage=5, target_profit_percentage=10)
    visualizer.visualize_risk()