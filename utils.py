import math


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
