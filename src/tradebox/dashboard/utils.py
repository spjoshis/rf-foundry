"""Utility functions for dashboard."""

from typing import Tuple


def format_currency(value: float) -> str:
    """
    Format value as Indian currency.

    Args:
        value: Amount in rupees

    Returns:
        Formatted string (e.g., "₹1,05,000")

    Example:
        >>> format_currency(105000)
        '₹1,05,000'
    """
    return f"₹{value:,.0f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.

    Args:
        value: Percentage value (e.g., 5.25 for 5.25%)
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "5.25%")

    Example:
        >>> format_percentage(5.25)
        '5.25%'
    """
    return f"{value:.{decimals}f}%"


def calculate_color(value: float) -> str:
    """
    Calculate color based on value (red for negative, green for positive).

    Args:
        value: Numeric value

    Returns:
        Color name ("red", "green", or "gray")

    Example:
        >>> calculate_color(5.5)
        'green'
        >>> calculate_color(-2.3)
        'red'
    """
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    else:
        return "gray"


def format_delta(value: float, is_percentage: bool = True) -> str:
    """
    Format delta with sign and color.

    Args:
        value: Delta value
        is_percentage: Whether value is percentage

    Returns:
        Formatted string with sign

    Example:
        >>> format_delta(5.5)
        '+5.50%'
        >>> format_delta(-2.3, is_percentage=False)
        '-2.30'
    """
    sign = "+" if value >= 0 else ""

    if is_percentage:
        return f"{sign}{value:.2f}%"
    else:
        return f"{sign}{value:,.2f}"
