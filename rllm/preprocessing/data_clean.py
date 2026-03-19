import re
from pandas import Series


def _clean_numerical_value(val):
    """
    Clean a single numerical value that may contain special formatting.

    Handles:
    - Currency symbols: $132.42, ¥1000, €50.5, etc.
    - Percentages: 1.572%, 25% (converts to decimal by dividing by 100)
    - Thousand separators: 1,234.56
    - Scientific notation: 1.5e10, 2.3E-5
    - Whitespace: "  123.45  ", "1 234.56"

    Args:
        val: Value to clean (can be str, int, float, or None)

    Returns:
        float or None: Cleaned numerical value or None if invalid
    """
    # Handle None and existing numeric types
    if val is None or isinstance(val, (int, float)):
        return val

    # Convert to string and strip whitespace
    val_str = str(val).strip()
    if not val_str:
        return None

    try:
        # Check for percentage format
        is_percentage = "%" in val_str
        if is_percentage:
            val_str = val_str.replace("%", "")
        # Remove all whitespace (handles "1 234.56" format)
        val_str = val_str.replace(" ", "")
        # Remove currency symbols and other non-numeric chars
        # Keep: digits, decimal point, minus sign, comma, and 'e/E' for scientific notation
        val_str = re.sub(r"[^\d.\-,eE]", "", val_str)
        # Remove thousand separators
        val_str = val_str.replace(",", "")
        # Validate and convert to float
        if val_str and val_str not in ["-", ".", "-.", "e", "E"]:
            num_val = float(val_str)
            # Convert percentage to decimal
            if is_percentage:
                num_val = num_val / 100.0
            return num_val
        return None

    except (ValueError, AttributeError):
        # Return None for invalid values (will be handled by fillna)
        return None


def preprocess_numerical_string(col_series: Series) -> Series:
    return col_series.map(_clean_numerical_value)
