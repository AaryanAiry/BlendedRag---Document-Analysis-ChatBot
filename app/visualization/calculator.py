# app/calculator.py
from typing import List, Dict, Union

def sum_column(table: List[List[Union[int, float, str]]], col_index: int) -> float:
    """
    Sum all numeric values in a given column of a 2D table.
    Non-numeric cells are ignored.
    """
    total = 0
    for row in table:
        try:
            total += float(row[col_index])
        except (ValueError, IndexError):
            continue
    return total

def avg_column(table: List[List[Union[int, float, str]]], col_index: int) -> float:
    """
    Average all numeric values in a given column of a 2D table.
    Non-numeric cells are ignored.
    """
    values = []
    for row in table:
        try:
            values.append(float(row[col_index]))
        except (ValueError, IndexError):
            continue
    if values:
        return sum(values) / len(values)
    return 0

def sum_columns(table: List[List[Union[int, float, str]]], col_indices: List[int]) -> Dict[int, float]:
    """Return a dict of sums for multiple columns."""
    return {i: sum_column(table, i) for i in col_indices}

def avg_columns(table: List[List[Union[int, float, str]]], col_indices: List[int]) -> Dict[int, float]:
    """Return a dict of averages for multiple columns."""
    return {i: avg_column(table, i) for i in col_indices}
