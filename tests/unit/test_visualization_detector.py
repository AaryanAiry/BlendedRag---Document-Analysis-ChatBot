# tests/test_visualisation_detector.py
import pytest
from app.visualization.detector import detect_visualization_type

@pytest.mark.parametrize(
    "query,expected",
    [
        ("Show me a table of sales from 2015 to 2020", "table"),
        ("Plot a bar chart of revenue by month", "chart"),
        ("Create a flowchart of the approval process", "flowchart"),
        ("How many pages are in the PDF?", "none"),
        ("Visualize the number of students per class", "chart"),
        ("Generate a pivot table for the data", "table"),
        ("Draw a workflow diagram", "flowchart"),
        ("Just tell me the total revenue", "none")
    ]
)
def test_detect_visualization_type(query, expected):
    assert detect_visualization_type(query) == expected

if __name__ == "__main__":
    pytest.main([__file__])
