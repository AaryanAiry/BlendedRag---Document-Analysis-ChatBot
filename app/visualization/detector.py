# app/visualization/detector.py
from typing import Literal

# Define the types of visualizations we currently support
VISUAL_TYPES = ["table", "chart", "flowchart", "none"]

# Keywords associated with each visualization type
VISUAL_KEYWORDS = {
    "table": ["table", "grid", "spreadsheet", "matrix", "tabulate", "pivot"],
    "chart": ["chart", "graph", "plot", "line", "bar", "scatter", "visualize", "histogram", "pie"],
    "flowchart": ["flowchart", "diagram", "process", "sequence", "workflow", "step-by-step"],
}

def detect_visualization_type(query: str) -> Literal["table", "chart", "flowchart", "none"]:
    """
    Detects the visualization type intended by the user's query.

    Returns:
        - "table" if the query intends a table
        - "chart" if the query intends a chart/graph
        - "flowchart" if the query intends a flowchart/diagram
        - "none" if no visualization is detected
    """
    q = query.lower()

    for vis_type, keywords in VISUAL_KEYWORDS.items():
        if any(keyword in q for keyword in keywords):
            return vis_type

    return "none"

# # ------------------- Optional Test -------------------
# if __name__ == "__main__":
#     test_queries = [
#         "Show me a table of sales from 2015 to 2020",
#         "Plot a bar chart of revenue by month",
#         "Create a flowchart of the approval process",
#         "How many pages are in the PDF?"
#     ]

#     for q in test_queries:
#         print(f"Query: {q}")
#         print("Detected visualization type:", detect_visualization_type(q))
