# testQueryDecomposition.py
import sys
import os

# Ensure app package is importable if run standalone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.llm.llmClient import llmClient

def main():
    # Example prompt for query decomposition
    prompt = """
You are a query decomposition assistant.

Take the following complex question and break it down into multiple smaller,
executable sub-queries. Each sub-query should be something that could be run
independently against a knowledge base.

⚠️ Output must be a valid JSON array of strings and nothing else.

Example:
Question: "Fetch sales data from 2020 to 2022"
Output: [
"Fetch sales data for 2020", 
"Fetch sales data for 2021", 
"Fetch sales data for 2022",
"Combine all into a single result"]

Question:
"Can you create a table of the sales data from 2012 to 2016 with year in column 1 and sales in column 2?"
"""

    print("=== Running Query Decomposition Test ===")
    print(f"Prompt:\n{prompt}\n")

    response = llmClient.generateAnswer(prompt, temperature=0.3)
    print("=== Model Response ===")
    print(response)


if __name__ == "__main__":
    main()
