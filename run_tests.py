"""Runs all test questions and writes test_results.md."""

import os
import sys
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

QUESTIONS = [
    "List all clients with their industries.",
    "Which clients are based in the UK?",
    "List all invoices issued in March 2024 with their statuses.",
    'Which invoices are currently marked as "Overdue"?',
    "For each service_name in InvoiceLineItems, how many line items are there?",
    "List all invoices for Acme Corp with their invoice IDs, invoice dates, due dates, and statuses.",
    "Show all invoices issued to Bright Legal in February 2024, including their status and currency.",
    "For invoice I1001, list all line items with service name, quantity, unit price, tax rate, and compute the line total (including tax) for each.",
    "For each client, compute the total amount billed in 2024 (including tax) across all their invoices.",
    "Which client has the highest total billed amount in 2024, and what is that total?",
    # optional / extra
    "Across all clients, which three services generated the most revenue in 2024? Show the total revenue per service.",
    "Which invoices are overdue as of 2024-12-31? List invoice ID, client name, invoice_date, due_date, and status.",
    "Group revenue by client country: for each country, compute the total billed amount in 2024 (including tax).",
    'For the service "Contract Review", list all clients who purchased it and the total amount they paid for that service (including tax).',
    "Considering only European clients, what are the top 3 services by total revenue (including tax) in H2 2024 (2024-07-01 to 2024-12-31)?",
]


def main():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("Set GROQ_API_KEY in .env or environment.")
        sys.exit(1)

    pipeline = RAGPipeline(api_key=api_key)

    rows = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {q}")
        result = pipeline.ask(q)
        answer = result["answer"] if not result["error"] else f"ERROR: {result['error']}"
        rows.append((q, answer))
        print("  done\n")

    with open("test_results.md", "w") as f:
        f.write("# Test Results\n\n")
        f.write("| Question | Answer |\n")
        f.write("|----------|--------|\n")
        for q, a in rows:
            a_esc = a.replace("|", "\\|").replace("\n", "<br>")
            q_esc = q.replace("|", "\\|")
            f.write(f"| {q_esc} | {a_esc} |\n")

    print("Done -> test_results.md")


if __name__ == "__main__":
    main()
