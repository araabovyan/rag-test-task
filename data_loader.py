"""Loads the Excel data and builds schema descriptions for the LLM."""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_dataframes() -> dict[str, pd.DataFrame]:
    """Load all three Excel files into DataFrames."""
    clients = pd.read_excel(os.path.join(DATA_DIR, "Clients.xlsx"))
    invoices = pd.read_excel(os.path.join(DATA_DIR, "Invoices.xlsx"))
    line_items = pd.read_excel(os.path.join(DATA_DIR, "InvoiceLineItems.xlsx"))

    invoices["invoice_date"] = pd.to_datetime(invoices["invoice_date"])
    invoices["due_date"] = pd.to_datetime(invoices["due_date"])

    return {
        "clients": clients,
        "invoices": invoices,
        "line_items": line_items,
    }


def get_schema_description(dfs: dict[str, pd.DataFrame]) -> str:
    """Build a schema string the LLM uses to understand the tables."""
    lines = []

    lines.append("=== DATABASE SCHEMA ===\n")

    for name, key in [("clients", "clients"), ("invoices", "invoices"),
                      ("line_items", "line_items")]:
        df = dfs[key]
        lines.append(f"Table: {name} (variable name: {name})")
        lines.append(f"  Rows: {len(df)}")
        lines.append("  Columns:")
        for col in df.columns:
            lines.append(f"    - {col} ({df[col].dtype})")
        lines.append("  Sample rows:")
        lines.append(df.head(3).to_string(index=False))

        # extra metadata per table
        if name == "clients":
            lines.append(f"  Unique countries: {df['country'].unique().tolist()}")
        elif name == "invoices":
            lines.append(f"  Unique statuses: {df['status'].unique().tolist()}")
        elif name == "line_items":
            lines.append(f"  Unique services: {df['service_name'].unique().tolist()}")
        lines.append("")

    lines.append("=== RELATIONSHIPS ===")
    lines.append("- invoices.client_id -> clients.client_id")
    lines.append("- line_items.invoice_id -> invoices.invoice_id")
    lines.append("")

    lines.append("=== COMPUTATION NOTES ===")
    lines.append("- Line total (including tax) = quantity * unit_price * (1 + tax_rate)")
    lines.append("- To convert to USD: multiply local-currency amount by fx_rate_to_usd")
    lines.append("- 'Total billed amount' = sum of line totals across all invoices for a client.")
    lines.append("- Always include tax unless the question says otherwise.")
    lines.append("- For geographic/regional questions (e.g. 'European clients'), use your "
                 "general knowledge to classify the countries listed above.")

    return "\n".join(lines)
