# Chat Over Tabular Data (RAG-style)

A chat that answers natural-language questions about business data (clients, invoices, line items) by generating pandas code on the fly and feeding the results to an LLM for a grounded answer.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# copy .env.example to .env and fill in your Groq API key
cp .env.example .env

# web UI
streamlit run app.py

# or run all 15 test questions at once
python run_tests.py       # writes test_results.md
```

## Architecture

The pipeline has three stages:

1. **Code generation** -- the user's question plus table schemas (column names, types, sample rows, relationships) are sent to the LLM. It returns Python/pandas code that queries the DataFrames. Conversation history from prior turns is included so follow-up questions ("which of those…", "filter them by…") resolve correctly.

2. **Sandboxed execution** -- the generated code runs in a restricted namespace with the three DataFrames, `pd`, `np`, and `datetime` (builtins are limited to a safe allowlist and import statements are stripped). If it errors out, the traceback is fed back to the LLM for a retry (up to 2 attempts).

3. **Answer synthesis** -- the retrieved data plus the original question go to the LLM again, which formats a human-readable answer. It's instructed to use *only* the provided data, so numbers always come from actual pandas output rather than being hallucinated.

### Hallucination mitigation

The key design choice is the two-stage split: the first LLM call produces *code* (not answers), and the code is what actually touches the data. The second LLM call only gets the query results and is told to stick to them. This means every number in the final answer traces back to a real pandas computation. The exec sandbox also means the LLM can't sneak in arbitrary side effects.

## Assumptions and limitations

- All data fits in memory (20 clients, 40 invoices, 96 line items).
- "Total billed amount including tax" = `sum(qty * unit_price * (1 + tax_rate))`.
- The `exec()` sandbox uses a safe builtins allowlist and strips import statements. Fine for a demo, but in production I'd use something like RestrictedPython or run in a container.
- Conversation history grows unbounded -- for very long sessions it could exceed the model's context window. A sliding window or summary would fix this.
- LLM code generation can occasionally produce wrong code for ambiguous questions. The retry helps but isn't bulletproof.
- Only Groq is supported as an LLM provider; swapping in OpenAI/Anthropic would be straightforward.
