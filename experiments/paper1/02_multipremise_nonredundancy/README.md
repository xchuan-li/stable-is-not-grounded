# Experiment 02: Multi-Premise Non-Redundancy

Paper section:
- 5.4

Claim supported:
- Non-redundancy can be tested non-trivially on a two-premise transitive structure.
- Premise ablations test whether the model withholds when a strict subset of M is present.
- A class-3 shortcut regime decouples withholding from the premises and is independently flagged by `do(color)`.

Run:

```bash
python run_distilbert.py
```

Output:

```text
results/summary.json
```

Known limits:
- Synthetic setup.
- Single fine-tune seed.
- Shortcut competition is constructed, not naturalistic.

