"""Write the committed data snapshot: one gzipped CSV per season from the raw Baseball Savant pull.
Only fetch_full.py's KEEP columns. Rows with malformed dates (Savant occasionally emits broken lines) are dropped."""
import pandas as pd, hashlib, os, sys
src = sys.argv[1] if len(sys.argv) > 1 else "data/pitches_full.csv"
df = pd.read_csv(src, on_bad_lines="skip", low_memory=False, dtype=str)
df = df[df.game_date.astype(str).str.len() == 10]
for yr in ("2024", "2025", "2026"):
    p = f"data/statcast_{yr}.csv.gz"
    df[df.game_date.str[:4] == yr].to_csv(p, index=False, compression={"method": "gzip", "mtime": 0})
    print(p, os.path.getsize(p) // 2**20, "MB", hashlib.sha256(open(p, "rb").read()).hexdigest())
