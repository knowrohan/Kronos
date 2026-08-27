"""Sanity check: does predict_paths() agree with the library's own predict()?

Library predict(sample_count=N) returns the mean of N draws. Our predict_paths(N) returns
the N draws themselves. Their means should land in the same place; the library's mean should
sit inside our sampled envelope. Also checks the deterministic limit: T->tiny, 1 path.
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor
from bench.kronos_eval import predict_paths, FEATS, DATA

N = 20
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base")
p = KronosPredictor(mdl, tok, max_context=512)

df = pd.read_csv(os.path.join(DATA, "btc_1h.csv"), encoding="utf-8-sig")
df["timestamps"] = pd.to_datetime(df["timestamps"])
lb, hz = 512, 48
w = df.iloc[-(lb + hz):].reset_index(drop=True)
x_df, x_ts = w.loc[:lb-1, FEATS], w.loc[:lb-1, "timestamps"]
y_ts = w.loc[lb:, "timestamps"].reset_index(drop=True)
actual = w.loc[lb:, "close"].values
anchor = float(x_df["close"].iloc[-1])

torch.manual_seed(7)
lib = p.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=hz,
                T=1.0, top_p=0.9, sample_count=N, verbose=False)["close"].values

torch.manual_seed(7)
paths = predict_paths(p, x_df, x_ts, y_ts, hz, N, 1.0, 0.9)[:, :, 3]

mine_mean = paths.mean(axis=0)
lo, hi = paths.min(axis=0), paths.max(axis=0)
inside = ((lib >= lo) & (lib <= hi)).mean()
rel = np.abs(mine_mean - lib).mean() / anchor * 100

print(f"anchor close            {anchor:,.2f}")
print(f"library mean  final     {lib[-1]:,.2f}  ({(lib[-1]-anchor)/anchor*100:+.2f}%)")
print(f"our paths mean final    {mine_mean[-1]:,.2f}  ({(mine_mean[-1]-anchor)/anchor*100:+.2f}%)")
print(f"actual        final     {actual[-1]:,.2f}  ({(actual[-1]-anchor)/anchor*100:+.2f}%)")
print(f"mean|ours-lib| as % anchor : {rel:.3f}%   <- two means of {N} stochastic draws, small = agree")
print(f"library mean inside our envelope: {inside*100:.0f}% of bars")
print(f"our path spread at final bar: {lo[-1]:,.0f} .. {hi[-1]:,.0f}  (library mean collapses this to one number)")

torch.manual_seed(7)
cold = predict_paths(p, x_df, x_ts, y_ts, hz, 3, 0.05, 0.1)[:, :, 3]
print(f"near-deterministic (T=0.05,top_p=0.1) 3-path final-bar spread: "
      f"{cold[:,-1].min():,.0f} .. {cold[:,-1].max():,.0f}  (should be tight)")
