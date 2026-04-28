import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path(__file__).resolve().parent / "synthetic_data"
results = []

for nozzle_dir in data_dir.glob('*'):
    clean_dir = nozzle_dir / 'cdf' / 'clean'
    if not clean_dir.exists():
        continue
        
    for f in clean_dir.glob('*.csv'):
        try:
            df = pd.read_csv(f)
            if df.empty or 'k_sqrt' not in df.columns:
                continue
            for _, row in df.iterrows():
                # We need valid rows
                k_sqrt = row.get('k_sqrt', np.nan)
                k_quarter = row.get('k_quarter', np.nan)
                t0 = row.get('t0', np.nan)
                
                # Check for NaNs
                if pd.notna(k_sqrt) and pd.notna(k_quarter) and pd.notna(t0) and t0 > 0:
                    amp_sqrt = k_sqrt * np.sqrt(t0)
                    amp_quarter = k_quarter * (t0 ** 0.25)
                    ratio = amp_sqrt / amp_quarter if amp_quarter > 0 else np.nan
                    results.append({
                        'nozzle': nozzle_dir.name,
                        'k_sqrt': k_sqrt,
                        'k_quarter': k_quarter,
                        't0': t0,
                        'amp_sqrt': amp_sqrt,
                        'amp_quarter': amp_quarter,
                        'ratio': ratio
                    })
        except Exception as e:
            pass

df_res = pd.DataFrame(results)
df_res = df_res.dropna(subset=['ratio'])
total_valid = len(df_res)
count_less_than_half = len(df_res[df_res['ratio'] < 0.5])
percentage = (count_less_than_half / total_valid) * 100

print(f'Total accepted samples (clean fits): {total_valid}')
print(f'Samples where sqrt amp < 0.5 * quarter amp at t0: {count_less_than_half} ({percentage:.1f}%)')
print(f'Median ratio: {df_res["ratio"].median():.3f}')

plt.figure(figsize=(8, 6))
plt.hist(df_res['ratio'], bins=np.linspace(0, 2, 100), alpha=0.7, color='steelblue', edgecolor='black')
plt.axvline(0.5, color='r', linestyle='--', label='Ratio = 0.5')
plt.axvline(df_res["ratio"].median(), color='green', linestyle='-', label=f'Median = {df_res["ratio"].median():.2f}')
plt.xlabel('Amplitude Ratio ($k_{sqrt}\sqrt{t_0} / k_{quarter}t_0^{1/4}$)')
plt.ylabel('Count')
plt.title('Distribution of Branch Amplitude Ratio at Transition $t_0$')
plt.legend()
plt.grid(alpha=0.3)
out_path = Path(__file__).resolve().parent / "figures" / "amplitude_ratio_dist.png"
out_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved distribution plot to {out_path}')