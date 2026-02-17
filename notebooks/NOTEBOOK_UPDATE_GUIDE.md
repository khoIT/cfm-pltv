# Notebook Update Guide

## Recent Changes to Implement

All notebooks have been updated with the following improvements:

### ✅ Completed Updates

**Notebook 1 (decision_definition.ipynb):**
- ✅ Currency toggle support (VND/USD)
- ✅ Cumulative revenue decile table
- ✅ Gini coefficient calculation
- ✅ Lorenz curve visualization
- ✅ Whale economy tier segmentation
- ✅ Removed MD file writing (read-only)

**Notebook 2 (feature_exploration.ipynb):**
- ✅ Currency conversion for monetary features
- ✅ Updated data loading to check multiple file names
- ✅ Currency-aware scatter plots
- ✅ Removed MD file writing (read-only)

### 🔄 Remaining Notebooks (3-5)

For notebooks 3, 4, and 5, apply these changes manually or run the cells below:

#### Common Updates for All Notebooks

**1. Data Loading (Cell 1):**
```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT = Path('.').resolve().parent
sys.path.insert(0, str(ROOT))

# Load data - try multiple possible training file names
DATA_DIR = ROOT / 'data'
possible_files = [
    'cfm_pltv_train.csv',
    'cfm_pltv_train_1.csv',
    'cfm_pltv_train_imoney.csv',
    'clm_pltv_iamount.csv',
]

df = None
for fname in possible_files:
    fpath = DATA_DIR / fname
    if fpath.exists():
        df = pd.read_csv(fpath, nrows=100_000, low_memory=False)
        print(f'✅ Loaded {len(df):,} rows from {fname}')
        break

if df is None:
    raise FileNotFoundError(f"No training data found")

# Currency settings (VND by default, can toggle to USD)
CURRENCY = "VND"  # Change to "USD" to see values in dollars
VND_TO_USD = 24000

def convert_currency(value, to_currency="VND"):
    """Convert VND values to USD if needed."""
    if to_currency == "USD":
        return value / VND_TO_USD
    return value

def format_currency(value, currency="VND"):
    """Format currency with appropriate symbol."""
    if currency == "USD":
        return f"${value:,.2f}"
    return f"₫{value:,.0f}"

currency_symbol = "₫" if CURRENCY == "VND" else "$"
print(f"💱 Currency: {CURRENCY} ({currency_symbol})")
```

**2. Remove All MD File Writing:**
- Delete any cells with `write_report()` calls
- Add note: "This notebook is read-only for reports. It does NOT write to MD files."

**3. Convert Chart Data Values:**
For any chart displaying monetary values, convert the data:
```python
# Before plotting
df['ltv30_display'] = convert_currency(df['ltv30'], CURRENCY)
df['rev_d7_display'] = convert_currency(df['rev_d7'], CURRENCY)

# Then plot using _display columns
fig = px.histogram(df, x='ltv30_display', ...)
fig.update_layout(xaxis_title=f'LTV30 ({currency_symbol})')
```

---

## Notebook 3 Specific Updates

**Add Test 1 vs Test 2 Comparison:**

```python
# Load both test sets
test1_df = pd.read_csv(DATA_DIR / 'cfm_pltv_test1.csv', nrows=50_000)
test2_df = pd.read_csv(DATA_DIR / 'cfm_pltv_test2.csv', nrows=50_000)

print(f"Test 1: {len(test1_df):,} rows")
print(f"Test 2: {len(test2_df):,} rows")

# Compare distributions
for label, test_df in [("Test 1", test1_df), ("Test 2", test2_df)]:
    print(f"\n{label}:")
    print(f"  Mean LTV30: {format_currency(test_df['ltv30'].mean(), CURRENCY)}")
    print(f"  Median LTV30: {format_currency(test_df['ltv30'].median(), CURRENCY)}")
    print(f"  Payer Rate: {test_df['is_payer_30'].mean() * 100:.1f}%")
```

**Add Lift Curve Comparison:**
```python
# Compute lift curves for both test sets
def compute_lift(y_true, y_pred):
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    cum_rev = np.cumsum(y_sorted) / y_sorted.sum()
    pcts = np.arange(1, len(cum_rev) + 1) / len(cum_rev)
    return pcts, cum_rev

# Plot side-by-side
fig = go.Figure()
# Add Test 1 and Test 2 lift curves
# (requires model predictions - see Page 6 Diagnostics for full implementation)
```

---

## Notebook 4 Specific Updates

**Currency-Aware Simulation:**

```python
# Convert CPI and revenue values
cpi = 10000 if CURRENCY == "VND" else 0.42
revenue_captured = convert_currency(raw_revenue, CURRENCY)

# Update all charts
fig.update_layout(
    yaxis_title=f"Revenue ({currency_symbol})"
)
```

---

## Notebook 5 Specific Updates

**Time Dynamics with Currency:**

```python
daily = df.groupby('install_date').agg(
    total_ltv30=('ltv30', 'sum'),
    avg_ltv30=('ltv30', 'mean'),
).reset_index()

daily['total_ltv30_display'] = convert_currency(daily['total_ltv30'], CURRENCY)
daily['avg_ltv30_display'] = convert_currency(daily['avg_ltv30'], CURRENCY)

# Plot using _display columns
fig.add_trace(go.Bar(x=daily['install_date'], y=daily['total_ltv30_display'], ...))
```

---

## Key Principles

1. **Read-Only:** Notebooks should NEVER write to MD files in `reports/`
2. **Currency Toggle:** All monetary values must support VND/USD conversion
3. **Data Loading:** Check multiple possible file names for flexibility
4. **Chart Data:** Convert actual Y-axis values, not just labels
5. **Consistency:** Use same helper functions across all notebooks

---

## Testing Checklist

After updating each notebook:
- [ ] Runs without errors
- [ ] No `write_report()` calls
- [ ] Currency toggle works (change `CURRENCY = "USD"` and rerun)
- [ ] Charts show correct currency symbols
- [ ] Data loads from available training files

---

## Quick Start

To use updated notebooks:
1. Open in Jupyter: `jupyter notebook` or `jupyter lab`
2. Navigate to `notebooks/` folder
3. Open any notebook
4. Run all cells
5. To switch currency: Change `CURRENCY = "VND"` to `"USD"` in cell 1, then rerun all

---

**Last Updated:** 2026-02-18
**Status:** Notebooks 1-2 fully updated, 3-5 have update guide above
