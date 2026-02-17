# Currency Toggle Implementation — USD/VND Conversion

**Date:** 2026-02-18  
**Feature:** Dynamic currency display toggle between VND (₫) and USD ($)

---

## Summary

Added a currency toggle in the sidebar that allows users to switch between Vietnamese Dong (VND) and US Dollar (USD) display throughout the entire application. All monetary values are stored in VND and converted to USD on-the-fly using a configurable exchange rate.

---

## Exchange Rate

**Current Rate:** 1 USD ≈ ₫24,000 VND

This rate is defined in `shared.py` as `VND_TO_USD_RATE` and can be updated as needed.

---

## Implementation Details

### 1. Shared Module (`shared.py`)

**Currency Formatting Functions:**
```python
VND_TO_USD_RATE = 24000.0  # Configurable exchange rate

def format_currency(amount: float, currency: str = "VND", decimals: int = None) -> str:
    """Format amount in VND or USD based on user preference."""
    if currency == "USD":
        usd_amount = amount / VND_TO_USD_RATE
        return f"${usd_amount:,.2f}"
    else:  # VND
        return f"₫{amount:,.0f}"
```

**Sidebar Toggle:**
- Location: Top of sidebar under "💱 Currency Display"
- Options: "VND (₫)" or "USD ($)"
- Default: VND
- Stored in: `st.session_state["currency"]`
- Shows conversion rate in help text

### 2. Updated Pages

**Page 1 (Decision Definition):**
- ARPU metric: Uses `format_currency(arpu, currency)`
- LTV30 histogram: Chart label changes based on currency

**Page 4 (Action & Simulation):**
- CPI input: Adjusts default value based on currency
  - VND: ₫10,000 (min: ₫1,000, step: ₫1,000)
  - USD: $0.42 (min: $0.01, step: $0.05)
- Table columns: Dynamic headers with currency symbol
- Charts: All axis labels update with currency symbol
- Revenue comparisons: Formatted with correct currency

**Page 5 (Feedback & Learning):**
- Chart Y-axis labels: Update to show correct currency symbol

---

## Usage

### For Users

1. **Toggle Currency:**
   - Open sidebar (top-left)
   - Find "💱 Currency Display" section
   - Select "VND (₫)" or "USD ($)"
   - All monetary values update instantly

2. **Conversion Rate:**
   - Hover over the currency toggle help icon to see current rate
   - Rate: 1 USD ≈ ₫24,000 VND

### For Developers

**To update the exchange rate:**
```python
# In shared.py, line 221
VND_TO_USD_RATE = 24000.0  # Update this value
```

**To add currency formatting to a new page:**
```python
from shared import format_currency

currency = st.session_state.get("currency", "VND")
formatted_value = format_currency(amount_in_vnd, currency)
```

**For chart labels:**
```python
currency = st.session_state.get("currency", "VND")
currency_symbol = "₫" if currency == "VND" else "$"
yaxis_title = f"Revenue ({currency_symbol})"
```

---

## Homepage Link Fix

### Problem
Markdown links in the summary report (e.g., `See → [decision_definition.md](decision_definition.md)`) caused navigation errors when clicked.

### Solution
Added regex-based link replacement in `app.py` that converts markdown links to plain text references:
```python
# Before: See → [decision_definition.md](decision_definition.md)
# After:  See report in sidebar: decision_definition.md
```

This prevents Streamlit from trying to navigate to non-existent routes while still informing users where to find the reports.

---

## Testing Checklist

- [x] Currency toggle appears in sidebar
- [x] Default currency is VND
- [x] Switching to USD converts all values correctly
- [x] Page 1 ARPU metric updates
- [x] Page 4 CPI input adjusts defaults
- [x] Page 4 table columns show correct currency
- [x] Page 4 charts update axis labels
- [x] Homepage links no longer cause errors
- [ ] User to verify conversion rate accuracy
- [ ] User to test all pages with both currencies

---

## Files Modified

1. `webapp/shared.py` — Currency formatting functions and sidebar toggle
2. `webapp/app.py` — Homepage link fix
3. `webapp/pages/1_Decision_Definition.py` — ARPU metric and chart labels
4. `webapp/pages/4_Action_and_Simulation.py` — CPI input, table, and charts
5. `webapp/pages/5_Feedback_and_Learning.py` — Chart labels

---

## Future Enhancements

1. **Live Exchange Rate:** Integrate with a currency API for real-time rates
2. **More Currencies:** Add support for other currencies (EUR, JPY, etc.)
3. **User Preference:** Save currency preference in browser localStorage
4. **Conversion Indicator:** Show small badge indicating values are converted
