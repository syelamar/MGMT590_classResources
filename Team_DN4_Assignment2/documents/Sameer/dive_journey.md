# Store Sales Forecasting: DIVE Journey Report

**Authors:** Rachakonda Srikanth, Sameer Yelamarthi & Valerie Robert

This report outlines our end-to-end journey using the DIVE (Discover, Investigate, Validate, Extend) framework to build and evaluate an XGBoost model for store-level sales prediction on GCP.

---

## 🔍 Discover (5 points)

Initial analysis revealed that store sales follow a **highly predictable, structured rhythm**. Our baseline BOOSTED_TREE_REGRESSOR model captured **79.3% of variance** with a MAE of ~$186, while our XGBoost model improved performance to:

- **MSE:** 43,245,584.28  
- **MAE:** 3,816.75  
- **R²:** 0.8055

These results confirmed that sales dynamics are governed by:
- Weekly shopping cycles (e.g., Sunday sales 64% higher than Thursday),
- Structural store differences (Type A stores earn 3.5× more than Type C),
- Distinct demand curves by product family.

---

## 🔎 Investigate (10 points)

We explored **why** certain factors drive sales:

- **Day-of-Week Patterns:** A disciplined shopping schedule results in strong weekday variance. 
- **Store Typology:** Store size and format play pivotal roles, with Type A stores facing more demand volatility.
- **Predictability Variation:** Our model is **most accurate for low-volume Type C stores (MAE ~$109)** and **least accurate for high-volume Type A stores (MAE ~$324)**, suggesting complex dynamics at larger outlets.

---

## 🧪 Validate (10 points)

The XGBoost model assumes past patterns predict the future, but we identified blind spots:

- **External shocks:** Events like local marketing, weather, or competition aren't captured.
- **Data gaps:** Data ends in 2017, excluding recent trends and lacks operational factors like staffing and inventory.
- **Failure cases:** Renovations, pandemics, or supply disruptions can break historic patterns.
- **Error analysis:** Top 10 error-prone stores contributed **$1.8M in forecast error**, showing where to refine models or operations.

---

## 🚀 Extend (5 points)

We translated insights into actionable tactics:

1. **Forecast-Driven Operations (Next Week):** Use family/day-level predictions to align labor and stock for weekend peaks.
   - 📉 Goal: 10% fewer stock-outs, stable overtime.
2. **Store-Specific Tactics (Next Month):** 
   - Type A: Manual override with manager intuition.
   - Type C: Automate restocking.
   - 🧪 Metric: 5% waste reduction in Type C; higher in-stock rate in Type A.
3. **Early Warning System (Long Term):** Alerts for ≥25% deviation from forecast across 3+ days.
   - 🔔 Turns blind spots into discovery tools.

---

This journey highlights the power of combining machine learning with strategic business frameworks like DIVE to drive real-world, store-level operational excellence.