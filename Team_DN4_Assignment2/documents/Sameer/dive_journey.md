# Store Sales Forecasting: DIVE Journey Report

This documents the process and outcomes of applying the DIVE (Discover, Investigate, Validate, Extend) framework to develop and evaluate an XGBoost-based sales forecasting model on Google Cloud Platform. The goal was to predict daily store sales with high accuracy and translate technical insights into actionable business recommendations.

---

## 🔍 Discover

Initial analysis focused on identifying key sales patterns. Results indicated that store sales followed a highly predictable and structured rhythm. The baseline BOOSTED_TREE_REGRESSOR model captured 79.3% of the variance with a Mean Absolute Error (MAE) of approximately $186.

Subsequently, the XGBoost model was trained and evaluated, resulting in the following performance:

- **MSE:** 43,245,584.28  
- **MAE:** 3,816.75  
- **R²:** 0.8055

Key findings included:
- A consistent weekly shopping cycle, with Sunday sales averaging 64% higher than Thursday sales.
- Structural sales variation by store type, with Type A stores generating 3.5× the sales of Type C.
- Unique sales rhythms across product families.

---

## 🔎 Investigate

The next phase explored the underlying drivers of the discovered sales patterns:

- **Weekly Rhythm:** The day of the week emerged as a critical feature due to consumers’ consistent shopping routines.
- **Store Characteristics:** Store type and format aligned with different business strategies, significantly influencing sales volumes.
- **Predictability by Store Type:** The model achieved higher accuracy for low-volume Type C stores (MAE ~$109) and showed lower accuracy for high-volume Type A stores (MAE ~$324), highlighting increased complexity in larger outlets.

These insights supported the interpretation that structural and behavioral patterns significantly drive store-level sales variability.

---

## 🧪 Validate

A critical evaluation of the model was performed to understand its assumptions, limitations, and potential failure modes:

- **Assumption:** The model extrapolates from historical patterns to predict future behavior.
- **Limitations:** The dataset ends in 2017, omitting recent consumer trends and excluding real-time operational data such as inventory levels and staffing.
- **Vulnerabilities:** The model lacks awareness of unexpected events (e.g., weather disruptions, competitive campaigns) and may underperform during unprecedented scenarios such as pandemics or renovations.
- **Error Distribution:** An error analysis identified that the top 10 stores contributed over $1.8M in total forecast error, with Store 52 alone accounting for ~$279K. Store 3 had the highest percentage error (522%).

These observations emphasized the need for operational sensitivity and model refinement, especially in high-error environments.

---

## 🚀 Extend

Insights from the model were translated into specific, actionable strategies:

1. **Short-Term Operations (Next Week):**  
   Leverage forecast data to adjust staffing and inventory levels, especially for weekends.  
   - **Target:** Reduce weekend stock-outs in top 5 product families by 10%.  
   - **Mitigation:** Maintain an on-call staff list and safety stock thresholds.

2. **Mid-Term Strategy (Next Month):**  
   Apply differentiated approaches based on store performance.  
   - For Type A: Allow local managers to override forecasts using contextual knowledge.  
   - For Type C: Implement automated restocking for non-perishable goods.  
   - **Target:** 5% reduction in inventory waste and improved product availability.

3. **Long-Term Monitoring:**  
   Establish anomaly detection using forecast deviations as alerts for unknown events.  
   - **Threshold:** Deviation of 25% or more for 3 consecutive days.  
   - **Purpose:** Turn model limitations into strategic opportunities.  
   - **Risk Control:** Avoid alert fatigue by tuning thresholds to major disruptions.

---

The DIVE framework provided a structured pathway from technical model development to strategic operational recommendations, enabling predictive insights to be embedded into store-level decision-making.
