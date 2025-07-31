# Part 3: Model Results

---

## Model Summary

After an iterative development process, a **`BOOSTED_TREE_REGRESSOR`** was selected as the final model for predicting daily sales. This model successfully explains **79.3%** of the variance in sales (`R-squared`) and has a Mean Absolute Error (MAE) of approximately **$186**, providing a strong and reliable basis for business analysis.

The model is highly accurate for typical sales days, with the average error being skewed by a small number of hard-to-predict events. Its core strength lies in capturing the fundamental weekly rhythms of the business and the sales differences across store types and product families.

---

## Model Profile

### The Modeling Journey
The final model was chosen after a structured comparison:
* **Baseline Model:** A simple `linear_reg` model established a performance baseline with an R-squared of **0.64** and an MAE of **~$278**. This showed that a basic linear approach could not capture the complex sales patterns.
* **Advanced Model:** The `BOOSTED_TREE_REGRESSOR` model was then trained, demonstrating a significant improvement and was therefore selected.

### Model Architecture and Features
The final model is an ensemble algorithm chosen for its ability to capture complex, non-linear relationships between sales drivers.

* **Feature Set:**
    * **Store Features:** `store_nbr`, `store_type`, `cluster` (to learn unique sales patterns for each store's format and location).
    * **Product Features:** `family` (to understand distinct sales velocity for each product category).
    * **Temporal Features:** `day_of_week`, `day_of_month` (to capture weekly and monthly shopping rhythms).

* **Training Parameters:**
    * The model was trained using default BigQuery ML hyperparameters. Manual hyperparameter tuning is a potential area for future improvement.

---

## In-Depth Performance Analysis 📊

A detailed evaluation on the test data provides a comprehensive view of its performance and reveals crucial insights.

### Key Performance Metrics

| Metric                      | Value   | Insight                                           |
|-----------------------------|---------|---------------------------------------------------|
| **R-squared** | `0.793` | Strong overall fit, explaining 79.3% of variance. |
| **Mean Absolute Error (MAE)** | `~$186` | The average error per prediction across all days. |
| **Median Abs. Error (MedAE)**| `~$32`  | The error for a *typical* day, showing high accuracy.|
| **Root Mean Sq. Error (RMSE)**| `~$566` | The high value confirms large errors on a few volatile days.|

### Key Findings from Prediction Analysis

* **High Accuracy on Normal Days:** The large gap between the average error (**MAE ~$186**) and the median error (**MedAE ~$32**) is the most important finding. It proves the model is **highly accurate for the majority of typical sales days**. The higher average error is being skewed by a minority of predictions with large errors, likely on volatile, hard-to-predict sales events.

* **Variable Accuracy by Store Type:** The model's accuracy is not uniform. Analysis shows it is most accurate on low-volume stores (**Type C, MAE ~$109**) and least accurate on high-volume stores (**Type A, MAE ~$324**).

* **Core Drivers Identified:** The model's success is rooted in its ability to learn predictable, structural patterns, including the **weekly shopping cycle** (with sales peaking on weekends) and the fundamental differences in sales volume between store types and product families.
