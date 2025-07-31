# XGBoost Model Evaluation Results

This document summarizes the performance of the XGBoost regression model used to predict daily store sales.

## Evaluation Metrics

- **Mean Squared Error (MSE):** 43,245,584.28  
- **Mean Absolute Error (MAE):** 3,816.75  
- **R-squared (R²):** 0.8055

## Analysis

The XGBoost model demonstrates strong predictive performance, as indicated by the R² score of **0.8055**. This means the model is able to explain approximately **80.55%** of the variance in daily sales across stores.

A **MAE of 3,816.75** suggests that on average, the model’s sales predictions deviate from actual sales by around $3,817. This level of error is acceptable given the potential scale of sales and implies good real-world applicability for strategic decision-making.

Moreover, the **MSE of 43.2 million**—while large in absolute terms—aligns with the scale of the dataset and helps in penalizing large errors, ensuring robust performance even under varied store and date conditions.

Overall, these metrics indicate that the XGBoost model is effective for forecasting store-level daily sales and can be integrated into business intelligence tools for operations planning, inventory management, and marketing strategy.