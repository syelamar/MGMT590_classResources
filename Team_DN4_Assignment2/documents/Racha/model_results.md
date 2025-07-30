# Part 3: Model Results

This document summarizes the results of the linear regression model built to predict daily sales per store.

## Model Evaluation

The model was evaluated using the `ML.EVALUATE` function. The key performance metrics are:

* **mean_absolute_error:** 245.24
* **r2_score:** 0.6053
* **mean_squared_error:** 199346.53

*The r2_score of approximately 0.61 suggests that the model's features (store number, day of week, store type, etc.) explain about 61% of the variance in daily store sales. While this shows a moderate relationship, it also indicates that a significant portion of the sales variation is not captured by the model.

The mean_absolute_error of ~$245 means that, on average, the model's prediction for a store's daily sales is off by $245. This is a substantial margin of error for operational purposes.*

## Prediction Results

The model was used to predict sales on a test set (August 1-15, 2017), and the results were analyzed in a Colab notebook. Several key findings emerged:

*

High Average Error: The model's large average error makes it unreliable for precise daily tasks like inventory ordering. Its value is likely higher for directional, weekly trend analysis.

Segmented Performance: The model's accuracy varies significantly across different store types. It is most accurate for Store Type D and least accurate for Store Type B.

Uncaptured Factors: The poor performance for certain store types suggests that there are important factors driving sales in those stores that are not included in our current dataset (e.g., local competition, specific marketing campaigns, or other external events).*