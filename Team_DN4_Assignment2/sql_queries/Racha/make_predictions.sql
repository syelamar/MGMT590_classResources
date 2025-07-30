-- This query uses the final model to make predictions on the unseen test data.
-- The output includes the original data plus a new column with the model's predictions,
-- allowing for a direct comparison between actual and predicted sales.

SELECT
  *
FROM
  ML.PREDICT(MODEL `mgmt599-rachakondas-lab1.store_sales_team_dn4.store_sales_boosted_tree_model`,
    (
      -- The inner SELECT provides the test data and creates the
      -- date-based features the model requires for prediction.
      SELECT
        *,
        EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
        EXTRACT(DAY FROM date) AS day_of_month
      FROM
        `mgmt599-rachakondas-lab1.store_sales_team_dn4.test_data`
    )
  );