
-- Build XGBoost regression model
CREATE OR REPLACE MODEL `mgmt590-465220.assignment_2.store_sales_model`
OPTIONS(
  model_type='BOOSTED_TREE_REGRESSOR',
  input_label_cols=['daily_sales']
) AS
SELECT 
  daily_sales,
  store_nbr,
  EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
  EXTRACT(DAY FROM date) AS day_of_month,
  store_type,
  cluster
FROM `mgmt590-465220.assignment_2.store_daily_sales`;
