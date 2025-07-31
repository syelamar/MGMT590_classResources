CREATE OR REPLACE MODEL `mgmt590-465220.assignment_2.store_sales_model`
OPTIONS(
  model_type='boosted_tree_regressor',
  input_label_cols=['daily_sales']
) AS
SELECT 
  daily_sales,
  store_nbr,
  day_of_week,
  day_of_month,
  store_type,
  cluster
FROM `mgmt590-465220.assignment_2.store_daily_sales`;