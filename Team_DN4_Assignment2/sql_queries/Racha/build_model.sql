CREATE OR REPLACE MODEL `mgmt599-rachakondas-lab1.store_sales_team_dn4.store_sales_boosted_tree_model`
OPTIONS(
  model_type='BOOSTED_TREE_REGRESSOR',
  input_label_cols=['daily_sales']
) AS
SELECT
  daily_sales,
  store_nbr,
  family,
  EXTRACT(DAYOFWEEK FROM date) as day_of_week,
  EXTRACT(DAY FROM date) as day_of_month,
  store_type,
  cluster
FROM
  `mgmt599-rachakondas-lab1.store_sales_team_dn4.store_daily_sales`;