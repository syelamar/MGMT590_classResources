-- Create training data (before Aug 1, 2017)
CREATE OR REPLACE TABLE `mgmt590-465220.assignment_2.store_daily_sales` AS
SELECT
  s.date,
  s.store_nbr,
  SUM(s.sales) AS daily_sales,
  i.type AS store_type,
  i.cluster,
  EXTRACT(DAYOFWEEK FROM s.date) AS day_of_week,
  EXTRACT(DAY FROM s.date) AS day_of_month
FROM `mgmt590-465220.assignment_2.sales_data` s
JOIN `mgmt590-465220.assignment_2.store_info` i
  ON s.store_nbr = i.store_nbr
WHERE s.date < '2017-08-01'
GROUP BY s.date, s.store_nbr, store_type, cluster;

-- Create test data (Aug 1 to Aug 15, 2017)
CREATE OR REPLACE TABLE `mgmt590-465220.assignment_2.test_data` AS
SELECT
  s.date,
  s.store_nbr,
  i.type AS store_type,
  i.cluster,
  EXTRACT(DAYOFWEEK FROM s.date) AS day_of_week,
  EXTRACT(DAY FROM s.date) AS day_of_month
FROM `mgmt590-465220.assignment_2.sales_data` s
JOIN `mgmt590-465220.assignment_2.store_info` i
  ON s.store_nbr = i.store_nbr
WHERE s.date >= '2017-08-01'
  AND s.date < '2017-08-16';