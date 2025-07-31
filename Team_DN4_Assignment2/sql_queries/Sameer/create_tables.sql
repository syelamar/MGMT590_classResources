
-- Prepare store-level data (no date restriction)
CREATE OR REPLACE TABLE `mgmt590-465220.assignment_2.store_daily_sales` AS
SELECT 
  s.date,
  s.store_nbr,
  SUM(s.sales) AS daily_sales,
  i.type AS store_type,
  i.cluster
FROM `mgmt590-465220.assignment_2.sales_data` s
JOIN `mgmt590-465220.assignment_2.store_info` i
  ON s.store_nbr = i.store_nbr
WHERE s.date < '2017-08-01'
GROUP BY s.date, s.store_nbr, store_type, cluster;
