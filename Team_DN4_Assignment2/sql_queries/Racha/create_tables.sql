-- Creates the table for store information
-- This DDL represents the final schema for the store_info table.
CREATE TABLE `mgmt599-rachakondas-lab1.store_sales_team_dn4.store_info`
(
  store_nbr INT64,
  city STRING,
  state STRING,
  type STRING,
  cluster INT64
);

-- Creates the table for historical sales data
-- This DDL represents the final schema for the sales_data table.
CREATE TABLE `mgmt599-rachakondas-lab1.store_sales_team_dn4.sales_data`
(
  id INT64,
  date DATE,
  store_nbr INT64,
  family STRING,
  sales FLOAT64,
  onpromotion INT64
);