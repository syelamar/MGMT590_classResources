--MODEL 1 (No family)
-- Check your data
SELECT COUNT(*) as row_count, 
       MIN(date) as earliest_date,
       MAX(date) as latest_date
FROM `focal-furnace-465023-g3.assignment_dataflow.sales_data`;


-- Prepare store-level data
CREATE OR REPLACE TABLE `focal-furnace-465023-g3.assignment_dataflow.store_daily_sales` AS
SELECT 
  s.date,
  s.store_nbr,
  SUM(sales) as daily_sales,
  i.type,
  i.cluster
FROM `focal-furnace-465023-g3.assignment_dataflow.sales_data` s
JOIN `focal-furnace-465023-g3.assignment_dataflow.stores_data` i
  ON s.store_nbr = i.store_nbr
WHERE date < '2017-08-01'
GROUP BY date, store_nbr, type, cluster;

-- Build regression model: The process of feeding labeled data to a model so it can learn relationships
CREATE OR REPLACE MODEL `focal-furnace-465023-g3.assignment_dataflow.store_sales_model`
OPTIONS(
  model_type='linear_reg',
  input_label_cols=['daily_sales']
) AS
SELECT 
  daily_sales,
  store_nbr,
  EXTRACT(DAYOFWEEK FROM date) as day_of_week,
  EXTRACT(DAY FROM date) as day_of_month,
  type,
  cluster
FROM `focal-furnace-465023-g3.assignment_dataflow.store_daily_sales`;

-- Check model performance
SELECT * FROM ML.EVALUATE(MODEL `focal-furnace-465023-g3.assignment_dataflow.store_sales_model`);

-- For regression model  
SELECT *
FROM ML.PREDICT(
  MODEL `focal-furnace-465023-g3.assignment_dataflow.store_sales_model`,
  (
    SELECT
      store_nbr,
      EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
      EXTRACT(DAY FROM date) AS day_of_month,
      type,
      cluster
    FROM `focal-furnace-465023-g3.assignment_dataflow.store_daily_sales`
    WHERE date BETWEEN '2017-08-01' AND '2017-08-31'
  )
);

--Check how many rows match the date range since when i ran the regression model it gave me no data to display
SELECT COUNT(*) 
FROM `focal-furnace-465023-g3.assignment_dataflow.store_daily_sales`
WHERE date BETWEEN '2017-08-01' AND '2017-08-31';


--checking what data exists in the raw sales data
SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as total_rows
FROM `focal-furnace-465023-g3.assignment_dataflow.sales_data`;

-- For regression model try again with different range 
SELECT *
FROM ML.PREDICT(
  MODEL `focal-furnace-465023-g3.assignment_dataflow.store_sales_model`,
  (
    SELECT
      store_nbr,
      EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
      EXTRACT(DAY FROM date) AS day_of_month,
      type,
      cluster
    FROM `focal-furnace-465023-g3.assignment_dataflow.store_daily_sales`
    
  )
);

-- For regression model try again with different range 
SELECT *
FROM ML.PREDICT(
  MODEL `focal-furnace-465023-g3.assignment_dataflow.store_sales_model`,
  (
    SELECT
      store_nbr,
      EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
      EXTRACT(DAY FROM date) AS day_of_month,
      type,
      cluster
    FROM `focal-furnace-465023-g3.assignment_dataflow.store_daily_sales`
    WHERE date >= '2017-07-25' AND date < '2017-08-01'
  )
);

--MODEL #2 New Model (With family)
-- training data is at the store_nbr + date + family level
CREATE OR REPLACE TABLE `assignment_dataflow.store_family_sales` AS
SELECT
  store_nbr,
  date,
  family,
  SUM(sales) AS total_sales,
  AVG(onpromotion) AS avg_promotion
FROM `assignment_dataflow.sales_data`
GROUP BY store_nbr, date, family;
-- join store type, cluster, and date info
SELECT
  sfs.store_nbr,
  sfs.date,
  sfs.family,
  st.type,
  st.cluster,
  EXTRACT(DAYOFWEEK FROM sfs.date) AS day_of_week,
  EXTRACT(DAY FROM sfs.date) AS day_of_month,
  sfs.total_sales,
  sfs.avg_promotion
FROM `assignment_dataflow.store_family_sales` sfs
JOIN `assignment_dataflow.stores_data` st
ON sfs.store_nbr = st.store_nbr;
-- create table combining sales + store info
CREATE OR REPLACE TABLE `focal-furnace-465023-g3.assignment_dataflow.store_family_sales_features` AS
SELECT
  sd.store_nbr,
  sd.date,
  sd.family,
  s.type,
  s.cluster,
  EXTRACT(DAYOFWEEK FROM sd.date) AS day_of_week,
  EXTRACT(DAY FROM sd.date) AS day_of_month,
  sd.onpromotion,
  sd.sales AS total_sales
FROM `focal-furnace-465023-g3.assignment_dataflow.sales_data` sd
JOIN `focal-furnace-465023-g3.assignment_dataflow.stores_data` s
  ON sd.store_nbr = s.store_nbr;
-- train model including family as a categorical feature
CREATE OR REPLACE MODEL `assignment_dataflow.store_family_sales_model`
OPTIONS(
  model_type='linear_reg',
  input_label_cols=['total_sales']
) AS
SELECT
  store_nbr,
  family,
  type,
  cluster,
  day_of_week,
  day_of_month,
  onpromotion,
  total_sales
FROM `assignment_dataflow.store_family_sales_features`;
-- make predictions with family included
SELECT *
FROM ML.PREDICT(
  MODEL `focal-furnace-465023-g3.assignment_dataflow.store_family_sales_model`,
  (
    SELECT
      store_nbr,
      family,
      type,
      cluster,
      EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
      EXTRACT(DAY FROM date) AS day_of_month,
      onpromotion
    FROM `focal-furnace-465023-g3.assignment_dataflow.store_family_sales_features`
    WHERE date BETWEEN DATE '2017-08-01' AND DATE '2017-08-15'
  )
);
--Evaluate the model
SELECT *
FROM ML.EVALUATE(
  MODEL `focal-furnace-465023-g3.assignment_dataflow.store_family_sales_model`,
  (
    SELECT
      store_nbr,
      family,
      type,
      cluster,
      EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
      EXTRACT(DAY FROM date) AS day_of_month,
      onpromotion,
      total_sales
    FROM `focal-furnace-465023-g3.assignment_dataflow.store_family_sales_features`
    WHERE date BETWEEN DATE '2017-08-01' AND DATE '2017-08-15'
  )
);



