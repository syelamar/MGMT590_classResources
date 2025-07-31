
-- Check model performance
SELECT * FROM ML.EVALUATE(MODEL `mgmt590-465220.assignment_2.store_sales_model`);

-- Make Predictions
SELECT * FROM ML.PREDICT(MODEL `mgmt590-465220.assignment_2.store_sales_model`,
  (SELECT * FROM `mgmt590-465220.assignment_2.store_daily_sales`));
