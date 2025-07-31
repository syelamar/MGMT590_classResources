SELECT * FROM ML.PREDICT(MODEL `mgmt590-465220.assignment_2.store_sales_model`,
  (SELECT * FROM `mgmt590-465220.assignment_2.test_data`)
);