# german_credit_python
 
A Logistic Regression Python model to predict loan default.

To run the attached Jupyter notebook, create a Python 3.6.9  virtual environments with the libraries versioned in `requirements.txt`.

Model was trained on the German Credit Data dataset.
`logreg_classifier.pickle` is the trained model artifact.
Sample inputs to the scoring function are included (`df_baseline.json`, `df_sample.json`).

Model code includes a metrics function used to compute Group and Bias metrics.
The metrics function expects a DataFrame with at lease the following three columns three columns: `score` (predicted), `label_value` (actual), and `gender` (protected attribute).
Sample inputs to the metrics function are included (`df_baseline_scored.json`, `df_sample_scored.json`).

The output of the scoring job when the input data is `df_sample.json` is a JSONS file (one-line JSON records). Here are the first three output records:
```json
[{"id": 993, "duration_months": 36, "credit_amount": 3959, "installment_rate": 4, "present_residence_since": 3, "age_years": 30, "number_existing_credits": 1, "checking_status": "A11", "credit_history": "A32", "purpose": "A42", "savings_account": "A61", "present_employment_since": "A71", "debtors_guarantors": "A101", "property": "A122", "installment_plans": "A143", "housing": "A152", "job": "A174", "number_people_liable": 1, "telephone": "A192", "foreign_worker": "A201", "gender": "male", "label_value": 0, "score": 1}]
[{"id": 859, "duration_months": 9, "credit_amount": 3577, "installment_rate": 1, "present_residence_since": 2, "age_years": 26, "number_existing_credits": 1, "checking_status": "A14", "credit_history": "A32", "purpose": "A40", "savings_account": "A62", "present_employment_since": "A73", "debtors_guarantors": "A103", "property": "A121", "installment_plans": "A143", "housing": "A151", "job": "A173", "number_people_liable": 2, "telephone": "A191", "foreign_worker": "A202", "gender": "male", "label_value": 0, "score": 0}]
[{"id": 298, "duration_months": 18, "credit_amount": 2515, "installment_rate": 3, "present_residence_since": 4, "age_years": 43, "number_existing_credits": 1, "checking_status": "A14", "credit_history": "A32", "purpose": "A42", "savings_account": "A61", "present_employment_since": "A73", "debtors_guarantors": "A101", "property": "A121", "installment_plans": "A143", "housing": "A152", "job": "A173", "number_people_liable": 1, "telephone": "A192", "foreign_worker": "A201", "gender": "male", "label_value": 0, "score": 0}]
```
