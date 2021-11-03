# modelop.schema.0: input_schema.avsc
# modelop.schema.1: output_schema.avsc

import pandas as pd
import pickle

import modelop.monitors.performance as performance


# modelop.init
def begin():

    global logreg_classifier

    # load pickled logistic regression model
    logreg_classifier = pickle.load(open("logreg_classifier.pickle", "rb"))


# modelop.score
def action(data):

    # Turn data into DataFrame
    data = pd.DataFrame([data])

    # There are only two unique values in data.number_people_liable.
    # Treat it as a categorical feature
    data.number_people_liable = data.number_people_liable.astype("object")

    predictive_features = [
        "duration_months",
        "credit_amount",
        "installment_rate",
        "present_residence_since",
        "age_years",
        "number_existing_credits",
        "checking_status",
        "credit_history",
        "purpose",
        "savings_account",
        "present_employment_since",
        "debtors_guarantors",
        "property",
        "installment_plans",
        "housing",
        "job",
        "number_people_liable",
        "telephone",
        "foreign_worker",
    ]

    data["score"] = logreg_classifier.predict(data[predictive_features])

    # MOC expects the action function to be a *yield* function
    yield data.to_dict(orient="records")


# modelop.metrics
def metrics(dataframe):
    # Initialize ModelEvaluator
    model_evaluator = performance.ModelEvaluator(
        dataframe=dataframe,
        score_column="score",  # MONITORING_PARAMETERS["score_column"],
        label_column="label_value",  # MONITORING_PARAMETERS["label_column"],
    )
    # Compute classification metrics
    classification_metrics = model_evaluator.evaluate_performance(
        pre_defined_metrics="classification_metrics"
    )
    result = {
        # Top-level metrics
        "accuracy": classification_metrics["values"]["accuracy"],
        "precision": classification_metrics["values"]["precision"],
        "recall": classification_metrics["values"]["recall"],
        "auc": classification_metrics["values"]["auc"],
        "f1_score": classification_metrics["values"]["f1_score"],
        "confusion_matrix": classification_metrics["values"]["confusion_matrix"],
        # Vanilla ModelEvaluator output
        "performance": [classification_metrics],
    }
    yield result


if __name__ == "__main__":
    import pandas

    sample = pandas.read_json("df_sample_scored.json", lines=True)

    from pprint import pprint

    print(next(metrics(sample)))
