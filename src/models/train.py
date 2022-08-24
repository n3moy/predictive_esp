import os

import click
import pickle
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

DROP_COLS = ["time", "event_id"]
TEST_PATH = "/c/py/predictive_esp/data/processed/train.csv"


@click.command()
@click.argument("train_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("target_name", type=click.STRING)
def train_lr(
    train_path: str,
    output_path: str,
    target_name: str,
) -> None:
    """
    Learns LogisticRegression model to find failures in data
    :param model_name:
    :param train_path:
    :param target_name:
    :param output_path:
    :return: lr_model.pkl
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(TEST_PATH)
    train_data = train_data.select_dtypes(include=[float, int])
    test_data = test_data.select_dtypes(include=[float, int])
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    X, y = train_data.drop(target_name, axis=1), train_data[target_name]
    X_test, y_test = test_data.drop(target_name, axis=1), test_data[target_name]

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow_experiment_id = "lr_model"
    mlflow.set_experiment(mlflow_experiment_id)

    with mlflow.start_run():
        lr = LogisticRegression(random_state=42)
        lr.fit(X, y)
        prediction = lr.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        roc_auc = roc_auc_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("ROC_AUC", roc_auc)
        mlflow.log_metric("F1_score", f1)

        # print("Testing metrics")
        # output_path = os.path.join(output_path, model_name)
        with open(output_path, "wb") as file:
            pickle.dump(lr, file)

        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="lr_model",
            registered_model_name="default_lr_model"
        )

        # mlflow.end_run()


if __name__ == "__main__":
    train_lr()
