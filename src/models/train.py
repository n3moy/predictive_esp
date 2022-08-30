import os

import click
import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

DROP_COLS = ["time", "event_id"]
TEST_PATH = "/c/py/predictive_esp/data/processed/train.csv"


@click.command()
@click.argument("train_path", type=click.Path())
@click.argument("model_path", type=click.Path())
@click.argument("target_name", type=click.STRING)
def train_lr(
    train_path: str,
    model_path: str,
    target_name: str,
) -> None:
    """
    Learns LogisticRegression model to find failures in data
    :param train_path:
    :param target_name:
    :param model_path:
    :return: lr_model.pkl
    """
    train_data = pd.read_csv(train_path)
    train_data = train_data.select_dtypes(include=[float, int])
    X, y = train_data.drop([target_name, "event_id"], axis=1), train_data[target_name]

    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)

    # print("Testing metrics")
    output_path = os.path.join(model_path)
    joblib.dump(lr, output_path)


if __name__ == "__main__":
    train_lr()
