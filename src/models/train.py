import os

import click
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    Learns LogisticRegression model to find failures and saves trained model into 'model_path'

    :param train_path: filepath to train.csv
    :param target_name: column name to learn, hardcoded as 'target_failure'
    :param model_path: relative path to save trained model

    """
    train_data = pd.read_csv(train_path)
    train_data = train_data.select_dtypes(include=[float, int])
    X, y = train_data.drop([target_name, "event_id"], axis=1), train_data[target_name]

    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)

    output_path = os.path.join(model_path)
    joblib.dump(lr, output_path)


if __name__ == "__main__":
    train_lr()
