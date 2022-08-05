import os

import click
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command()
@click.argument("train_path", type=click.Path())
@click.argument("target_name", type=click.STRING)
@click.argument("output_path", type=click.Path())
def train_lr(
    train_path: str,
    target_name: str,
    output_path: str
) -> None:
    """
    Learns LogisticRegression model to find failures in data
    :param train_path:
    :param target_name:
    :param output_path:
    :return: lr_model.pkl
    """
    train_data = pd.read_csv(train_path)
    X, y = train_data.drop(target_name, axis=1), train_data[target_name]
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)
    pkl_filename = "lr_model.pkl"
    output_path = os.path.join(output_path, pkl_filename)
    with open(output_path, "wb") as file:
        pickle.dump(lr, file)


if __name__ == "__main__":
    train_lr()
