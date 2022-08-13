import os

import click
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
# "C:\\py\\predictive_esp\\data\\processed\\train.csv C:\\py\\predictive_esp\\models \"Survived\"


@click.command()
@click.argument("train_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("target_name", type=click.STRING)
@click.argument("model_name", type=click.STRING)
def train_lr(
    train_path: str,
    output_path: str,
    target_name: str,
    model_name: str
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
    train_data = train_data.select_dtypes(include=[int, float])
    train_data = train_data.dropna()
    X, y = train_data.drop(target_name, axis=1), train_data[target_name]
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)
    output_path = os.path.join(output_path, model_name)
    with open(output_path, "wb") as file:
        pickle.dump(lr, file)


if __name__ == "__main__":
    train_lr()
