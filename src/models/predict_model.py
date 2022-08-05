import os

import click
import pickle
import pandas as pd
# from sklearn.linear_model import LogisticRegression


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("model_name", type=click.STRING())
def predict_model(
    input_path: str,
    output_path: str,
    model_name: str
):
    """
    Looks for failures in observations provided in "input_path"

    :param input_path:
    :param output_path:
    :param model_name:
    :return: predictions.csv
    """
    predict_data = pd.read_csv(input_path)
    output_path = os.path.join(output_path, model_name)
    with open(output_path, "rb") as file:
        model = pickle.load(file)
    model_predictions = model.predict(predict_data)
    model_predictions = pd.DataFrame(model_predictions)

    FILENAME = "predictions.csv"
    output_path = os.path.join(output_path, FILENAME)
    model_predictions.to_csv(output_path)


if __name__ == "__main__":
    predict_model()
