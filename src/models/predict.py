import os

import click
import joblib
import pandas as pd

FILENAME = "predictions.csv"


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("model_path", type=click.Path())
def predict_model(
    input_path: str,
    output_path: str,
    model_path: str
) -> None:
    """
    Looks for failures in observations provided in "input_path"

    :param model_path:
    :param input_path:
    :param output_path:
    :return: predictions.csv
    """
    predict_data = pd.read_csv(input_path)
    model = joblib.load(model_path)
    model_predictions = model.predict(predict_data)
    model_predictions = pd.DataFrame(model_predictions)

    output_path = os.path.join(output_path, FILENAME)
    model_predictions.to_csv(output_path)


if __name__ == "__main__":
    predict_model()
