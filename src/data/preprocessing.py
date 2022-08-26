import os

import click
import pandas as pd

DROP_COLS = []


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def preprocess(
    input_path: str,
    output_path: str
) -> None:

    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        data_file = pd.read_csv(file_path, parse_dates=["time"])
        data_file = data_file.set_index("time")
        data_file = data_file.drop(DROP_COLS, axis=1)
        data_file = data_file.replace(float("inf"), 0)
        cols = data_file.columns
        row_number = data_file.shape[0]

        for col in cols:
            count_nan = data_file[col].isna().sum()
            if count_nan >= row_number // 2:
                data_file = data_file.drop(col, axis=1)
                DROP_COLS.append(col)

        data_file = data_file.select_dtypes(include=[int, float])
        data_file = data_file.dropna()

        new_name = filename[:-4] + "_preprocessed.csv"
        save_path = os.path.join(output_path, new_name)
        data_file.to_csv(save_path)


if __name__ == "__main__":
    preprocess()
