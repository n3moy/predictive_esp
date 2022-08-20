import pandas as pd
import numpy as np
import click


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("valid_size", type=click.FLOAT())
@click.argument("random_seed", type=click.INT())
def validation_split(
    input_data: pd.DataFrame,
    valid_size: float,
    random_state: int = None
) -> list:
    if random_state:
        np.random.seed(random_state)
    data = input_data.copy()
    target_values = input_data["failure_target"].values
    data_size = data.shape[0]
    split_size = int(data_size * valid_size)
    split_point = np.random.randint(0, data_size-split_size-1)
    # Due to similarity in data between observations with '1' class, data may leak
    # So its better to begin with '0' signal to check quality and train properly
    check_flag = target_values[split_point] == 1

    while check_flag:
        split_point += 1
        check_flag = target_values[split_point] == 1
    val_indices = list(range(split_point, split_point + split_size))

    return val_indices

    # val_data = data.iloc[val_indices]
    # train_data = data.loc[~data.index.isin(val_indices)]
    # val_data = val_data.reset_index(drop=True)
    # train_data = train_data.reset_index(drop=True)

