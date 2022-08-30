import os

import click
import pandas as pd
import numpy as np
# import yaml

from feature_builder import FeatureCalculation

COLS_TO_CALC = None
CONFIG_PATH = "/c/py/predictive_esp/config/params_all.yaml"


# This func should be used after joining data by well
@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def build_features(
    input_path: str,
    output_path: str
) -> None:
    """
    Calculating unbalances, derivatives, min, max, mean, std, spike
    of operating parameters with window as new features

    """
    # Train or test, we can confirm that by folder name
    global COLS_TO_CALC

    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        data_file = pd.read_csv(file_path, parse_dates=["time"])
        data_file = data_file.set_index("time")
        data_file = FeatureCalculation.reduce_mem_usage(data_file)

        # Current and voltage unbalance
        voltage_names = ["voltageAB", "voltageBC", "voltageCA"]
        current_names = ["op_current1", "op_current2", "op_current3"]
        voltages = data_file[voltage_names]
        currents = data_file[current_names]
        mean_voltage = voltages.mean(axis=1)
        mean_current = currents.mean(axis=1)
        deviation_voltage = voltages.sub(mean_voltage, axis=0).abs()
        deviation_current = currents.sub(mean_current, axis=0).abs()

        data_file["voltage_unbalance"] = (
                deviation_voltage.max(axis=1).div(mean_voltage, axis=0) * 100
        )
        data_file["current_unbalance"] = (
                deviation_current.max(axis=1).div(mean_current, axis=0) * 100
        )

        # Impute zeros where currents are zeros
        data_file["current_unbalance"] = data_file["current_unbalance"].fillna(0)
        # Impute zeros where voltages are zeros
        data_file["voltage_unbalance"] = data_file["voltage_unbalance"].fillna(0)
        # I don't need currents anymore cause active power present variability
        # Lets keep only one voltage and current to save variability
        data_file["voltage"] = data_file["voltageAB"]
        data_file["current"] = data_file["op_current1"]
        data_file["resistance"] = np.where(
            (data_file["current"] == 0), 0, data_file["voltage"].div(data_file["current"], axis=0)
        )

        # Testing all ideas to choose best ones
        data_file["power_A"] = data_file["op_current1"] * data_file["voltageAB"] / 1000
        data_file["power_B"] = data_file["op_current2"] * data_file["voltageBC"] / 1000
        data_file["power_C"] = data_file["op_current3"] * data_file["voltageCA"] / 1000
        data_file["theory_power"] = data_file["power_A"] + data_file["power_B"] + data_file["power_C"]
        data_file["power_diff"] = data_file["active_power"] - data_file["theory_power"]

        data_file["power_lossesA"] = np.power(data_file["op_current1"], 2) * data_file["resistance"]
        data_file["power_lossesB"] = np.power(data_file["op_current2"], 2) * data_file["resistance"]
        data_file["power_lossesC"] = np.power(data_file["op_current3"], 2) * data_file["resistance"]

        # data_file["watercut"] = 1 - data_file["oil_rate"] / data_file["liquid_rate"]
        data_file["pressure_drop"] = data_file["intake_pressure"] - data_file["line_pressure"]
        data_file["theory_rate"] = data_file["pressure_drop"] * 8
        data_file["rate_diff"] = data_file["liquid_rate"] - data_file["theory_rate"]

        data_file["freq_ratio"] = data_file["frequency"] / 50
        data_file["freq_squared_ratio"] = np.power(data_file["frequency"] / 50, 2)
        data_file["freq_cubic_ratio"] = np.power(data_file["frequency"] / 50, 3)
        k = 500
        data_file["skin"] = (k - data_file["intake_pressure"] - data_file["liquid_rate"]) / data_file["liquid_rate"]

        # Calculating derivatives and statistics
        if COLS_TO_CALC is None:
            COLS_TO_CALC = [
                "current",
                "voltage",
                "active_power",
                "frequency",
                "electricity_gage",
                # "motor_load",    # Mistake in initial data_file, so I don't need it here. Should be resolved some day
                "pump_temperature",
            ]

        windows = [60 * 14 * 3]
        # windows = [60 * 14 * 3, 60 * 14 * 1, 60 * 14 * 7]
        for col in COLS_TO_CALC:
            for window in windows:
                data_file[f"{col}_rol_mean_{window}"] = (
                    data_file[col].rolling(min_periods=1, window=window).mean()
                )
                data_file[f"{col}_rol_std_{window}"] = (
                    data_file[col].rolling(min_periods=1, window=window).std()
                )
                data_file[f"{col}_rol_max_{window}"] = (
                    data_file[col].rolling(min_periods=1, window=window).max()
                )
                data_file[f"{col}_rol_min_{window}"] = (
                    data_file[col].rolling(min_periods=1, window=window).min()
                )
                data_file[f"{col}_spk_{window}"] = np.where(
                    (data_file[f"{col}_rol_mean_{window}"] == 0), 0, data_file[col] / data_file[f"{col}_rol_mean_{window}"]
                )
            data_file[f"{col}_deriv"] = pd.Series(np.gradient(data_file[col]), data_file.index)
            data_file[col] = data_file[col].rolling(min_periods=1, window=30).mean()
            data_file[f"{col}_squared"] = np.power(data_file[col], 2)
            data_file[f"{col}_root"] = np.power(data_file[col], 0.5)

        new_name = filename[:-4] + "_featured.csv"
        save_path = os.path.join(output_path, new_name)
        data_file.to_csv(save_path)

        # cols_to_drop = [
        #     "reagent_rate",
        #     "oil_rate",
        #     "gas_rate",
        #     "motor_temperature",
        #     *voltage_names,
        #     *current_names,
        # ]
        # cols_to_drop = [col for col in cols_to_drop.copy() if col in initial_cols]
        # data_file = data_file.drop(cols_to_drop, axis=1)


if __name__ == "__main__":
    build_features()

