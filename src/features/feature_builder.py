import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime


class FeatureCalculation:
    @staticmethod
    def reduce_mem_usage(data_in: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Reduces pd.DataFrame memory usage based on columns types

        """
        numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
        start_mem = data_in.memory_usage().sum() / 1024 ** 2

        for col in data_in.columns:
            col_type = data_in[col].dtypes
            if col_type in numerics:
                c_min = data_in[col].min()
                c_max = data_in[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        data_in[col] = data_in[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        data_in[col] = data_in[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        data_in[col] = data_in[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        data_in[col] = data_in[col].astype(np.int64)
                else:
                    if (
                            c_min > np.finfo(np.float32).min
                            and c_max < np.finfo(np.float32).max
                    ):
                        data_in[col] = data_in[col].astype(np.float32)
                    else:
                        data_in[col] = data_in[col].astype(np.float64)
        end_mem = data_in.memory_usage().sum() / 1024 ** 2
        if verbose:
            print(
                "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                    end_mem, 100 * (start_mem - end_mem) / start_mem
                )
            )
        return data_in

    @staticmethod
    def expand_target(data_in: pd.DataFrame, target_window=7) -> pd.DataFrame:
        """
        Based on 'target_window' creates additional variable as target in advance to failure
        For example if 'target_window'=7 -> all observations (rows) before failure in 7 days are signed as class '1'

        """
        data = data_in.copy()
        if is_datetime(data.index):
            data = data.reset_index()
        if not is_datetime(data["time"]):
            data["time"] = pd.to_datetime(data["time"])
        data.loc[(data["event_id"] != 0), "failure_date"] = data.loc[
            (data["event_id"] != 0), "time"
        ]
        data["failure_date"] = data["failure_date"].bfill().fillna(data["time"].max())
        data["failure_date"] = pd.to_datetime(data["failure_date"])
        data["fail_range"] = data["failure_date"] - data["time"]
        data["time_to_failure"] = data["fail_range"] / np.timedelta64(1, "D")
        data.loc[data["failure_date"] == data["time"].max(), "time_to_failure"] = 999
        # I use window between 7 and 3 days before failure
        data["failure_target"] = np.where(
            ((data["time_to_failure"] <= target_window) & (data["time_to_failure"] > 3)),
            1,
            0,
        )
        data = data.drop(["failure_date", "fail_range"], axis=1)
        # data["stable"] = np.where(((data["time_to_failure"] >= 30) & (data["time_to_failure"] <= 90)), 1, 0)
        data = data[data["time_to_failure"] != 999]
        data["stable"] = np.where((data["time_to_failure"] >= 20), 1, 0)
        return data

    @staticmethod
    def join_events_to_data(data_in: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """
        This function assigns multiple events as marks into dataset based on startDate and endDate in events dataframe

        """
        out_data = data_in.copy()
        # if not is_datetime(out_data.index):
        #     # out_data["time"] = pd.to_datetime(out_data["time"])
        #     out_data.set_index("time")

        events_dates = events[["startDate", "endDate"]].values
        events_id = events["result"].values
        out_data["event_id"] = 0

        for ev_id, (start_date, end_date) in zip(events_id, events_dates):
            mask = (out_data.index >= start_date) & (out_data.index <= end_date)
            out_data.loc[mask, "event_id"] = ev_id

        out_data = out_data.reset_index()
        return out_data

    @staticmethod
    def calculate_data_features(
            input_data: pd.DataFrame, cols_to_calc: list = None
    ) -> pd.DataFrame:
        """
        Calculating unbalances, derivatives, min, max, mean, std, spike
        of operating parameters with window as new features

        """
        data = input_data.copy()
        initial_cols = data.columns
        # Current and voltage unbalance
        voltage_names = ["voltageAB", "voltageBC", "voltageCA"]
        current_names = ["op_current1", "op_current2", "op_current3"]
        voltages = data[voltage_names]
        currents = data[current_names]
        mean_voltage = voltages.mean(axis=1)
        mean_current = currents.mean(axis=1)
        deviation_voltage = voltages.sub(mean_voltage, axis=0).abs()
        deviation_current = currents.sub(mean_current, axis=0).abs()

        data["voltage_unbalance"] = (
                deviation_voltage.max(axis=1).div(mean_voltage, axis=0) * 100
        )
        data["current_unbalance"] = (
                deviation_current.max(axis=1).div(mean_current, axis=0) * 100
        )

        # Impute zeros where currents are zeros
        data["current_unbalance"] = data["current_unbalance"].fillna(0)
        # Impute zeros where voltages are zeros
        data["voltage_unbalance"] = data["voltage_unbalance"].fillna(0)
        # I don't need currents anymore cause active power present variability
        # Lets keep only one voltage and current to save variability
        data["voltage"] = data["voltageAB"]
        data["current"] = data["op_current1"]
        data["resistance"] = np.where(
            (data["current"] == 0), 0, data["voltage"].div(data["current"], axis=0)
        )
        # Пробные признаки, ебашим все что есть
        data["power_A"] = data["op_current1"] * data["voltageAB"] / 1000
        data["power_B"] = data["op_current2"] * data["voltageBC"] / 1000
        data["power_C"] = data["op_current3"] * data["voltageCA"] / 1000
        data["theory_power"] = data["power_A"] + data["power_B"] + data["power_C"]
        data["power_diff"] = data["active_power"] - data["theory_power"]

        data["power_lossesA"] = np.power(data["op_current1"], 2) * data["resistance"]
        data["power_lossesB"] = np.power(data["op_current2"], 2) * data["resistance"]
        data["power_lossesC"] = np.power(data["op_current3"], 2) * data["resistance"]

        # data["watercut"] = 1 - data["oil_rate"] / data["liquid_rate"]
        data["pressure_drop"] = data["intake_pressure"] - data["line_pressure"]
        data["theory_rate"] = data["pressure_drop"] * 8
        data["rate_diff"] = data["liquid_rate"] - data["theory_rate"]

        data["freq_ratio"] = data["frequency"] / 50
        data["freq_squared_ratio"] = np.power(data["frequency"] / 50, 2)
        data["freq_cubic_ratio"] = np.power(data["frequency"] / 50, 3)
        k = 500
        data["skin"] = (k - data["intake_pressure"] - data["liquid_rate"]) / data["liquid_rate"]

        # Calculating derivatives and statistics
        if cols_to_calc is None:
            cols_to_calc = [
                "current",
                "voltage",
                "active_power",
                "frequency",
                "electricity_gage",
                # "motor_load",         # Mistake in initial data, so I don't need it here. Should be resolved some day
                "pump_temperature",
            ]

        windows = [60 * 14 * 3]
        # windows = [60 * 14 * 3, 60 * 14 * 1, 60 * 14 * 7]
        for col in cols_to_calc:
            for window in windows:
                data[f"{col}_rol_mean_{window}"] = (
                    data[col].rolling(min_periods=1, window=window).mean()
                )
                data[f"{col}_rol_std_{window}"] = (
                    data[col].rolling(min_periods=1, window=window).std()
                )
                data[f"{col}_rol_max_{window}"] = (
                    data[col].rolling(min_periods=1, window=window).max()
                )
                data[f"{col}_rol_min_{window}"] = (
                    data[col].rolling(min_periods=1, window=window).min()
                )
                data[f"{col}_spk_{window}"] = np.where(
                    (data[f"{col}_rol_mean_{window}"] == 0), 0, data[col] / data[f"{col}_rol_mean_{window}"]
                )
            data[f"{col}_deriv"] = pd.Series(np.gradient(data[col]), data.index)
            data[col] = data[col].rolling(min_periods=1, window=30).mean()
            data[f"{col}_squared"] = np.power(data[col], 2)
            data[f"{col}_root"] = np.power(data[col], 0.5)

        cols_to_drop = [
            "reagent_rate",
            "oil_rate",
            "gas_rate",
            "motor_temperature",
            *voltage_names,
            *current_names,
        ]
        cols_to_drop = [col for col in cols_to_drop.copy() if col in initial_cols]
        data = data.drop(cols_to_drop, axis=1)
        data = FeatureCalculation.reduce_mem_usage(data)
        return data
