import yaml
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.sensors.external_task_sensor import ExternalTaskMarker, ExternalTaskSensor


default_args = {
    "owner": "n3moy",
    "start_date": days_ago(1),
    "retries": 1,
    "max_active_tis_per_dag'": 1,
    "trigger_rule": "all_success",
    "max_active_runs": 1,
    "parallelism": 1
}

CONFIG_PATH = os.path.join("/c/py/predictive_esp/config/params_all.yaml")


with DAG(
    "predict_dag",
    default_args=default_args,
    description="Predicts test.csv file",
    schedule_interval="5 * * * *",
    catchup=False
) as predict_dag:

    sensor_predict = ExternalTaskSensor(
        task_id="sensor_predict",
        external_dag_id="preprocess_predict_dag",
        external_task_id="maker_predict_preprocess",
        allowed_states=["success", "skipped"],
        failed_states=["failed"],
        mode="reschedule"
    )

    t1_id = "predict"
    t1_params = yaml.safe_load(open(CONFIG_PATH))[t1_id]
    t1_script = t1_params["src_dir"]
    t1_args = t1_params["CLI_params"]
    t1 = BashOperator(
        task_id=t1_id,
        bash_command=f"python3 {t1_script} {t1_args}"
    )

    sensor_predict >> t1

