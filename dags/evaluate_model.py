import yaml
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.sensors.external_task_sensor import ExternalTaskMarker, ExternalTaskSensor
from dotenv import load_dotenv


default_args = {
    "owner": "n3moy",
    "start_date": days_ago(1),
    "retries": 1,
    "max_active_tis_per_dag'": 1,
    "trigger_rule": "all_success",
    "max_active_runs": 1,
    "parallelism": 1
}

load_dotenv()
config_path = os.getenv("CONFIG_PATH_CLI")


with DAG(
    "evaluate_dag",
    default_args=default_args,
    description="Evaluate model efficiency",
    schedule_interval="30 * * * *",
    catchup=False
) as evaluate_dag:
    train_config = yaml.safe_load(open(config_path))["evaluate"]
    sensor_evaluate = ExternalTaskSensor(
        task_id="test_data_processed_2",
        external_dag_id="preprocess_predict_dag",
        external_task_id="maker_predict_preprocess_2",
        allowed_states=["success", "skipped"],
        failed_states=["failed"],
        mode="reschedule"
    )
    sensor_train_model = ExternalTaskSensor(
        task_id="model_trained",
        external_dag_id="train_dag",
        external_task_id="maker_train1",
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        mode="reschedule"
    )
    t1_name = "evaluate"
    t1 = BashOperator(
        task_id=t1_name,
        bash_command=f"python3 {train_config['src_dir']} {train_config['CLI_params']}"
    )

    [sensor_evaluate, sensor_train_model] >> t1

