import yaml
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.sensors.external_task_sensor import ExternalTaskMarker, ExternalTaskSensor


default_args = {
    "owner": "n3moy",
    "start_date": days_ago(0),
    "retries": 3,
    "max_active_tis_per_dag'": 1,
    "trigger_rule": "all_success",
    "max_active_runs": 1
}

CONFIG_PATH = os.path.join("/c/py/predictive_esp/config/params_all.yaml")

pipelines = {
    "train": {"schedule": "5 * * * *"},
    "predict": {"schedule": "* * * * *"},
    "build_features": {"schedule": "5 * * * *"},
    "create_dataset": {"schedule": "5 * * * *"}
}

with DAG(
    "train",
    default_args=default_args,
    description="From raw to processed",
    schedule_interval=pipelines["create_dataset"]["schedule"]
) as train_dag:

    t1_id = "build_features"
    t1_params = yaml.safe_load(open(CONFIG_PATH))[t1_id]
    t1_script = t1_params["src_dir"]
    t1_args = t1_params["CLI_params"]
    t1 = BashOperator(
        task_id=t1_id,
        bash_command=f"python3 {t1_script} {t1_args}"
    )

    t2_id = "create_dataset"
    t2_params = yaml.safe_load(open(CONFIG_PATH))[t2_id]
    t2_script = t2_params["src_dir"]
    t2_args = t2_params["CLI_params"]
    t2 = BashOperator(
        task_id=t2_id,
        bash_command=f"python3 {t2_script} {t2_args}"
    )

    t3_id = "train"
    t3_params = yaml.safe_load(open(CONFIG_PATH))[t3_id]
    t3_script = t3_params["src_dir"]
    t3_args = t3_params["CLI_params"]
    train_task = BashOperator(
        task_id=t3_id,
        bash_command=f"python3 {t3_script} {t3_args}"
    )

    inter_task_1 = ExternalTaskMarker(
        task_id="inter_task_1",
        external_dag_id="predict_dag",
        external_task_id="inter_task_2"
    )

    t1 >> t2 >> train_task >> inter_task_1


with DAG(
    "predict",
    default_args=default_args,
    description="Predicts test.csv file",
    schedule_interval=pipelines["predict"]["schedule"]
) as predict_dag:

    inter_task_2 = ExternalTaskSensor(
        task_id="inter_task_2",
        external_dag_id=train_dag.dag_id,
        external_task_id=inter_task_1.task_id,
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
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

    inter_task_2 >> t1


# def init_dag(dag, task_id):
#     config = yaml.safe_load(open(CONFIG_PATH))[task_id]
#     src_dir = config["src_dir"]
#     CLI_params = config["CLI_params"]
#     with dag:
#         t1 = BashOperator(
#             task_id=f"{task_id}",
#             bash_command=f"python3 {src_dir} {CLI_params}"
#         )
#     return dag


# for task_id, params in pipelines.items():
#
#     dag = DAG(task_id,
#               schedule_interval=params['schedule'],
#               max_active_runs=1,
#               default_args=default_args
#               )
#     init_dag(dag, task_id)
#     globals()[task_id] = dag
