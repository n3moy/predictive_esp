import yaml
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.sensors.external_task_sensor import ExternalTaskMarker, ExternalTaskSensor


default_args = {
    "owner": "n3moy",
    "start_date": days_ago(1),
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
    "preprocess",
    default_args=default_args,
    description="From raw data to processed data",
    schedule_interval=pipelines["create_dataset"]["schedule"],
    catchup=False
) as train_dag:
    tasks_params = yaml.safe_load(open(CONFIG_PATH))

    t1_name = "train_test_split"
    t1 = BashOperator(
        task_id=t1_name,
        bash_command=f"python3 {tasks_params[t1_name]['src_params']} {tasks_params[t1_name]['CLI_params']}"
    )

    t2_name = "resample_data"
    t2 = BashOperator(
        task_id=t2_name,
        bash_command=f"python3 {tasks_params[t2_name]['src_params']} {tasks_params[t2_name]['CLI_params']}"
    )

    t3_name = "merge_by_well"
    t3 = BashOperator(
        task_id=t3_name,
        bash_command=f"python3 {tasks_params[t3_name]['src_params']} {tasks_params[t3_name]['CLI_params']}"
    )

    t4_name = "join_events"
    t4 = BashOperator(
        task_id=t4_name,
        bash_command=f"python3 {tasks_params[t4_name]['src_params']} {tasks_params[t4_name]['CLI_params']}"
    )

    t5_name = "expand_target"
    t5 = BashOperator(
        task_id=t5_name,
        bash_command=f"python3 {tasks_params[t5_name]['src_params']} {tasks_params[t5_name]['CLI_params']}"
    )

    t6_name = "build_features"
    t6 = BashOperator(
        task_id=t6_name,
        bash_command=f"python3 {tasks_params[t6_name]['src_params']} {tasks_params[t6_name]['CLI_params']}"
    )

    t7_name = "merge"
    t7 = BashOperator(
        task_id=t7_name,
        bash_command=f"python3 {tasks_params[t7_name]['src_params']} {tasks_params[t7_name]['CLI_params']}"
    )

    t8_name = "preprocess"
    t8 = BashOperator(
        task_id=t8_name,
        bash_command=f"python3 {tasks_params[t8_name]['src_params']} {tasks_params[t8_name]['CLI_params']}"
    )

    t9_name = "create_dataset"
    t9 = BashOperator(
        task_id=t8_name,
        bash_command=f"python3 {tasks_params[t9_name]['src_params']} {tasks_params[t9_name]['CLI_params']}"
    )

    inter_task_1 = ExternalTaskMarker(
        task_id="inter_task_1",
        external_dag_id="predict_dag",
        external_task_id="inter_task_2"
    )

    t1 >> t2 >> t3 >> [[t4 >> t5], t6] >> t7 >> t8 >> t9


with DAG(
    "predict",
    default_args=default_args,
    description="Predicts test.csv file",
    schedule_interval=pipelines["predict"]["schedule"],
    catchup=False
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
