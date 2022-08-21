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

pipelines = {
    "train": {"schedule": "5 * * * *"},
    "predict": {"schedule": "* * * * *"},
    "build_features": {"schedule": "5 * * * *"},
    "create_dataset": {"schedule": "5 * * * *"}
}


with DAG(
    "extract_data_dag",
    default_args=default_args,
    description="From raw data to renamed and split",
    schedule_interval="5 * * * *",
    catchup=False
) as extract_data_dag:
    tasks_params = yaml.safe_load(open(CONFIG_PATH))

    t1_name = "train_test_split"
    t1 = BashOperator(
        task_id=t1_name,
        bash_command=f"python3 {tasks_params[t1_name]['src_dir']} {tasks_params[t1_name]['CLI_params']}"
    )
    maker_extract = ExternalTaskMarker(
        task_id="inter_task_extract",
        external_dag_id="preprocess_dag",
        external_task_id="inter_task_preprocess"
    )

    maker_extract.set_upstream(t1)


with DAG(
    "preprocess_dag",
    default_args=default_args,
    description="From renamed data to processed data",
    schedule_interval=pipelines["create_dataset"]["schedule"],
    catchup=False
) as preprocess_dag:
    tasks_params = yaml.safe_load(open(CONFIG_PATH))

    sensor_preprocess = ExternalTaskSensor(
        task_id="inter_task_preprocess",
        external_dag_id=extract_data_dag.dag_id,
        external_task_id=maker_extract.task_id,
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        mode="reschedule"
    )

    t1_name = "resample_data"
    t1 = BashOperator(
        task_id=t1_name,
        bash_command=f"python3 {tasks_params[t1_name]['src_dir']} {tasks_params[t1_name]['CLI_params']}"
    )
    t2_name = "merge_by_well"
    t2 = BashOperator(
        task_id=t2_name,
        bash_command=f"python3 {tasks_params[t2_name]['src_dir']} {tasks_params[t2_name]['CLI_params']}"
    )
    t3_name = "join_events"
    t3 = BashOperator(
        task_id=t3_name,
        bash_command=f"python3 {tasks_params[t3_name]['src_dir']} {tasks_params[t3_name]['CLI_params']}"
    )
    t4_name = "expand_target"
    t4 = BashOperator(
        task_id=t4_name,
        bash_command=f"python3 {tasks_params[t4_name]['src_dir']} {tasks_params[t4_name]['CLI_params']}"
    )
    t5_name = "build_features"
    t5 = BashOperator(
        task_id=t5_name,
        bash_command=f"python3 {tasks_params[t5_name]['src_dir']} {tasks_params[t5_name]['CLI_params']}"
    )
    t6_name = "merge"
    t6 = BashOperator(
        task_id=t6_name,
        bash_command=f"python3 {tasks_params[t6_name]['src_dir']} {tasks_params[t6_name]['CLI_params']}"
    )
    t7_name = "clear_nulls"
    t7 = BashOperator(
        task_id=t7_name,
        bash_command=f"python3 {tasks_params[t7_name]['src_dir']} {tasks_params[t7_name]['CLI_params']}"
    )
    t8_name = "create_dataset"
    t8 = BashOperator(
        task_id=t8_name,
        bash_command=f"python3 {tasks_params[t8_name]['src_dir']} {tasks_params[t8_name]['CLI_params']}"
    )
    maker_preprocess = ExternalTaskMarker(
        task_id="maker_preprocess",
        external_dag_id="train_dag",
        external_task_id="sensor_train"
    )

    t1.set_upstream(sensor_preprocess)
    t2.set_upstream(t1)
    t3.set_upstream(t2)
    t4.set_upstream(t3)
    t5.set_upstream(t2)
    t6.set_upstream(t5)
    t6.set_upstream(t4)
    t7.set_upstream(t6)
    t8.set_upstream(t7)
    maker_preprocess.set_upstream(t8)

    # t1 >> t2 >> t3 >> [t3 >> t4 >> t5, t6] >> t7 >> t8 >> t9


with DAG(
    "train_dag",
    default_args=default_args,
    description="From processed data to trained model",
    schedule_interval=pipelines["pipelines"]["schedule"],
    catchup=False
) as train_dag:
    train_config = yaml.safe_load(open(CONFIG_PATH))["train"]
    sensor_train = ExternalTaskSensor(
        task_id="sensor_train",
        external_dag_id=preprocess_dag.dag_id,
        external_task_id=maker_preprocess.task_id,
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        mode="reschedule"
    )
    t1_name = "train"
    t1 = BashOperator(
        task_id=t1_name,
        bash_command=f"python3 {train_config['src_dir']} {train_config['CLI_params']}"
    )

    maker_train = ExternalTaskMarker(
        task_id="maker_train",
        external_dag_id="predict_dag",
        external_task_id="sensor_predict"
    )

    sensor_train >> t1 >> maker_train


with DAG(
    "predict_dag",
    default_args=default_args,
    description="Predicts test.csv file",
    schedule_interval=pipelines["predict"]["schedule"],
    catchup=False
) as predict_dag:

    sensor_predict = ExternalTaskSensor(
        task_id="sensor_predict",
        external_dag_id=preprocess_dag.dag_id,
        external_task_id=maker_train.task_id,
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

    sensor_predict >> t1


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
