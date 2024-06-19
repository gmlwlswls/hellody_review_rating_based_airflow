from airflow import DAG
import airflow.utils.dates
from airflow.operators.bash import BashOperator
from airflow.operatos.python import PythonOperator
from datetime import datetime, timedelta

dag = DAG(
  dag_id = 'review_rating',
  description = 'review_rating_based_recommend',
  start_date = airflow.utils.dates.days_ago(1),
  schedule_interval = timedelta(days = 1)
)

t1 = BashOperator(
  task_id = 'review_classification',
  bash_command = 'python3 /home/ubuntu/review_rating_s3/review_classification.py'
)

t2 = BashOperator(
  task_id = 'rating_prediction_recommend',
  bash_command = 'python3 /home/ubuntu/review_rating_s3/rating_prediction_recommend.py'
)

notify = BashOperator(
  task_id = 'notify',
  bash_command = 'echo "ALL Done',
)


t1 >> t2 >> notify


import airflow.utils.dates
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# DAG 인스턴스 생성
with DAG(
    dag_id="review_rating",
    description="review_rating_based_recommendations",
    start_date= airflow.utils.dates.days_ago(1)
    schedule_interval=timedelta(days=1),  # 하루에 한번씩 실행
    catchup=False,  # 과거의 DAG 실행 방지
) as dag:

  t1 = BashOperator(
    task_id = 'review_classification',
    bash_command = 'python3 /home/ubuntu/review_rating_s3/review_classification.py'
  )
  t2 = BashOperator(
    task_id = 'rating_prediction_recommend',
    bash_command = 'python3 /home/ubuntu/review_rating_s3/rating_prediction_recommend.py'
  )

  notify = BashOperator(
    task_id = 'notify',
    bash_command = 'echo "ALL Done',
  )

  t1 >> t2 >> notify
