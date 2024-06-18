import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import os
import boto3
import chardet
import io
from surprise import Dataset, Reader, BaselineOnly
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from airflow import DAG
import airflow.utils.dates
from airflow.operators.bash import BashOperator
from airflow.operatos.python import PythonOperator
from datetime import datetime, timedelta

dag = DAG(
  dag_id = 'review_rating',
  description = 'review_rating_based_recommend',
  start_date = airflow.utils.dates.dats_ago(1),
  schedule_interval = timedelta(days = 1)
)

def 
