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

load_dotenv()

# AWS 자격 증명 로드
S3_ACCESS_KEY = os.getenv("AWS_S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("AWS_S3_SECRET_KEY")

# 데이터 로드 및 전처리
# S3 클라이언트 생성
s3 = boto3.client("s3", aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY)

# S3 버킷과 파일 경로 설정
bucket = 'spotifymodel'
s3_dir = 'review-data/reviewdata.csv'

# S3에서 파일을 읽어와서 pandas 데이터프레임으로 변환
def read_s3_csv_to_dataframe(bucket, s3_dir):
    # S3에서 파일 가져오기
    response = s3.get_object(Bucket=bucket, Key=s3_dir)
    content = response['Body'].read()
    
    # 파일의 인코딩 감지
    result = chardet.detect(content)
    encoding = result['encoding']
    
    # 컬럼명 정의
    column_names = ["user_id", "vod_id", "comment", "rating"]
    if encoding is None:
       encoding = 'cp949'
    # 데이터프레임으로 읽기
    df = pd.read_csv(io.StringIO(content.decode(encoding, errors='ignore')), names=column_names)
    
    return df

# 데이터프레임 읽기
df = read_s3_csv_to_dataframe(bucket, s3_dir)

# 데이터프레임 확인
print(df.head())

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


#한국어 제외 제거
def text_cleaning(text) :
  hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
  result = hangul.sub('', text)
  return result

#토큰화
okt = Okt()
def get_token(x) :
  tokenized_sentence = okt.morphs(x, stem = True)
  stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
  return stopwords_removed_sentence

df.dropna(axis= 0, how= 'any', inplace= True)
df['ko_review'] = df['comment'].apply(lambda x : text_cleaning(x))
del df['comment']
df['ko_review'] = df['ko_review'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
df['ko_review'].replace('', np.nan, inplace=True)
print(df.isnull().sum()) 
df.dropna(axis= 0, how= 'any', inplace= True)
df.reset_index(inplace=True, drop=True)
df['ko_review'] = df['ko_review'].apply(get_token) 
drop_train = [index for index, sentence in enumerate(df['ko_review']) if len(sentence) < 1]
# len(drop_train) #
df.drop(index= drop_train, axis= 0, inplace= True)
# print(df) #

# 토크나이저 불러오기
with open('/home/ubuntu/review_rating_s3/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 테스트 데이터 전처리 및 토크나이저 적용
X = tokenizer.texts_to_sequences(df['ko_review'])

vocab_size = len(tokenizer.word_index) + 1
# vocab_size #

X = pad_sequences(X, maxlen= 43)

# 저장된 모델 로드
loaded_model = load_model('/home/ubuntu/review_rating_s3/best_model_LSTM_30.keras')

# 로드한 모델로 예측 수행
predictions = loaded_model.predict(X)

# 모델 요약 정보 출력
loaded_model.summary()

# 예측 결과를 긍정/부정으로 변환
threshold = 0.5
predicted_labels = (predictions > threshold).astype(int)

# predicted_labels를 데이터프레임의 새로운 열로 추가
df['model_label'] = predicted_labels

# 결과 출력
# df[df['model_label'] == 0] #8304
# df[df['model_label'] == 1] #18117

df['rating'] = df['rating'].astype(float)
df['rating_label'] = df['rating'].apply(lambda x: 1 if x > 5 else 0)
# print(df)

positive_ratings = df[df['rating_label'] == 1]
negative_ratings = df[df['rating_label'] == 0]
# print(len(positive_ratings)) #21897
# print(len(negative_ratings)) #4524
# df : 26520

df_correspond = df[(df['model_label'] == df['rating_label'])]
# print(df_correspond)
df_correspond = df[['user_id', 'vod_id', 'rating']]
df_correspond.to_csv('/home/ubuntu/review_rating_s3/df_correspond.csv', index = False)
