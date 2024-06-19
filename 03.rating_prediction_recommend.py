import pandas as pd
import numpy as np
from surprise import Dataset, Reader, BaselineOnly
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

df_correspond = pd.read_csv('./df_correspond.csv')
df_correspond.columns = ['user_id', 'vod_id', 'rating']
traindata, testdata = train_test_split(df_correspond, test_size= 0.5 , random_state= 42)

# Reader 객체 생성 (rating_scale은 평점의 최소값과 최대값을 지정)
reader = Reader(rating_scale=(1, 5))

# 데이터셋 생성
traindata = Dataset.load_from_df(traindata[['user_id', 'vod_id', 'rating']], reader)
testdata = Dataset.load_from_df(testdata[['user_id', 'vod_id', 'rating']], reader)

# 하이퍼파라미터 그리드 설정
bsl_options =  {'method': 'sgd',
                'reg': 0.05,
                'learning_rate': 0.01}

algo = BaselineOnly(bsl_options=bsl_options)

trainset = traindata.build_full_trainset()
testset = testdata.build_full_trainset()
testset = testset.build_testset()
algo.fit(trainset)
predictions = algo.test(testset)
predictions_df = pd.DataFrame(predictions, columns=['user_id', 'vod_id', 'r_ui', 'est', 'details'])
predictions_df = predictions_df[['user_id', 'vod_id', 'r_ui', 'est']]

def get_top_n_recommendations(algo, user_movie_rating_df, user_id, n=20):
    # 모든 영화 목록
    all_vod_ids = user_movie_rating_df['vod_id'].unique()
    
    # 사용자가 이미 본 영화 목록
    seen_vod_ids = user_movie_rating_df[user_movie_rating_df['user_id'] == user_id]['vod_id'].unique()
    
    # 사용자가 보지 않은 영화 목록
    unseen_vod_ids = [movie_id for movie_id in all_vod_ids if movie_id not in seen_vod_ids]
    
    # 사용자가 보지 않은 영화들에 대한 예측 평점 계산
    predictions = [algo.predict(user_id, vod_id) for vod_id in unseen_vod_ids]
    
    # 예측 평점이 높은 순으로 정렬하여 상위 n개 영화 추천
    top_n_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    
    # 추천 영화 ID 리스트 추출
    top_n_vod_ids = [pred.iid for pred in top_n_predictions]

    return top_n_vod_ids

# 추천 생성
recommend_for_id = {}
for user_id in df_correspond['user_id'].values:
    recommend_for_id[user_id] = get_top_n_recommendations(algo, df_correspond, user_id, 20)

vod_id_list = recommend_for_id[1]
print(vod_id_list)

# print(recommend_for_id)

#MongoDB 추천 vod_list 업데이트
client = MongoClient("mongodb://wang:0131@3.37.201.211:27017")

db = client['hellody']
movies = db['MOVIES']
recommend_list = db['recommend_list']

# MongoDB 추천 VOD 리스트 업데이트
for user_id, vod_id_list in recommend_for_id.items():
    result = []
    for vod_id in vod_id_list :
        movie = movies.find_one(
            { "VOD_ID": int(vod_id) },
            { "_id": 0, "VOD_ID":1,"TITLE": 1, "POSTER": 1 }
        )
        result.append(movie)
    # print(result)

    recommend_list.update_one(
              { "user_id": int(user_id) },
              { 
                "$set": {
                    "review_rating_based": result
          }
      },
      upsert=True  # 존재하지 않는 경우 새로 삽입
    )

print("MongoDB 업데이트 완료")