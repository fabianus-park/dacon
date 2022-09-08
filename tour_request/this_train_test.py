import pandas as pd
# train = pd.read_csv('data/train.csv')
train = pd.read_csv('./tour_request/train_preprocessed.csv')
test = pd.read_csv('./tour_request/test_preprocessed.csv')


train = train.drop(columns=['TypeofContact','Occupation','NumberOfPersonVisiting','NumberOfTrips','OwnCar','NumberOfChildrenVisiting'])
test  = test.drop(columns=['TypeofContact','Occupation','NumberOfPersonVisiting','NumberOfTrips','OwnCar','NumberOfChildrenVisiting'])


### 3. Modeling
#### 1. 모델 선택
from sklearn.ensemble import RandomForestClassifier

# 모델 선언
model = RandomForestClassifier()


#### 2. 학습/예측
# 분석할 의미가 없는 칼럼을 제거합니다.
# train = train.drop(columns=['id'])
# test = test.drop(columns=['id'])

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
y_train = train[['ProdTaken']]
x_train = train.drop(columns=['ProdTaken'])

# 모델 학습
model.fit(x_train,y_train)


# 학습된 모델을 이용해 결과값 예측후 상위 10개의 값 확인
prediction = model.predict(test)
print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')
print(prediction[:10])


### IV. 데이콘 제출하기
#### 1. submission 파일 생성

# sample_submission 불러오기
sample_submission = pd.read_csv('./tour_request/data/sample_submission.csv')
# print(sample_submission.info())

# 예측된 값을 정답파일과 병합
sample_submission['ProdTaken'] = prediction

# 정답파일 데이터프레임 확인
print( sample_submission.head() )

# submission을 csv 파일로 저장합니다.
# index=False란 추가적인 id를 부여할 필요가 없다는 뜻입니다. 
# 정확한 채점을 위해 꼭 index=False를 넣어주세요.
sample_submission.to_csv('./tour_request/submission.csv',index = False)

