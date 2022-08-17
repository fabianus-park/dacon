# 출처 : https://minimin2.tistory.com/137

from pycaret.datasets import get_data 
from pycaret.classification import *
from pycaret.utils import check_metric

# dataset 생성
dataset = get_data('juice')

# setup으로 init
clf = setup(data=dataset, target='Purchase')

# ML 모델 성능 비교
top5 = compare_models(sort='Accuracy', n_select=5)
# 모델 튜닝, blending
tuned_top5 = [tune_model(i) for i in top5]
blender_top5 = blend_models(estimator_list=tuned_top5)

# 모델 최종 확정, 전체 데이터 학습
final_model = finalize_model(blender_top5)
# 테스트셋 predict
prediction = predict_model(final_model, data=dataset.iloc[-100:])

# metric 확인
check_metric(prediction['Purchase'], prediction['Label'], metric = 'Accuracy')
