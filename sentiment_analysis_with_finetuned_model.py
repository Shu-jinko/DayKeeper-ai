import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification
import numpy as np

MODEL_SAVE_PATH = 'fine-tuned-klue-bert-base'

tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

# 예측 함수
def classify_text(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1)
    predicted_class = tf.argmax(probs, axis=1).numpy()[0]
    return {
        "label": int(predicted_class),
        "scores": probs.numpy()[0].tolist()
    }

# 라벨 목록 (index 0부터 순서대로)
labels = ["unknown", "기쁨", "두려움", "분노", "슬픔"]

# 출력 포맷 함수
def pretty_print_result(result):
    pred_idx = result["label"]
    print(f"예측 감정: {labels[pred_idx]}")
    print("확률 분포:")
    for i, score in enumerate(result["scores"]):
        print(f"  {labels[i]}: {score:.2%}")

# 테스트
sample_text = """오늘은 너무 몸이 가벼운 하루. 아침에 일어났더니 너무 상쾌했고, 해야할 과제도 어제 다 끝내고 기분 최상이었다. 점심도 맛있었고, 저녁은 끝내주게 좋았다. 배가 너무 불러서 저녁때는 공원에서 산책도 함. 지금 tv보고 있는데 지락실이 너무 재밌다."""
result = classify_text(sample_text)
pretty_print_result(result)
