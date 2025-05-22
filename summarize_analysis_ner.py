# -*- coding: utf-8 -*-
from openai import OpenAI
from pydantic import BaseModel
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification
from dotenv import load_dotenv
import os
from gliner import GLiNER
import MeCab
import re

# openai key 설정
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = openai_api_key
client = OpenAI(api_key=api_key)

# BERT 모델 및 토크나이저 로드
MODEL_SAVE_PATH = 'fine-tuned-klue-bert-base'
tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

# 감정 목록
labels = ["unknown", "기쁨", "두려움", "분노", "슬픔"]

# 사용자 입력 받기
user_input = input("오늘 하루를 입력해 주세요: ")

# 자유 형식 글 -> 주제별 정리 (글에 대해 다수의 주제로 정리하도록 유도)
class Each_Paragraph(BaseModel):
    subject: str
    content: str

class Final_Result(BaseModel):
    final:list[Each_Paragraph]

response = client.responses.parse(
    model="gpt-4.1",
    input=[
        {
            "role": "system",
            "content": """
                    주어진 글은 하루 동안 있었던 일을 자유 형식으로 적은 글입니다.
                    이 글을 다음 조건에 따라 주제별로 정리해 주세요:
                    1. 원문에 없는 내용을 새로 덧붙이지 마세요.
                    2. 필요하다면 문장 순서를 조금 바꾸거나 간단한 문장 다듬기는 허용합니다.
                    3. 주제별로 내용을 묶고, 문단을 나눠 정리해주세요. 각 문단의 주제는 드러나도록 구성해주세요.  
                    4. 전체 톤과 말투는 원문의 느낌을 유지해주세요. 단, 문장은 가능한 완전한 문장으로 작성해주세요.
                    5. 오타가 있다면 수정해주세요.
                    """
        },
        {
            "role": "user",
            "content": user_input,
        },
    ],
    text_format=Final_Result
)

result=response.output_parsed
result_json = result.model_dump_json()
# 모델 활용 감정 분석
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


# 감정/확률 출력 함수
def print_emotion(result):
    pred_idx = result["label"]
    print(f"예측 감정: {labels[pred_idx]}")
    print("확률 분포:")
    for i, score in enumerate(result["scores"]):
        print(f"  {labels[i]}: {score:.2%}")

# 출력
# for i in result.final:
#     print(i.subject)
#     print(i.content)

print(result_json)

result = classify_text(user_input)
print_emotion(result)


# 모델 불러오기
model = GLiNER.from_pretrained("urchade/gliner_multiv2.1")

# MeCab-ko-msvc 태거 설정
tagger = MeCab.Tagger()

# 명사 추출 함수
def extract_nouns(text):
    result = []
    parsed = tagger.parse(text)
    for line in parsed.split('\n'):
        if line == 'EOS' or line == '':
            continue
        surface, features = line.split('\t')
        pos = features.split(',')[0]
        if pos in ('NNG', 'NNP'):  # 일반명사, 고유명사
            result.append(surface)
    return result

# 분석 대상 텍스트
text = user_input

# 개체명 라벨 정의
labels = ["date", "person", "location", "organization", "event", "time", "family"]

# 문장 단위 분할
sentences = re.split(r'(?<=[.!?])\s+', text.strip())

# 개체 예측 및 명사 정제
all_entities = []

for sent in sentences:
    entities = model.predict_entities(sent, labels, threshold=0.5)
    for entity in entities:
        noun_list = extract_nouns(entity["text"])
        if noun_list:  # 명사 없으면 스킵
            cleaned = ' '.join(noun_list)
            all_entities.append({
                "text": cleaned,
                "label": entity["label"]
            })

# 중복 제거
unique_entities = { (e["text"], e["label"]) for e in all_entities }

# 출력
for text_label in unique_entities:
    print(text_label[0], "=>", text_label[1])
