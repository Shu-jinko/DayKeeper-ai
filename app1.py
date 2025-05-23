from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification
from gliner import GLiNER
import MeCab
import re

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 모델 로딩
MODEL_SAVE_PATH = 'fine-tuned-klue-bert-base'
tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
bert_model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
gliner_model = GLiNER.from_pretrained("urchade/gliner_multiv2.1")
tagger = MeCab.Tagger()

labels = ["unknown", "기쁨", "두려움", "분노", "슬픔"]
ner_labels = ["date", "person", "location", "organization", "event", "time", "family"]

app = FastAPI()

# 입력 데이터
class UserInput(BaseModel):
    text: str

# GPT 문단 정리 함수
def organize_paragraphs(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
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
                "content": text
            }
        ]
    )
    return response.choices[0].message.content

# 감정 분석
def classify_text(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1)
    predicted_class = tf.argmax(probs, axis=1).numpy()[0]
    return {
        "label": labels[predicted_class],
        "scores": {labels[i]: float(f"{score:.4f}") for i, score in enumerate(probs.numpy()[0])}
    }

# 명사 추출
def extract_nouns(text):
    result = []
    parsed = tagger.parse(text)
    for line in parsed.split('\n'):
        if line == 'EOS' or line == '':
            continue
        surface, features = line.split('\t')
        pos = features.split(',')[0]
        if pos in ('NNG', 'NNP'):
            result.append(surface)
    return result

# 엔티티 추출
def extract_entities(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    all_entities = []
    for sent in sentences:
        entities = gliner_model.predict_entities(sent, ner_labels, threshold=0.4)
        for entity in entities:
            noun_list = extract_nouns(entity["text"])
            if noun_list:
                cleaned = ' '.join(noun_list)
                all_entities.append({"text": cleaned, "label": entity["label"]})
    unique = { (e["text"], e["label"]) for e in all_entities }
    return [{"text": text, "label": label} for text, label in unique]

# API 엔드포인트
@app.post("/analyze/")
def analyze_text(user_input: UserInput):
    try:
        emotion = classify_text(user_input.text)
        entities = extract_entities(user_input.text)
        paragraphs = organize_paragraphs(user_input.text)
        return {
            "emotion": emotion,
            "entities": entities,
            "organized_text": paragraphs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
