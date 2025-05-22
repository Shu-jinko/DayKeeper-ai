from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification
from dotenv import load_dotenv
import os
from gliner import GLiNER
import MeCab
import re
from typing import List
from openai import OpenAI

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# BERT 모델 로드
MODEL_SAVE_PATH = 'fine-tuned-klue-bert-base'
tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
bert_model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
labels_emotion = ["unknown", "기쁨", "두려움", "분노", "슬픔"]

# GLiNER 모델 로드
gliner_model = GLiNER.from_pretrained("urchade/gliner_multiv2.1")

# MeCab 설정
tagger = MeCab.Tagger()

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

def classify_text(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1)
    predicted_class = tf.argmax(probs, axis=1).numpy()[0]
    return {
        "label": labels_emotion[predicted_class],
        "scores": {labels_emotion[i]: float(f"{probs[0][i]:.4f}") for i in range(len(labels_emotion))}
    }

# FastAPI 앱 초기화
app = FastAPI()

# 요청/응답 모델 정의
class UserInput(BaseModel):
    text: str

class EntityResult(BaseModel):
    text: str
    label: str

class EachParagraph(BaseModel):
    subject: str
    content: str

class EmotionResult(BaseModel):
    label: str
    scores: dict

@app.post("/emotion", response_model=EmotionResult)
def analyze_emotion(input: UserInput):
    try:
        return classify_text(input.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/entities", response_model=List[EntityResult])
def analyze_entities(input: UserInput):
    try:
        text = input.text
        gliner_labels = ["date", "person", "location", "organization", "event", "time", "family"]
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        all_entities = []
        for sent in sentences:
            entities = gliner_model.predict_entities(sent, gliner_labels, threshold=0.5)
            for entity in entities:
                nouns = extract_nouns(entity["text"])
                if nouns:
                    cleaned = ' '.join(nouns)
                    all_entities.append({"text": cleaned, "label": entity["label"]})

        unique_entities = list({(e["text"], e["label"]): e for e in all_entities}.values())
        return [EntityResult(**e) for e in unique_entities]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/paragraphs", response_model=List[EachParagraph])
def summarize_paragraphs(input: UserInput):
    try:
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
                {"role": "user", "content": input.text}
            ],
            text_format=List[EachParagraph]
        )
        return response.output_parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
