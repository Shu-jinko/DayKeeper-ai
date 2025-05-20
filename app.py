# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification
from dotenv import load_dotenv
import os
from openai import OpenAI

# .env에서 API 키 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# BERT 모델 로드
MODEL_SAVE_PATH = 'fine-tuned-klue-bert-base'
tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
labels = ["unknown", "기쁨", "두려움", "분노", "슬픔"]

# FastAPI 앱 초기화
app = FastAPI()

# 요청 모델
class UserInput(BaseModel):
    text: str

# 출력 모델
class Each_Paragraph(BaseModel):
    subject: str
    content: str

class Final_Result(BaseModel):
    final: list[Each_Paragraph]

# 감정 분석 함수
def classify_text(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1)
    predicted_class = tf.argmax(probs, axis=1).numpy()[0]
    return {
        "label": labels[predicted_class],
        "scores": {labels[i]: float(f"{score:.4f}") for i, score in enumerate(probs.numpy()[0])}
    }

# 주제 정리 및 감정 분석 API
@app.post("/analyze")
def analyze(input_data: UserInput):
    try:
        # GPT를 통한 문단 정리
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
                {"role": "user", "content": input_data.text}
            ],
            text_format=Final_Result
        )

        parsed = response.output_parsed.model_dump()
        emotion_result = classify_text(input_data.text)

        return {
            "paragraphs": parsed["final"],
            "emotion": emotion_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
