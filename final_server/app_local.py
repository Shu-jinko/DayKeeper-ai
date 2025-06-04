from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from transformers import BlipProcessor, BlipForConditionalGeneration, BertTokenizerFast, TFBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from gliner import GLiNER
from PIL import Image
import tensorflow as tf
import torch
import os
import shutil
import re
from typing import List
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()
sbert = SentenceTransformer('distiluse-base-multilingual-cased-v1')

tokenizer = BertTokenizerFast.from_pretrained('fine-tuned-klue-bert-basev2')
bert_model = TFBertForSequenceClassification.from_pretrained('fine-tuned-klue-bert-basev2')

gliner_model = GLiNER.from_pretrained("urchade/gliner_multiv2.1")

app = FastAPI()

emotion_labels = ["기쁨", "슬픔", "분노", "두려움", "놀람", "불쾌함", "죄책감", "사랑", "수치심", "기대감"]
ner_labels = ["person", "location", "activity", "event", "family", "object", "food", "weather"]

def classify_text(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1)
    predicted_class = tf.argmax(probs, axis=1).numpy()[0]
    return {
        "label": emotion_labels[predicted_class],
        "scores": {emotion_labels[i]: float(f"{score:.4f}") for i, score in enumerate(probs.numpy()[0])}
    }

def filter_ner(entities: list[dict]) -> list[dict]:
    prompt = f"""
    다음은 문장에서 추출된 키워드와 그에 할당된 개체명(NER) 라벨입니다.

    각 항목은 다음 형식으로 구성되어 있습니다:
    [text] - label

    라벨 목록: ["person", "location", "activity", "event", "family", "object", "food", "weather"]

    이 중에서 다음과 같은 경우는 제거 대상입니다:
    1. 단어와 라벨이 의미적으로 맞지 않음 (예: "이두근" - person)
    2. 일반 명사이지만 특정 의미적 범주에 속하지 않음 (예: "생각" - activity)
    3. 너무 일반적이거나 모호해서 라벨로 분류하기 어려움 (예: "장소" - location)

    다음은 유효한 예시입니다:
    - "이순신" - person 
    - "등산" - activity 
    - "김치찌개" - food

    출력은 JSON 배열 형식이며, 각 항목은 아래처럼 구성되어야 합니다:
    [
      {{ "text": "이순신", "label": "person" }},
      {{ "text": "등산", "label": "activity" }}
    ]

    각 항목의 텍스트가 오타거나 잘못 띄어쓴 경우(예: "돈 까스")에는 자연스러운 표기(예: "돈까스")로 수정해 주세요.
    '운동장에' 처럼 조사가 붙은 경우도 제거해 주세요.
    단, 라벨 수정은 하지 마세요. 의미가 맞지 않는 항목은 아예 제거하세요.

    입력:
    {chr(10).join([f"{e['text']} - {e['label']}" for e in entities])}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "너는 NER 라벨 검수 시스템이야. 의미가 맞지 않으면 해당 항목을 제거하고 JSON 형식을 정확히 지켜서 응답해."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        import json
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"[GPT 오류] {e}")
        return []

def extract_entities(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    all_entities = []
    for sent in sentences:
        entities = gliner_model.predict_entities(sent, ner_labels, threshold=0.3)
        for entity in entities:
            all_entities.append({"text": entity["text"], "label": entity["label"]})
    unique = { (e["text"], e["label"]) for e in all_entities }
    return [{"text": text, "label": label} for text, label in unique]

def generate_captions(image_infos):
    captions = []
    for path, filename in image_infos:
        image = Image.open(path).convert('RGB')
        inputs = blip_processor(image, return_tensors="pt")
        with torch.no_grad():
            out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        captions.append((filename, caption))
    return captions

def match_paragraphs_to_images(paragraphs, image_captions):
    paragraph_texts = [p['content'] for p in paragraphs]
    para_embeddings = sbert.encode(paragraph_texts, convert_to_tensor=True)
    caption_texts = [c[1] for c in image_captions]
    caption_embeddings = sbert.encode(caption_texts, convert_to_tensor=True)

    similarities = util.cos_sim(para_embeddings, caption_embeddings)
    used_indices = set()
    results = []

    for i, sims in enumerate(similarities):
        best_score = -float('inf')
        best_idx = -1
        for j, score in enumerate(sims):
            if j not in used_indices and score.item() > best_score:
                best_score = score.item()
                best_idx = j

        if best_idx == -1:
            best_idx = sims.argmax().item()

        used_indices.add(best_idx)
        best_img_name = image_captions[best_idx][0]
        results.append({
            "paragraph": paragraphs[i]['content'],
            "matched_image": best_img_name,
            "image_caption": image_captions[best_idx][1]
        })
    return results

def save_uploaded_images(images):
    saved_paths = []
    os.makedirs("temp_images", exist_ok=True)
    for img in images:
        filename = os.path.basename(img.filename).strip()
        if not filename or filename in (".", ".."):
            continue
        path = os.path.join("temp_images", filename)
        if os.path.isdir(path):
            continue
        with open(path, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
        saved_paths.append((path, filename))
    return saved_paths

class UserInput(BaseModel):
    text: str
class Each_Paragraph(BaseModel):
    subject: str
    content: str
class Final_Result(BaseModel):
    final: list[Each_Paragraph]
    summary: str

@app.post("/analyze")
async def analyze(text: str = Form(...), images: List[UploadFile] = File(default=[])):
    if not text.strip():
        raise HTTPException(status_code=400, detail="입력 텍스트가 비어 있습니다.")
    try:
        gpt_response = client.responses.parse(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": """
                    주어진 글은 하루 동안 있었던 일을 자유 형식으로 적은 글입니다.
                    이 글을 다음 조건에 따라 주제별로 정리해 주세요:
                    1. 원문에 없는 내용을 새로 덧붙이지 마세요.
                    2. 필요하다면 문장 순서를 조금 바꾸거나 간단한 문장 다듬기는 허용합니다.
                    3. 주제별로 내용을 묶고, 문단을 나눠 정리해주세요. 각 문단의 주제는 드러나도록 구성해주세요.  
                    4. 전체 톤과 말투는 원문의 느낌을 유지해주세요. 단, 문장은 가능한 완전한 문장으로 작성해주세요.
                    5. 오타가 있다면 수정해주세요.
                    6. 마지막으로, 전체 내용을 대표할 수 있는 한 문장 요약도 함께 작성해주세요.
                """},
                {"role": "user", "content": text}
            ],
            text_format=Final_Result
        )

        parsed = gpt_response.output_parsed.model_dump()
        merged_content = "\n".join([para["content"] for para in parsed["final"]])

        # 감정 분석 및 엔티티 추출
        emotion_result = classify_text(merged_content)
        entities = extract_entities(merged_content)
        saved_image_paths = save_uploaded_images(images)
        image_captions = generate_captions(saved_image_paths)
        entities = filter_ner(entities)

        paragraphs_with_images = []
        if image_captions:
            matched = match_paragraphs_to_images(parsed["final"], image_captions)
            for para, match in zip(parsed["final"], matched):
                paragraphs_with_images.append({
                    "subject": para["subject"],
                    "content": para["content"],
                    "matched_image": match["matched_image"],
                    "image_caption": match["image_caption"]
                })
        else:
            for para in parsed["final"]:
                paragraphs_with_images.append({
                    "subject": para["subject"],
                    "content": para["content"],
                    "matched_image": None,
                    "image_caption": None
                })

        return {
            "paragraphs": paragraphs_with_images,
            "summary": parsed["summary"],
            "emotion": emotion_result,
            "keywords": entities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))