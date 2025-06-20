from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field
from transformers import BlipProcessor, BlipForConditionalGeneration, BertTokenizerFast, TFBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from gliner import GLiNER
from PIL import Image
import torch
import os
import shutil
import re
from typing import List, Dict
from dotenv import load_dotenv
import time
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Dict

MATCH_THRESHOLD = 0.078

load_dotenv()
api_key1 = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key1)

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").eval()
sbert = SentenceTransformer('distiluse-base-multilingual-cased-v1')
gliner_model = GLiNER.from_pretrained("urchade/gliner_multiv2.1")

tokenizer = BertTokenizerFast.from_pretrained('fine-tuned-klue-bert-basev2')
bert_model = TFBertForSequenceClassification.from_pretrained('fine-tuned-klue-bert-basev2')

app = FastAPI()

emotion_labels = ["기대감", "기쁨", "놀람", "두려움", "분노", "불쾌함", "사랑", "수치심", "슬픔"]
ner_labels = ["person", "location", "activity", "family", "object", "food"]

# 감정 분류
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

class UserInput(BaseModel):
    text: str

class Each_Paragraph(BaseModel):
    subject: str
    content: str
    label: str

class Final_Result(BaseModel):
    final: List[Each_Paragraph]
    summary: str

def filter_ner(entities: list[dict]) -> list[dict]:
    prompt = f"""
    다음은 문장에서 추출된 키워드와 그에 할당된 개체명(NER) 라벨입니다.

    각 항목은 다음 형식으로 구성되어 있습니다:
    [text] - label

    라벨 목록: ["person", "location", "activity", "family", "object", "food"]

    이 중에서 다음과 같은 경우는 제거 대상입니다:
    1. 단어와 라벨이 의미적으로 맞지 않음 (예: "이두근" - person)
    2. 너무 일반적이거나 모호해서 라벨로 분류하기 어려움 (예: "장소" - location)
    3. 중복된 항목 (예: text-label 쌍이 동일한 경우)

    다음과 같은 경우는 수정 대상입니다:
    1. 오타나 잘못된 띄어쓰기가 있는 경우, 자연스러운 표기로 수정 (예: "돈 까스" → "돈까스")
    2. 조사가 붙은 경우, 조사 제거 (예: "이순신의" → "이순신")

    출력은 JSON 배열 형식이며, 각 항목은 아래처럼 구성되어야 합니다:
    [
      {{ "text": "이순신", "label": "person" }},
      {{ "text": "등산", "label": "activity" }}
    ]
    입력:
    {chr(10).join([f"{e['text']} - {e['label']}" for e in entities])}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "너는 NER 라벨 검수 시스템이야. 프롬프트 조건에 따라 JSON 형식을 정확히 지켜서 응답해."
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
        #print(f"[캡션 생성] {filename} → {caption}")  # 추가된 로그
        captions.append((filename, caption))
    return captions

def match_paragraphs_to_images(paragraphs, image_captions, threshold=MATCH_THRESHOLD):
    paragraph_texts = [p['content'] for p in paragraphs]
    para_embeddings = sbert.encode(paragraph_texts, convert_to_tensor=True)
    caption_texts = [c[1] for c in image_captions]
    caption_embeddings = sbert.encode(caption_texts, convert_to_tensor=True)

    similarities = util.cos_sim(para_embeddings, caption_embeddings)
    sim_matrix = similarities.cpu().numpy()

    cost_matrix = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = []
    used_image_indices = set()
    for i in range(len(paragraphs)):
        if i in row_ind:
            j = col_ind[np.where(row_ind == i)[0][0]]
            score = sim_matrix[i][j]
            if score < threshold:
                print(f"문단 {i+1} → 매칭 없음 (유사도 {score:.4f} < threshold {threshold})")
                matched.append({"paragraph": paragraphs[i]['content'], "matched_image": None, "image_caption": None})
            else:
                print(f"문단 {i+1} → 이미지 {j+1} ({image_captions[j][0]}) | 유사도 {score:.4f}")
                matched.append({"paragraph": paragraphs[i]['content'], "matched_image": image_captions[j][0], "image_caption": image_captions[j][1]})
                used_image_indices.add(j)
        else:
            print(f"문단 {i+1} → 매칭 없음 (index 미포함)")
            matched.append({"paragraph": paragraphs[i]['content'], "matched_image": None, "image_caption": None})

    unmatched_images = [image_captions[i][0] for i in range(len(image_captions)) if i not in used_image_indices]
    return matched, unmatched_images

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

@app.post("/analyze")
async def analyze(text: str = Form(...), images: List[UploadFile] = File(default=[])):
    print(f"[업로드된 이미지 수] {len(images)}")
    for img in images:
        print(f"[이미지 이름] {img.filename}")
    if not text.strip():
        raise HTTPException(status_code=400, detail="입력 텍스트가 비어 있습니다.")
    try:
        t0 = time.time()
        # Step 1: GPT로 문단 구조 + 요약 생성
        gpt_response = client.responses.parse(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": """
                    주어진 글은 하루 동안 있었던 일을 자유 형식으로 적은 글입니다.
                    이 글을 다음 조건에 따라 정리해 주세요:
                    1. 원문에 없는 내용을 새로 덧붙이지 마세요.
                    2. 필요하다면 문장 순서를 조금 바꾸거나 간단한 문장 다듬기는 허용합니다.
                    3. 일기의 내용을 주요 사건의 전개, 감정의 변화, 또는 경험의 연속적인 흐름에 따라 유기적으로 묶어 문단을 나눠 구성해 주세요. 각 문단은 하나의 포괄적인 맥락을 지닌 주제로 나눠야 합니다. 문단 내에서 여러 세부적인 내용이나 짧은 시간의 활동이 자연스럽게 이어지도록 해주세요. 
                    4. 전체 톤과 말투는 원문의 느낌을 유지해주세요. 단, 문장은 가능한 완전한 문장으로 작성해주세요.
                    5. 오타가 있다면 수정해주세요.
                    6. 마지막으로, 전체 내용을 대표할 수 있는 한 문장 요약도 함께 작성해주세요.
                """},
                {"role": "user", "content": text}
            ],
            text_format=Final_Result
        )

        parsed = gpt_response.output_parsed.model_dump()
        print(f"[GPT 문단 분석] {time.time() - t0:.2f}초")

        paragraph_texts = [p["content"] for p in parsed["final"]]
        translated_paragraphs = []
        for para in paragraph_texts:
            translation = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "Translate the following Korean diary paragraph into natural English. Preserve the tone and meaning."},
                    {"role": "user", "content": para}
                ],
                temperature=0
            )
            translated_text = translation.choices[0].message.content.strip()
            translated_paragraphs.append(translated_text)

        
        # Step 4: 전체 감정 요약 계산
        emotion_score_accumulator = [0.0] * len(emotion_labels)
        for result in emotion_data:
            for i, label in enumerate(emotion_labels):
                emotion_score_accumulator[i] += result["scores"].get(label, 0.0)

        paragraph_count = len(emotion_data)
        avg_scores = [round(score / paragraph_count, 4) for score in emotion_score_accumulator]
        overall_label_idx = avg_scores.index(max(avg_scores))
        emotion_result = {
            "label": emotion_labels[overall_label_idx],
            "scores": {emotion_labels[i]: avg_scores[i] for i in range(len(emotion_labels))}
        }

        # Step 5: NER 및 이미지 처리
        t2 = time.time()
        #merged_content = "\n".join(paragraph_texts)
        merged_content = "\n".join(
            f"{p['subject']}\n{p['content']}" for p in parsed["final"]
        )

        entities = extract_entities(merged_content)
        print(f"[개체명 추출] {time.time() - t2:.2f}초")

        saved_image_paths = save_uploaded_images(images)
        image_captions = generate_captions(saved_image_paths)

        t3 = time.time()
        print(f"[NER 전처리 결과 수] {len(entities)}")
        print(entities) # 디버깅용 출력
        entities = filter_ner(entities)
        print(f"[NER 필터링 후 결과 수] {len(entities)}")
        print(f"[NER 필터링] {time.time() - t3:.2f}초")

        # Step 6: 문단-이미지 매칭
        t4 = time.time()
        paragraphs_with_images = []
        unmatched_images = []

        if image_captions:
            matched, unmatched_images = match_paragraphs_to_images(
                [{"content": p} for p in translated_paragraphs],
                image_captions
            )

            for para, match in zip(parsed["final"], matched):
                paragraphs_with_images.append({
                    "subject": para["subject"],
                    "content": para["content"],
                    "matched_image": match["matched_image"],
                    "image_caption": match["image_caption"],
                    "label": para["emotion"]["label"]
                })
        else:
            for para in parsed["final"]:
                paragraphs_with_images.append({
                    "subject": para["subject"],
                    "content": para["content"],
                    "matched_image": None,
                    "image_caption": None,
                    "label": para["emotion"]["label"]
                })
        print(f"[문단-이미지 매칭] {time.time() - t4:.2f}초")
        
        print(f"entities: {entities}")
        return {
            "paragraphs": paragraphs_with_images,
            "summary": parsed["summary"],
            "emotion": emotion_result,
            "keywords": entities,
            "unmatched_images": unmatched_images
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))