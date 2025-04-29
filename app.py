import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import openai

# OpenAI API 키 설정
client = openai.OpenAI(api_key="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")

# BLIP 모델 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_keywords_from_image(raw_image):
    inputs = processor(raw_image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    # 간단 키워드 추출
    keywords = [word.strip(",.") for word in caption.lower().split()]
    meaningful_keywords = [word for word in keywords if len(word) > 2]

    return meaningful_keywords, caption

def generate_clean_short_questions(event, keywords):
    keywords_str = ', '.join(keywords)
    prompt = (
        f"일정: {event}, 사진 키워드: {keywords_str}.\n\n"
        "이 일정을 실제로 어떻게 진행했는지에 대한 추가 정보가 필요해.\n"
        "사용자가 경험한 '사실'을 묻는 객관식 질문 3개를 생성해줘.\n"
        "질문은 너무 구체적인 상황을 가정하지 말고, 누구나 쉽게 답할 수 있도록 보편적인 내용을 묻는 질문이어야 해.\n"
        "이미 사진 키워드로 확인된 사실은 반복해서 묻지 마.\n"
        "선택지는 긴 설명문이 아니라 5~7단어 이내의 짧은 문장이나 명사구 형태로 작성해.\n"
        "질문과 선택지는 모두 한국어로 작성해."
    )

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "너는 사용자가 일정을 바탕으로 사실 기반 추가 정보를 수집하는 객관식 질문을 만드는 비서야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )

    questions_text = response.choices[0].message.content
    return questions_text

# Streamlit UI
st.title("사진 기반 키워드 추출 및 질문 생성기")

uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "jpeg", "png"])
event = st.text_input("이 사진과 관련된 일정을 입력하세요 (예: 영화 보기)")

if uploaded_file and event:
    
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="업로드된 사진", use_column_width=True)

    with st.spinner('사진에서 키워드를 추출하는 중...'):
        keywords, caption = generate_keywords_from_image(image)

    st.success(f"사진 설명: {caption}")
    st.info(f"추출된 키워드: {', '.join(keywords)}")

    with st.spinner('객관식 질문을 생성하는 중...'):
        questions = generate_clean_short_questions(event, keywords)

    st.subheader("생성된 객관식 질문")
    st.markdown(questions)

else:
    st.warning("사진을 업로드하고 일정을 입력하세요.")

