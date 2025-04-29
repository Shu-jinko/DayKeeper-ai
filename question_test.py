import openai
# 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

# OpenAI API 키 설정
client = openai.OpenAI(api_key="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")


def generate_clean_short_questions(event, keywords):
    keywords_str = ', '.join(keywords)
    prompt = (
        f"일정: {event}, 사진 키워드: {keywords_str}.\n\n"
        "이 일정을 실제로 어떻게 진행했는지에 대한 추가 정보가 필요해.\n"
        "사용자가 경험한 '사실'을 묻는 객관식 질문 3개를 생성해줘.\n"
        "질문은 너무 구체적인 상황을 가정하지 말고, 누구나 쉽게 답할 수 있도록 보편적인 내용을 묻는 질문이어야 해.\n"
        "사진 키워드(예: {keywords_str})와 관련된 상황을 다룰 수는 있지만, 이미 사진 키워드를 통해 확인된 정보(예: 팝콘, 콜라)를 반복적으로 묻는 질문은 생성하지 마.\n"
        "각 질문에는 4개의 선택지를 포함하고, 선택지는 긴 설명문이 아니라 5~7단어 이내의 짧고 간결한 문장이나 명사구 형태로 작성해.\n"
        "질문과 선택지는 모두 한국어로 작성해."
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "너는 사용자가 일정을 바탕으로 사실 기반 추가 정보를 수집하는 객관식 질문을 만드는 비서야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )

    questions_text = response.choices[0].message.content
    return questions_text

# 사용 예시
if __name__ == "__main__":
    event = "영화 보기" # 일정을 가정.
    keywords = ["팝콘", "콜라"] # 사진에서 키워드 추출 가정
    questions = generate_clean_short_questions(event, keywords)
    print(questions)
