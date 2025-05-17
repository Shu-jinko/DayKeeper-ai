# -*- coding: utf-8 -*-
from openai import OpenAI
from pydantic import BaseModel

api_key = ""
client = OpenAI(api_key=api_key)

# 자유 형식 글 -> 주제별 정리
class Each_Paragraph(BaseModel):
    subject: str
    content: str

class Final_Result(BaseModel):
    final:list[Each_Paragraph]

user_input = input("오늘 하루를 입력해 주세요: ")

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

for i in result.final:
    print(i.subject)
    print(i.content)