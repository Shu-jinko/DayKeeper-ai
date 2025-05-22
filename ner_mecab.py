from gliner import GLiNER
import MeCab
import re

# 1. 모델 불러오기
model = GLiNER.from_pretrained("urchade/gliner_multiv2.1")

# 2. MeCab-ko-msvc 태거 설정
tagger = MeCab.Tagger()

# 3. 명사 추출 함수
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

# 4. 분석 대상 텍스트
text = """
가족 모두 건강검진을 받기로 한 날이다. 민수랑 나도 함께 받기로 했다. 잠원동쪽 병원에서 종합검진을 거의 3시간동안 받았다. 우리 나이대에 잘 하지 않지만 ct와 위, 대장 내시경도 받았다. 대장내시경 때문에 전날부터 약을 먹느라 정말 힘들었다. 새벽4시까지도 약을 먹느라 잠을 거의 못잤고 1시간쯤 잤나..? 그 상태로 검진을 받으러 와서 검진을 받는 내내 너무 힘들었다. 수면이어서 순식간에 끝났지만 진료가 끝나자마자 급하게 깨워서 비몽사몽상태로 휘청이면서 병실을 나왔다. 그렇게 수면 내시경까지 끝내고 병원 가까이 있는 음식점을 가서 식사를 했다. 너무 피곤하기도 해서 어떤 정신으로 밥을 먹었는지도 기억이 가물하다. 집에 도착하자마자 아직 저녁 6시도 되지않았는데 정신없이 잠이 들었다가 밤 11시쯤 깨어났다. 내일까지 보내야할 과제가 있어서 다 끝내고 씻고 다시 잤다.
"""

# 5. 개체명 라벨 정의
labels = ["date", "person", "location", "organization", "event", "time", "family"]

# 6. 문장 단위 분할
sentences = re.split(r'(?<=[.!?])\s+', text.strip())

# 7. 개체 예측 및 명사 정제
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

# 8. 중복 제거
unique_entities = { (e["text"], e["label"]) for e in all_entities }

# 9. 출력
for text_label in unique_entities:
    print(text_label[0], "=>", text_label[1])
