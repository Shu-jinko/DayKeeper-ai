from gliner import GLiNER
# 초기 실행시
#model = GLiNER.from_pretrained("urchade/gliner_multiv2.1", force_download=True)

model = GLiNER.from_pretrained("urchade/gliner_multiv2.1")

text = """
가족 모두 건강검진을 받기로 한 날이다. 잠원동쪽 병원에서 종합검진을 거의 3시간동안 받았다. 우리 나이대에 잘 하지 않지만 ct와 위, 대장 내시경도 받았다. 대장내시경 때문에 전날부터 약을 먹느라 정말 힘들었다. 새벽4시까지도 약을 먹느라 잠을 거의 못잤고 1시간쯤 잤나..? 그 상태로 검진을 받으러 와서 검진을 받는 내내 너무 힘들었다. 수면이어서 순식간에 끝났지만 진료가 끝나자마자  급하게 깨워서 비몽사몽상태로 휘청이면서 병실을 나왔다. 그렇게 수면 내시경까지 끝내고 병원 가까이 있는 음식점을 가서 식사를 했다. 너무 피곤하기도 해서 어떤 정신으로 밥을 먹었는지도 기억이 가물하다. 집에 도착하자마자 아직 저녁 6시도 되지않았는데 정신없이 잠이 들었다가 밤 11시쯤 깨어났다. 내일까지 보내야할 과제가 있어서 다 끝내고 씻고 다시 잤다.
"""

labels = ["date", "person", "location", "organization", "event", "time"]

entities = model.predict_entities(text, labels, threshold=0.3)

for entity in entities:
    print(entity["text"], "=>", entity["label"])