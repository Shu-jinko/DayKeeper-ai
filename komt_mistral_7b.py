from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer, GenerationConfig

model_name='davidkim205/komt-mistral-7b-v1'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

def gen(x):
    generation_config = GenerationConfig(
        temperature=0.8,
        top_p=0.8,
        top_k=100,
        max_new_tokens=1024,
        early_stopping=True,
        do_sample=True,
    )
    q = f"[INST]{x} [/INST]"
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    result_str = tokenizer.decode(gened[0])

    start_tag = f"\n\n### Response: "
    start_index = result_str.find(start_tag)

    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()
    return result_str

prompt="""
다음 일기에서 '누가', '어디서', '무엇을' 정보를 간결히 추출해줘.
일기:
가족 모두 건강검진을 받기로 한 날이다. 잠원동쪽 병원에서 종합검진을 거의 3시간동안 받았다. 우리 나이대에 잘 하지 않지만 ct와 위, 대장 내시경도 받았다.

대장내시경 때문에 전날부터 약을 먹느라 정말 힘들었다. 새벽4시까지도 약을 먹느라 잠을 거의 못잤고 1시간쯤 잤나..? 그 상태로 검진을 받으러 와서 검진을 받는 내내 너무 힘들었다. 수면이어서 순식간에 끝났지만 진료가 끝나자마자  급하게 깨워서 비몽사몽상태로 휘청이면서 병실을 나왔다.

그렇게 수면 내시경까지 끝내고 병원 가까이 있는 음식점을 가서 식사를 했다. 너무 피곤하기도 해서 어떤 정신으로 밥을 먹었는지도 기억이 가물하다.

집에 도착하자마자 아직 저녁 6시도 되지않았는데 정신없이 잠이 들었다가 밤 11시쯤 깨어났다.

내일까지 보내야할 과제가 있어서 다 끝내고 씻고 다시 잤다."""
print(gen(prompt))
