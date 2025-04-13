import json, urllib.request # json 처리, http 통신
import base64 # 이미지 인코딩
import io
from PIL import Image
import websocket # 실시간 통신
import uuid, random # 고유 식별자 생성
import datetime

# ComfyUI 서버
server_address = "58.76.169.75:8000" 

# 사용자 이미지 입력 여부에 따라 ControlNet 관련 노드 처리
def update_workflow_with_image(workflow, image_path):
    # 사용자가 입력한 이미지가 없을 때 (순수 프롬프트만으로 이미지 생성해야 할 때)
    if image_path.strip() == "": 
        print("이미지를 입력하지 않았습니다. ControlNet 관련 노드를 제거합니다.")

        # ControlNet 관련 노드 제거
        for node_id in ["13", "14", "20", "21"]:
            if node_id in workflow:
                del workflow[node_id]

        # positive 프롬프트 연결 수정: ["14", 0] → ["6", 0]
        if "3" in workflow and "inputs" in workflow["3"]:
            if "positive" in workflow["3"]["inputs"]:
                if workflow["3"]["inputs"]["positive"] == ["14", 0]:
                    workflow["3"]["inputs"]["positive"] = ["6", 0]

        # negative 프롬프트도 수정: ["14", 1] → ["7", 0]
        if "3" in workflow and "inputs" in workflow["3"]:
            if "negative" in workflow["3"]["inputs"]:
                if workflow["3"]["inputs"]["negative"] == ["14", 1]:
                    workflow["3"]["inputs"]["negative"] = ["7", 0]

        # PreviewImage (18), SendImageWebSocket (22)의 images 연결 재설정 또는 제거
        for node_id in ["18", "22"]:
            if node_id in workflow and "inputs" in workflow[node_id]:
                if "images" in workflow[node_id]["inputs"]:
                    source = workflow[node_id]["inputs"]["images"]
                    if source == ["20", 0]:  # CannyEdge 출력을 참조하고 있었던 경우
                        workflow[node_id]["inputs"]["images"] = ["8", 0]  # VAE Decode 결과로 변경

        return workflow

    else:
        # 이미지 파일을 base64로 인코딩하여 워크플로우에 삽입
        base64_image = encode_image_to_base64(image_path)
        workflow["21"]["inputs"]["image"] = base64_image
        return workflow

# workflow.json 파일을 읽어와 Python 딕셔너리로 반환
def load_workflow(filename="workflow.json"):
    with open(filename, 'r') as file:
        return json.load(file)

# 텍스트 프롬프트 내용을 워크플로우에 반영
def update_prompt_in_workflow(workflow, prompt_text):
    workflow["6"]["inputs"]["text"]=prompt_text
    return workflow

# 이미지 파일을 base64 문자열로 인코딩
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
# 시드값을 무작위로 생성하여 워크플로우에 적용
def update_seed_in_workflow(workflow):
    random_seed = random.randint(0, 2**64 - 1)
    workflow["3"]["inputs"]["seed"] = random_seed
    print(f"사용된 시드값: {random_seed}")
    return workflow

# ComfyUI 서버에 워크플로우를 전송하고 응답을 반환
def queue_workflow(workflow):
    data = json.dumps({"prompt": workflow}).encode('utf-8') # 서버 명세상 "prompt" 키는 고정
    req = urllib.request.Request( # http post 요청
        f"http://{server_address}/prompt", 
        data=data, 
        headers={'Content-Type': 'application/json'}
        )
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read()) # 응답 JSON을 딕셔너리로 반환
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        print(f"Response body: {e.read().decode('utf-8')}")
        raise

# WebSocket으로 ComfyUI 실행 완료 및 이미지 전송 여부를 실시간 모니터링
def get_image(prompt_id):
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws")
    
    print(f"Waiting for image data for prompt ID: {prompt_id}")
    
    while True:
        message = ws.recv()
        if isinstance(message, str):
            data = json.loads(message)
            print(f"Received message: {data}")
            if data['type'] == 'executing':
                if data['data']['node'] is None and data['data']['prompt_id'] == prompt_id:
                    print("Execution completed")
                    break
        elif isinstance(message, bytes):
            print("Received binary data (likely image)")
            image = Image.open(io.BytesIO(message[8:]))
            ws.close()
            return image
    
    ws.close()
    return None

# ---------- 메인 실행 흐름 ----------
# 워크플로우 로드
workflow = load_workflow()
print("Workflow loaded successfully.")

# 사용자 프롬프트 입력 + 요청 ID 출력
prompt_text=input("사용할 프롬프트를 영어로 입력하세요:")
unique_suffix = str(uuid.uuid4())[:8]
print(f"요청 ID : {unique_suffix}")
workflow=update_prompt_in_workflow(workflow, prompt_text)

# 이미지 경로 입력 및 반영
input_image_path = input("사용할 이미지 파일 경로를 입력하세요 (없으면 엔터): ")
workflow = update_workflow_with_image(workflow, input_image_path)
print("Workflow updated with input image.")

# 무작위 시드값 반영
workflow = update_seed_in_workflow(workflow)
print("Seed Updated")

# 워크플로우 큐잉 요청 및 prompt ID 획득
response = queue_workflow(workflow)
prompt_id = response['prompt_id']
print(f"Prompt queued with ID: {prompt_id}")

# WebSocket으로 이미지 수신
image = get_image(prompt_id)
if image:
    output_filename = "generated_image.png"
    image.save(output_filename)
    print(f"Image saved as {output_filename}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
else:
    print("Failed to retrieve image")

print(f"[{datetime.datetime.now()}] 이미지 생성 및 저장 완료. 스크립트 종료.")