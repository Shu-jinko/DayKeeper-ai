import json
import urllib.request
import base64
import io
from PIL import Image
import websocket
import uuid, random

server_address = "58.76.169.75:8000" # ComfyUI Server

def load_workflow(filename="workflow.json"):
    with open(filename, 'r') as file:
        return json.load(file)

def update_prompt_in_workflow(workflow, prompt_text):
    workflow["6"]["inputs"]["text"]=prompt_text
    return workflow

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def update_seed_in_workflow(workflow):
    # 무작위 시드값 생성 (64비트 범위 내에서 무작위 값 생성)
    random_seed = random.randint(0, 2**64 - 1)
    workflow["3"]["inputs"]["seed"] = random_seed  # '3' 노드는 workflow.json에서 시드값을 설정하는 노드임
    print(f"사용된 시드값: {random_seed}")
    return workflow

def update_workflow_with_image(workflow, image_path):
    base64_image = encode_image_to_base64(image_path)
    # base64 노드 번호 수정 필요
    workflow["21"]["inputs"]["image"] = base64_image
    return workflow

def queue_prompt(prompt):
    data = json.dumps({"prompt": prompt}).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        print(f"Response body: {e.read().decode('utf-8')}")
        raise

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
            image = Image.open(io.BytesIO(message[8:]))  # Skip first 8 bytes (message type)
            ws.close()
            return image
    
    ws.close()
    return None

# Load workflow from JSON file
workflow = load_workflow()
print("Workflow loaded successfully.")

prompt_text=input("사용할 프롬프트를 영어로 입력하세요:")
unique_suffix = str(uuid.uuid4())[:8]  # 고유한 8자리 식별자 추가
prompt_text += f" [{unique_suffix}]"  # 프롬프트에 추가
workflow=update_prompt_in_workflow(workflow, prompt_text)

# Specify the path to the image you want to upload
#input_image_path = "sample-image.jpg"  # Make sure this file exists
input_image_path = input("사용할 이미지 파일 경로를 입력하세요: ")
# Update the workflow with the input image
workflow = update_workflow_with_image(workflow, input_image_path)
print("Workflow updated with input image.")
workflow = update_seed_in_workflow(workflow)
print("Seed Updated")

# Generate image
response = queue_prompt(workflow)
prompt_id = response['prompt_id']
print(f"Prompt queued with ID: {prompt_id}")

image = get_image(prompt_id)
if image:
    output_filename = "generated_image.png"
    image.save(output_filename)
    print(f"Image saved as {output_filename}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
else:
    print("Failed to retrieve image")

print("Script execution completed.")
