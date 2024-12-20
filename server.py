from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
import tritonclient.grpc as grpcclient
from typing import Dict
import threading
import cv2
import numpy as np
import tempfile
import uvicorn

app = FastAPI()


# Triton client setup
TRITON_SERVER_URL = "192.168.10.8:8001"
MODEL_NAME = "yolo11m"
client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Dictionary to keep track of live camera threads
camera_threads: Dict[str, threading.Thread] = {}

# Helper function for running inference
def run_inference(image: np.ndarray, object_class: str):
    inputs = [
        grpcclient.InferInput("input", image.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(image.astype(np.float32))

    outputs = [
        grpcclient.InferRequestedOutput("output")
    ]

    response = client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs
    )

    results = response.as_numpy("output")
    return [res for res in results if res["class"] == object_class]

# Background task for live video detection
def detect_from_camera(camera_ip: str, object_class: str):
    cap = cv2.VideoCapture(camera_ip)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Cannot open camera: {camera_ip}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_inference(frame, object_class)
        print(f"Detections from {camera_ip}: {detections}")

    cap.release()

@app.get("/")
async def read_root():
    # Get model list
    model_list = client.get_model_repository_index(as_json=True)

    return {f"message": "Welcome to the object detection API. The following models are available: " + str(model_list)}

@app.post("/detect_objects_in_video")
async def detect_objects_in_video(
    object_class: str = "person",
    video_file: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(await video_file.read())
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot process the uploaded video file.")

    all_detections = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_inference(frame, object_class)
        all_detections.extend(detections)

    cap.release()
    return {"detections": all_detections}

@app.post("/detect_objects_in_live_video")
async def detect_objects_in_live_video(
    background_tasks: BackgroundTasks,
    camera_ip: str = Form(...),
    object_class: str = Form(...)
):
    if camera_ip in camera_threads:
        raise HTTPException(status_code=400, detail="Detection for this camera is already running.")

    thread = threading.Thread(target=detect_from_camera, args=(camera_ip, object_class), daemon=True)
    camera_threads[camera_ip] = thread
    thread.start()

    def cleanup():
        if camera_ip in camera_threads:
            camera_threads.pop(camera_ip)

    background_tasks.add_task(cleanup)
    return {"message": f"Detection started for camera {camera_ip}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
