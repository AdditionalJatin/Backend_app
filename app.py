import base64
import cv2
import numpy as np
import time
from datetime import datetime
from fastapi import FastAPI, Request
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI()

class ImageInput(BaseModel):
    image: str  # base64 string

class StorySegmentGenerator:
    def __init__(self, model_path="yolov8n.pt", narration_interval=10):
        self.model = YOLO(model_path)
        self.narration_interval = narration_interval
        self.last_narration_time = time.time()
        self.frame_detections_buffer = []  # Buffer for detections over time
        self.story_segments = []

    def decode_base64_image(self, base64_string):
        try:
            img_data = base64.b64decode(base64_string)
            np_arr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print("Decoding error:", e)
            return None

    def get_object_position(self, x_center, y_center, width, height):
        horiz = "left" if x_center < width / 3 else "right" if x_center > 2 * width / 3 else "center"
        vert = "top" if y_center < height / 3 else "bottom" if y_center > 2 * height / 3 else "center"
        return f"{vert}-{horiz}"

    def process_frame(self, frame):
        results = self.model(frame)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls_id]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            pos = self.get_object_position(x_center, y_center, frame.shape[1], frame.shape[0])
            detections.append({
                "class": label,
                "confidence": round(conf, 2),
                "position": pos
            })

        return detections

    def create_story_segment(self, timestamp, detections):
        if not detections:
            story_text = "No objects detected in the last segment."
        else:
            summarized = {}
            for det in detections:
                key = (det['class'], det['position'])
                summarized[key] = summarized.get(key, 0) + 1

            object_mentions = [
                f"{cls} at {pos} ({count} times)"
                for (cls, pos), count in summarized.items()
            ]
            object_summary = ', '.join(object_mentions)
            story_text = f"In the last 10 seconds, I saw: {object_summary}."

        segment = {
            "timestamp": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            "story": story_text,
            "objects": detections
        }

        self.story_segments.append(segment)
        self.last_narration_time = timestamp
        return segment

    def handle_image_input(self, base64_image):
        frame = self.decode_base64_image(base64_image)
        if frame is None:
            return None

        detections = self.process_frame(frame)
        self.frame_detections_buffer.extend(detections)

        current_time = time.time()
        if current_time - self.last_narration_time >= self.narration_interval:
            segment = self.create_story_segment(current_time, self.frame_detections_buffer)
            self.frame_detections_buffer = []  # reset for next segment
            return segment

        return None

# Initialize story generator
story_gen = StorySegmentGenerator()


@app.get("/")
def read_root():
    return {"message": "Backend is up and running!"}


@app.post("/upload_image/")
async def upload_image(data: ImageInput):
    segment = story_gen.handle_image_input(data.image)
    if segment:
        return {"story_segment": segment}
    return {"message": "Image received. Waiting for next narration interval."}

@app.get("/segments/")
async def get_all_segments():
    return {"segments": story_gen.story_segments}