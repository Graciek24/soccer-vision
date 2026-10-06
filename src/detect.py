from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="input/match.mp4",
    save=True,
    project="output",
    name="detections"
)

print("Detection finished!")