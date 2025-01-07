from ultralytics import YOLO

model = YOLO('yolov8l.pt')

model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    project='./models',
    name='mandorle_model'
)

model = YOLO('./models/mandorle_model/weights/best.pt')

# Esegui l'inferenza
results = model.predict(
    source='../../template_images/detection_tryal.png',
    conf=0.5,
    save=True
)

print(results)