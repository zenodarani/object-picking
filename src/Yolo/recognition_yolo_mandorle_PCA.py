from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data='dataset.yaml',  # File di configurazione del datasets
    epochs=50,            # Numero di epoche
    imgsz=640,            # Dimensione delle immagini
    batch=8,              # Dimensione del batch
    project='./models',  # Directory per i risultati
    name='mandorle_model' # Nome del modello
)

model = YOLO('./models/mandorle_model/weights/best.pt')

# Esegui l'inferenza
results = model.predict(
    source='../../template_images/detection_tryal.png',  # Percorso dell'immagine
    conf=0.5,                   # Soglia di confidenza
    save=True                   # Salva i risultati
)

print(results)