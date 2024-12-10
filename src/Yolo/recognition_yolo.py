from ultralytics import YOLO
import os

dataset_yaml = "datasets.yaml"
pretrained_model = "yolov8s.pt"

def train_yolo():
    model = YOLO(pretrained_model)
    model.train(
        data=dataset_yaml,         # Dataset YAML
        epochs=20,                 # Numero di epoche per il fine-tuning
        imgsz=640,                 # Dimensione delle immagini
        project="./models",        # Directory di output
        name="custom_train",       # Nome del task di training
        verbose=True               # Stampa informazioni dettagliate
    )

def predict_yolo(image_path):
    model = YOLO("./models/custom_train/weights/best.pt")
    results = model.predict(
        source=image_path,        # Percorso dell'immagine o video
        imgsz=640,                # Dimensione immagine per inferenza
        conf=0.5,                 # Soglia di confidenza
        save=True,                # Salva i risultati
        project="runs/predict",   # Directory di output
        name="custom_predict"     # Nome del task di inferenza
    )
    return results

if __name__ == "__main__":

    #if not os.path.exists("runs/detect/custom_train/weights/best.pt"):
    #    print("Inizio del training...")
    #    train_yolo()


    print("Eseguo il detectamento...")
    image_path = "../../template_images/detection_tryal.png"
    results = predict_yolo(image_path)

    # Mostra i risultati
    print(results)
