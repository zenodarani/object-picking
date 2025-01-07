from ultralytics import YOLO
import os

dataset_yaml = "datasets.yaml"
pretrained_model = "yolov8s.pt"

def train_yolo():
    model = YOLO(pretrained_model)
    model.train(
        data=dataset_yaml,
        epochs=20,
        imgsz=640,
        project="./models",
        name="custom_train",
        verbose=True
    )

def predict_yolo(image_path):
    model = YOLO("./models/custom_train/weights/best.pt")
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.5,
        save=True,
        project="runs/predict",
        name="custom_predict"
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
