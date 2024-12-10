import cv2
import matplotlib.pyplot as plt


def find_bounding_boxes_with_preview(image_path):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    # Converte in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applica il threshold (puoi modificare i valori)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Trova i contorni
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ottieni i bounding box
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Disegna i bounding box sull'immagine originale
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Converti in RGB per matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mostra l'immagine con i bounding box
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Bounding Boxes Trovati")
    plt.show()


# Esempio di utilizzo
find_bounding_boxes_with_preview("../Yolo/datasets/train/images/almond_template_2.png")
