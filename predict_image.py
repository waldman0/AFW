from classifiers import Meso4
import cv2
import numpy as np

# Modell initialisieren
model = Meso4()
model.load('weights/Meso4_DF.h5')  # oder Meso4_F2F.h5


def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    print(f"Prediction Score: {prediction:.4f}")
    if prediction >= 0.5:
        print("→ Echt")
    else:
        print("→ Deepfake")


predict_image("Bilder/1.jpg")  # oder: "test_images/1.jpg", je nach Pfad
predict_image("Bilder/2.jpg")
predict_image("Bilder/3.jpg")
predict_image("Bilder/4.jpg")
predict_image("Bilder/5.jpg")
predict_image("Bilder/6.png")
predict_image("Bilder/7.png")
predict_image("Bilder/8.png")
predict_image("Bilder/9.png")
predict_image("Bilder/10.png")
predict_image("Bilder/11.png")
predict_image("Bilder/12.jpg")


