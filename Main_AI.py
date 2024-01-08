import cv2
import numpy as np
from keras.models import model_from_json
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load emotion detection model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(model_json)
emotion_model.load_weights("emotiondetector.h5")
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Load face recognition models
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_anyclasses.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
face_recognition_model = load_model('Facenet_model.h5')
haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

def extract_features(image):
    # Convert the image to grayscale if needed
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to (48, 48)
    resized_image = cv2.resize(image, (48, 48))

    # Ensure the image has the necessary shape for the model
    feature = np.array(resized_image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    for x, y, w, h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv2.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        ypred = ypred.reshape(-1, 1, 1, 512)

        face_emotion_pred = emotion_model.predict(extract_features(gray_img[y:y+h, x:x+w]))
        emotion_prediction_label = emotion_labels[face_emotion_pred.argmax()]

        face_name = face_recognition_model.predict(ypred)
        max_values = np.max(face_name, axis=1)
        argmax_indices = np.argmax(face_name, axis=1)
        final_name = encoder.inverse_transform(argmax_indices)[0]

        threshold = 0.1
        print(max_values)
        if max_values < threshold:
            text = f"Name: Unknow \n Emo: ({emotion_prediction_label})"
        else:
            text = f"Name: {final_name} \n Emo: ({emotion_prediction_label})"

        # Desired font size in pixels
        desired_font_size_px = 12

        # Calculate the font scale based on the desired font size
        font_scale = desired_font_size_px / 30.0  # 30 is a rough estimate for a font size of 12 pixels

        # Draw the rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Split the text into lines
        lines = text.split('\n')

        # Calculate the position for placing text next to the rectangle
        text_x = x + w + 10  # Adjust the horizontal position as needed
        text_y = y + 12  # Starting vertical position

        for line in lines:
            # Draw text line by line with the calculated font scale
            cv2.putText(frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
            text_y += int(desired_font_size_px) + 2  # Adjust vertical spacing (optional)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
