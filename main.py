
# Load Model
model = load_model('./api/expression_detection_modelCNN7565.h5')

def detect_emotion():
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # Change to 1 or 2 if you have multiple cameras

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # For each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the region of interest (the face) for emotion detection
            roi_gray = gray_frame[y:y + h, x:x + w]
            # Resize and normalize the ROI
            resized_face = cv2.resize(roi_gray, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(-1, 48, 48, 1)

            # Make predictions
            predictions = model.predict(reshaped_face)
            predicted_class = np.argmax(predictions)

            # Display the predicted emotion
            cv2.putText(frame, f'Predicted: {emotions[predicted_class]}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Expression Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# To run the real-time detection, uncomment the following line
detect_emotion()
