import tensorflow as tf
import numpy as np

# Load the .keras model (ensure the path matches)
model_path = "api/model/cnn_model.keras"
model = tf.keras.models.load_model(model_path)

def predict(image: np.ndarray):
    """
    Run a prediction using the CNN model.
    Args:
        image (numpy.ndarray): Input image preprocessed for the model.
    Returns:
        prediction (list): The model's prediction.
    """
    # Ensure the input image is expanded to (1, height, width, channels)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction.tolist()
