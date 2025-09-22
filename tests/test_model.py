# tests/test_model.py
import os
import tensorflow as tf

def test_model_file_exists():
    """Check if the saved model file exists"""
    assert os.path.exists("cnn_pneumonia_model.keras"), "Model file is missing!"

def test_model_loads():
    """Check if the model loads successfully"""
    model = tf.keras.models.load_model("cnn_pneumonia_model.keras")
    assert model is not None, "Failed to load the model!"
    assert model.input_shape[1:] == (224, 224, 3), "Unexpected input shape!"
    assert model.output_shape[-1] == 1, "Model should output a single probability!"
