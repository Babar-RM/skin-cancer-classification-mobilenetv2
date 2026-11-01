import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

def predict_with_tflite(image_path, model_path="skin_analyzer_model.tflite"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    raw_pred = interpreter.get_tensor(output_details[0]["index"])[0][0]

    label = "malignant" if raw_pred > 0.5 else "benign"
    confidence = raw_pred if raw_pred > 0.5 else (1.0 - raw_pred)

    return label, raw_pred, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image> [path_to_model.tflite]")
        sys.exit(1)

    img_file = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else "skin_analyzer_model.tflite"

    try:
        pred_label, score, conf = predict_with_tflite(img_file, model_file)
        print("=" * 45)
        print(" Skin Lesion Classification Result")
        print("=" * 45)
        print(f"Prediction: {pred_label.upper()}")
        print(f"Confidence: {conf * 100:.2f}%")
        print(f"Raw Score:  {score:.4f}")
        print("=" * 45)
    except Exception as e:
        print(f"Error during prediction: {e}")
