import os
import json
import threading
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================
# Paths (portable)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "models", "effnetB0_final.h5")
)

# خيار 1: ياخد أسماء الكلاسات من فولدر train (لو موجود على نفس المشروع)
# خيار 2 (الأفضل للويب): ملف classes.json داخل models/
TRAIN_DIR = os.environ.get(
    "TRAIN_DIR",
    os.path.join(BASE_DIR, "SPLIT_DATASET", "train")
)

CLASSES_JSON = os.environ.get(
    "CLASSES_JSON",
    os.path.join(BASE_DIR, "models", "classes.json")
)

IMG_SIZE = (224, 224)

_model = None
_class_names = None
_lock = threading.Lock()


def _load_class_names_from_train_dir(train_dir: str):
    if not train_dir or (not os.path.exists(train_dir)):
        return None
    classes = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    classes.sort()
    return classes if classes else None


def _load_class_names_from_json(json_path: str):
    if not json_path or (not os.path.exists(json_path)):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # متوقع يكون: {"class_names": ["acne", "...", ...]}
        classes = data.get("class_names", None)
        if isinstance(classes, list) and len(classes) > 0:
            return classes
        return None
    except Exception:
        return None


def get_model_and_classes():
    """
    Loads model once + class names once.
    Priority for class_names:
      1) TRAIN_DIR folders (if exists)
      2) models/classes.json (recommended for web)
    """
    global _model, _class_names

    with _lock:
        if _model is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        if _class_names is None:
            classes = _load_class_names_from_train_dir(TRAIN_DIR)
            if classes is None:
                classes = _load_class_names_from_json(CLASSES_JSON)

            if classes is None:
                raise FileNotFoundError(
                    "Could not load class names. "
                    "Either provide TRAIN_DIR with class folders or create models/classes.json"
                )

            _class_names = classes
            print("CLASS NAMES LOADED:", _class_names)

    return _model, _class_names


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)  # ✅ EfficientNet preprocess
    return np.expand_dims(arr, axis=0)


def predict_topk(image_path, k=3, threshold=0.30):
    model, class_names = get_model_and_classes()
    x = preprocess_image(image_path)

    preds = model.predict(x, verbose=0)[0]

    # ✅ لو logits: طبّق softmax
    if preds.min() < 0 or preds.max() > 1.0:
        preds = tf.nn.softmax(preds).numpy()

    # حماية إذا عدد الكلاسات أقل من k
    k = min(k, len(class_names))
    top_idx = np.argsort(preds)[::-1][:k]

    results = [{"label": class_names[i], "confidence": float(preds[i])} for i in top_idx]

    top1 = results[0]["label"] if results else None
    is_confident = (results[0]["confidence"] >= threshold) if results else False
    return results, top1, is_confident
