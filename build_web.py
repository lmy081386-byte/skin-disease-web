import os
from pathlib import Path

BASE = Path(r"C:\Users\lamam\Documents\skin_web")

# ✅ عدّلي هذا فقط لمسار train عندك
TRAIN_DIR = r"C:\Users\lamam\Documents\Skin_Disease_Dataset\SPLIT_DATASET\train"

# folders
(BASE / "static" / "uploads").mkdir(parents=True, exist_ok=True)
(BASE / "templates").mkdir(parents=True, exist_ok=True)

# requirements
(BASE / "requirements.txt").write_text(
"""flask
tensorflow==2.10.0
numpy==1.23.5
pillow
werkzeug
""", encoding="utf-8"
)

# predict_utils.py
(BASE / "predict_utils.py").write_text(f"""import os
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = os.path.join("models", "effnetB0_final.h5")
TRAIN_DIR = r"{TRAIN_DIR}"
IMG_SIZE = (224, 224)

_model = None
_class_names = None

def _load_class_names():
    classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    classes.sort()
    return classes

def get_model_and_classes():
    global _model, _class_names
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if _class_names is None:
        _class_names = _load_class_names()
    return _model, _class_names

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_topk(image_path, k=3):
    model, class_names = get_model_and_classes()
    x = preprocess_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:k]
    return [{{"label": class_names[i], "confidence": float(probs[i])}} for i in top_idx]
""", encoding="utf-8")

# app.py
(BASE / "app.py").write_text("""import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict_utils import predict_topk

app = Flask(__name__)
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED = {"png", "jpg", "jpeg", "webp"}

def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", results=None, image_url=None, error=None)

    if "file" not in request.files:
        return render_template("index.html", results=None, image_url=None, error="اختاري صورة أولاً.")

    f = request.files["file"]
    if f.filename == "":
        return render_template("index.html", results=None, image_url=None, error="اختاري صورة أولاً.")
    if not allowed_file(f.filename):
        return render_template("index.html", results=None, image_url=None, error="صيغة غير مدعومة. استخدمي jpg/png/jpeg/webp.")

    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    f.save(save_path)

    results = predict_topk(save_path, k=3)
    image_url = f"/static/uploads/{filename}"
    return render_template("index.html", results=results, image_url=image_url, error=None)

if __name__ == "__main__":
    app.run(debug=True)
""", encoding="utf-8")

# style.css
(BASE / "static" / "style.css").write_text("""body{background:#0b1220}
.card{border:0;border-radius:18px}
.hero{background:linear-gradient(135deg,#111a2e,#0b1220);border-radius:22px}
.badge-soft{background:rgba(255,255,255,.08);color:#fff}
.preview-img{max-height:320px;object-fit:contain;border-radius:16px;background:#0f172a}
.small-muted{color:rgba(255,255,255,.7)}
""", encoding="utf-8")

# index.html
(BASE / "templates" / "index.html").write_text("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Skin Disease Classifier</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="/static/style.css" rel="stylesheet">
</head>
<body class="text-white">
  <div class="container py-5">
    <div class="p-4 p-md-5 hero shadow-lg mb-4">
      <div class="d-flex flex-column flex-md-row justify-content-between gap-3 align-items-md-center">
        <div>
          <h1 class="fw-bold mb-2">Skin Disease Classifier</h1>
          <div class="small-muted">Upload an image and get Top-3 predictions (EfficientNet-B0)</div>
        </div>
        <span class="badge badge-soft px-3 py-2">AI Prediction • Not a Diagnosis</span>
      </div>
    </div>

    <div class="row g-4">
      <div class="col-lg-6">
        <div class="card bg-dark text-white shadow-lg">
          <div class="card-body p-4">
            <h5 class="fw-semibold mb-3">Upload Image</h5>

            {% if error %}
              <div class="alert alert-danger">{{ error }}</div>
            {% endif %}

            <form method="post" enctype="multipart/form-data">
              <div class="mb-3">
                <input class="form-control" type="file" name="file" id="fileInput" accept="image/*" required>
              </div>
              <button class="btn btn-primary w-100 fw-semibold" type="submit">Predict</button>
              <div class="small-muted mt-2">Supported: JPG, PNG, JPEG, WEBP</div>
            </form>

            <hr class="border-secondary my-4">

            <h6 class="mb-2">Preview</h6>
            <img id="preview" class="w-100 preview-img" src="{{ image_url if image_url else '' }}" alt="">
            <div id="previewHint" class="small-muted mt-2 {% if image_url %}d-none{% endif %}">
              Choose an image to preview it here.
            </div>
          </div>
        </div>
      </div>

      <div class="col-lg-6">
        <div class="card bg-dark text-white shadow-lg h-100">
          <div class="card-body p-4">
            <h5 class="fw-semibold mb-3">Results</h5>

            {% if results %}
              <div class="row g-3">
                {% for r in results %}
                  <div class="col-12">
                    <div class="p-3 rounded-4" style="background: rgba(255,255,255,.06);">
                      <div class="d-flex justify-content-between align-items-center">
                        <div class="fw-semibold">{{ loop.index }}. {{ r.label }}</div>
                        <div class="fw-bold">{{ (r.confidence*100) | round(2) }}%</div>
                      </div>
                      <div class="progress mt-2" style="height: 10px; background: rgba(255,255,255,.08);">
                        <div class="progress-bar" role="progressbar" style="width: {{ (r.confidence*100) | round(2) }}%"></div>
                      </div>
                    </div>
                  </div>
                {% endfor %}
              </div>

              <div class="mt-4 small-muted">
                *This output is generated by an AI model and should not be used as a medical diagnosis.
              </div>
            {% else %}
              <div class="small-muted">
                Upload an image and click <b>Predict</b> to see the Top-3 predictions here.
              </div>
            {% endif %}
          </div>
        </div>
      </div>
    </div>

    <div class="text-center small-muted mt-4">
      Built with Flask + TensorFlow • EfficientNet-B0
    </div>
  </div>

  <script>
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const hint = document.getElementById("previewHint");
    fileInput?.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      preview.src = url;
      preview.alt = "preview";
      hint.classList.add("d-none");
    });
  </script>
</body>
</html>
""", encoding="utf-8")

print("✅ Project built! Now run:")
print(r"   python -m pip install -r requirements.txt")
print(r"   python app.py")
print(r"   open http://127.0.0.1:5000/")
