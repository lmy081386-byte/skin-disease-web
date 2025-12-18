import os
import uuid

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from predict_utils import predict_topk

# ai_utils optional
try:
    from ai_utils import disease_info_ar
except Exception:
    disease_info_ar = None


APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def normalize_results(maybe_results):
    """
    Ensure results is: list of dicts [{'label': str, 'confidence': float}, ...]
    Handles cases where predict_topk returns tuple or nested lists.
    """
    if maybe_results is None:
        return []

    # لو بالغلط وصل tuple (results, top1, is_confident)
    if isinstance(maybe_results, tuple) and len(maybe_results) >= 1:
        maybe_results = maybe_results[0]

    # لو بالغلط صار [[{...}, {...}]]
    if isinstance(maybe_results, list) and len(maybe_results) == 1 and isinstance(maybe_results[0], list):
        maybe_results = maybe_results[0]

    out = []
    if isinstance(maybe_results, list):
        for x in maybe_results:
            if isinstance(x, dict):
                out.append({
                    "label": str(x.get("label", "")),
                    "confidence": float(x.get("confidence", 0.0))
                })
    return out


@app.errorhandler(413)
def file_too_large(e):
    return render_template(
        "index.html",
        results=None,
        top1=None,
        image_url=None,
        error="حجم الصورة كبير جدًا. الحد الأقصى 8MB.",
        ai_text=None,
        is_confident=None
    ), 413


@app.get("/")
def home():
    return render_template(
        "index.html",
        results=None,
        top1=None,
        image_url=None,
        error=None,
        ai_text=None,
        is_confident=None
    )


@app.post("/predict")
def predict():
    image_url = None
    results = None
    top1 = None
    is_confident = None
    ai_text = None
    error = None

    try:
        if "image" not in request.files:
            raise ValueError("لم يتم إرسال صورة.")

        image = request.files["image"]
        notes = request.form.get("notes", "")

        if not image or image.filename == "":
            raise ValueError("لم يتم اختيار ملف.")

        if not allowed_file(image.filename):
            raise ValueError("صيغة الملف غير مدعومة. استخدمي JPG/PNG/JPEG/WEBP فقط.")

        safe_name = secure_filename(image.filename)
        filename = f"{uuid.uuid4().hex}_{safe_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(save_path)

        image_url = url_for("static", filename=f"uploads/{filename}")

        # ✅ لازم نفك 3 قيم من predict_topk
        raw_results, top1_label, is_confident = predict_topk(save_path, k=3, threshold=0.30)

        # ✅ طبع النتائج لنتأكد (اختياري)
        # print("RAW:", type(raw_results), raw_results)

        results = normalize_results(raw_results)

        # top1 dict للعرض
        if results:
            top1 = results[0]
        else:
            top1 = {"label": str(top1_label or ""), "confidence": 0.0}

        label_for_ai = top1["label"] if isinstance(top1, dict) else str(top1)

        if disease_info_ar:
            ai_text = disease_info_ar(label_for_ai, results, notes)
        else:
            ai_text = None

    except Exception as e:
        error = f"خطأ أثناء التنبؤ: {e}"

    return render_template(
        "index.html",
        results=results,
        top1=top1,
        image_url=image_url,
        error=error,
        ai_text=ai_text,
        is_confident=is_confident
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
