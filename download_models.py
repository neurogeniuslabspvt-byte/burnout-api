import gdown
from pathlib import Path

MODEL_DIR = Path(__file__).parent

MODELS = {
    "burnout_extratree_classifier.pkl": "1lF6lLPxx8l8rl2vnEc2JYhe7m6PJjf-B",
    "burnout_extratree_regressor.pkl":  "1triPgIa-9jGLQKV79yfCcN7ABr_KOnCn",
}

def download_models():
    for filename, file_id in MODELS.items():
        output_path = MODEL_DIR / filename
        if output_path.exists():
            print(f"✔  {filename} already exists — skipping")
            continue
        print(f"⬇  Downloading {filename} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)
        print(f"✔  {filename} saved")

if __name__ == "__main__":
    download_models()