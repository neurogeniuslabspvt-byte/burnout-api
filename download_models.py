import gdown
from pathlib import Path

MODEL_DIR = Path(__file__).parent

MODELS = {
    "burnout_extratree_classifier.pkl": "https://drive.google.com/file/d/1lF6lLPxx8l8rl2vnEc2JYhe7m6PJjf-B/view?usp=drive_link",
    "burnout_extratree_regressor.pkl":  "https://drive.google.com/file/d/1triPgIa-9jGLQKV79yfCcN7ABr_KOnCn/view?usp=drive_link",
}

def download_models():
    for filename, url in MODELS.items():
        output_path = MODEL_DIR / filename
        if output_path.exists():
            print(f"✔  {filename} already exists — skipping")
            continue
        print(f"⬇  Downloading {filename} ...")
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        print(f"✔  {filename} saved")

if __name__ == "__main__":
    download_models()
