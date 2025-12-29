# download_subtitle_models.py
import os
from huggingface_hub import hf_hub_download

# 如果你需要代理（你现在是需要的）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

REPO_ID = "monkt/paddleocr-onnx"
SAVE_DIRECTORY = "models_monkt"

# 需要下载的文件列表 (已使用您提供的正确文件名)
FILES_TO_DOWNLOAD = {
    # File path in the repo : Purpose
    "detection/v5/det.onnx": "v5 Text Detection Model (Best)",
    "languages/chinese/rec.onnx": "v5 Chinese Recognition Model (Best)",
    "languages/chinese/dict.txt": "Dictionary for Chinese Model",

    "preprocessing/textline-orientation/PP-LCNet_x1_0_textline_ori.onnx": "Text Direction Classifier (Corrected Filename)"
}


def download_compatible_models():
    """Downloads a complete set of compatible models for Chinese OCR."""
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    print(f"Models will be saved to: {os.path.abspath(SAVE_DIRECTORY)}")

    for file_path, purpose in FILES_TO_DOWNLOAD.items():
        print(f"\nDownloading: {file_path}\nPurpose: {purpose}")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                local_dir=SAVE_DIRECTORY,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {os.path.basename(file_path)}.")
        except Exception as e:
            print(f"Failed to download {file_path}. Error: {e}")
            return

    print(f"\n✅ All compatible models have been downloaded to the '{SAVE_DIRECTORY}' folder!")


if __name__ == "__main__":
    download_compatible_models()