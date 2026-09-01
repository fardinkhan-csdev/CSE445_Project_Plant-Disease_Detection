import os
import sys
import zipfile
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PLANTDOC_DIR = os.path.join(PROJECT_ROOT, "data", "plantdoc_raw")
ZIP_PATH = os.path.join(PROJECT_ROOT, "data", "plantdoc_dataset.zip")

start_time = None

def download_progress(count, block_size, total_size):
    global start_time
    if start_time is None:
        start_time = time.time()
    
    downloaded = count * block_size
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    duration = time.time() - start_time
    speed = (downloaded / (1024 * 1024)) / duration if duration > 0 else 0

    downloaded_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024) if total_size > 0 else 0

    sys.stdout.write(f"\rDownloading PlantDoc: [{downloaded_mb:.1f} MB / {total_mb:.1f} MB] ({percent}%) @ {speed:.2f} MB/s")
    sys.stdout.flush()

def download_and_extract():
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
    
    print(f"Downloading PlantDoc dataset from {url} ...")
    if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 1000000:
        urllib.request.urlretrieve(url, ZIP_PATH, reporthook=download_progress)
        print("\nDownload finished!")
    else:
        print("Zip already exists locally.")

    print(f"Extracting PlantDoc dataset to {PLANTDOC_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(PLANTDOC_DIR)
    
    print("PlantDoc dataset ready in data/plantdoc_raw!")

if __name__ == "__main__":
    download_and_extract()
