import os
import zipfile
import requests

# Ganti dengan Raw URL dari Roboflow (sesuai yang kamu berikan)
roboflow_url = "https://app.roboflow.com/ds/m8o3xApkx2?key=NT0nabTXVb"

# Direktori penyimpanan dataset
dataset_dir = "helmet_dataset"
os.makedirs(dataset_dir, exist_ok=True)

# File ZIP tujuan
zip_path = os.path.join(dataset_dir, "dataset.zip")

# Download dataset
print("Downloading dataset...")
response = requests.get(roboflow_url, stream=True)

with open(zip_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

# Ekstrak dataset
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(dataset_dir)

# Hapus file ZIP setelah ekstraksi
os.remove(zip_path)

print("Dataset berhasil diunduh dan diekstrak ke folder:", dataset_dir)
