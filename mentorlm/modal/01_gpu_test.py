import modal

image = modal.Image.debian_slim().pip_install("torch")

app = modal.App("mentorlm-gpu-test")

@app.function(gpu="T4", image=image)
def check_gpu():
    import subprocess
    import torch

    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU detectada:", torch.cuda.get_device_name(0))
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

@app.local_entrypoint()
def main():
    check_gpu.remote()
