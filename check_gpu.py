import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability (sm):", torch.cuda.get_device_capability(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).sum().item()
    print("Matmul on GPU OK, sum =", y)
else:
    print("!!! CUDA NOT available — check driver / install")
