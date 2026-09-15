from __future__ import annotations

import json
import platform
from importlib.metadata import version

import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(torch.version.cuda is not None, "This PyTorch build has no CUDA runtime.")
    require(torch.cuda.is_available(), "PyTorch cannot access an NVIDIA GPU.")

    device = torch.device("cuda:0")
    device_index = 0
    # PyTorch 2.3 needs a CUDA context before resetting allocator statistics.
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device_index)

    # CUDA dense kernels and autograd.
    left = torch.randn(1024, 1024, device=device, requires_grad=True)
    right = torch.randn(1024, 1024, device=device)
    loss = left.matmul(right).square().mean()
    loss.backward()
    require(bool(torch.isfinite(loss)), "CUDA dense computation returned a non-finite loss.")
    require(bool(torch.isfinite(left.grad).all()), "CUDA autograd returned a non-finite gradient.")

    # The DCDLP environment also relies on CUDA builds of the PyG extensions.
    from torch_scatter import scatter_add
    from torch_sparse import SparseTensor

    values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    groups = torch.tensor([0, 0, 1, 1], device=device)
    scattered = scatter_add(values, groups, dim=0)
    require(torch.equal(scattered, torch.tensor([3.0, 7.0], device=device)),
            "torch_scatter CUDA result is incorrect.")

    row = torch.tensor([0, 1], device=device)
    col = torch.tensor([1, 0], device=device)
    sparse = SparseTensor(row=row, col=col, value=torch.ones(2, device=device), sparse_sizes=(2, 2))
    sparse_result = sparse.matmul(torch.ones((2, 1), device=device)).reshape(-1)
    require(torch.equal(sparse_result, torch.ones(2, device=device)),
            "torch_sparse CUDA result is incorrect.")

    torch.cuda.synchronize(device)
    payload = {
        "status": "PASS",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_geometric": version("torch-geometric"),
        "torch_scatter": version("torch-scatter"),
        "torch_sparse": version("torch-sparse"),
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "dense_loss": float(loss.detach().cpu()),
        "gradient_finite": True,
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated(device_index) / 2**20, 2),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
