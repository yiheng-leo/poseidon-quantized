"""Validate the optimizer runtime before expensive quantization preparation.

Creating an optimizer can lazily load torch._dynamo even in eager mode. A
partially updated PyTorch installation may run forwards but fail here. This
probe exercises the actual optimizer on a tiny tensor; it is not a workaround
for incompatible Python/native packages.
"""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import sys

import torch


def runtime_details(device, optimizer_name):
    try:
        installed_version = version('torch')
    except PackageNotFoundError:
        installed_version = None
    return {
        'python': sys.executable,
        'torch_version': str(torch.__version__),
        'installed_torch_version': installed_version,
        'torch_python': torch.__file__,
        'torch_native': getattr(torch._C, '__file__', None),
        'cuda_build': torch.version.cuda,
        'device': str(device),
        'optimizer': optimizer_name,
    }


def check_optimizer_runtime(device, optimizer_name='Adam'):
    """Exercise construction and one update without consuming random numbers."""
    if optimizer_name not in {'Adam', 'AdamW'}:
        raise ValueError(f'Unsupported optimizer probe: {optimizer_name}')
    details = runtime_details(device, optimizer_name)
    try:
        with torch.enable_grad():
            parameter = torch.nn.Parameter(torch.ones(1, device=device))
            optimizer = getattr(torch.optim, optimizer_name)([parameter], lr=1e-3)
            parameter.sum().backward()
            optimizer.step()
    except Exception as error:
        raise RuntimeError(
            f'PyTorch {optimizer_name} runtime check failed before calibration/model loading.\n'
            f'{type(error).__name__}: {error}\n'
            f'{json.dumps(details, indent=2)}\n'
            'Run python -m quantization.runtime --DEV cpu in a fresh process. '
            'An internal torch import failure can indicate inconsistent Python/native '
            'PyTorch files or an environment updated while a long job was running. '
            'If a fresh process also fails, resolve the reported installation/backend error '
            'before retrying; this probe does not change PyTorch.'
        ) from error
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--DEV', '--device', default='cpu')
    parser.add_argument('--optimizer', choices=('Adam', 'AdamW'), default='Adam')
    args = parser.parse_args(argv)
    try:
        details = check_optimizer_runtime(torch.device(args.DEV), args.optimizer)
    except RuntimeError as error:
        parser.exit(1, str(error) + '\n')
    print(json.dumps(details, indent=2))
    print(f'{args.optimizer} OK ({args.DEV})')


if __name__ == '__main__':
    main()
