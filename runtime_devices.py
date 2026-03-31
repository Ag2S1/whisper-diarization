from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeDevices:
    demucs_device: str
    whisper_device: str
    whisper_compute_type: str
    aligner_device: str
    aligner_dtype: torch.dtype
    diarizer_device: str


def resolve_runtime_devices(requested_device: str) -> RuntimeDevices:
    whisper_device = "cuda" if requested_device == "cuda" else "cpu"
    whisper_compute_type = "float16" if requested_device == "cuda" else "int8"
    aligner_dtype = torch.float16 if requested_device == "cuda" else torch.float32

    return RuntimeDevices(
        demucs_device=requested_device,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        aligner_device=requested_device,
        aligner_dtype=aligner_dtype,
        diarizer_device=requested_device,
    )
