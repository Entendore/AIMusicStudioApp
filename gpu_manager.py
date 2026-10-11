"""Smart GPU Manager — detection, VRAM monitoring, auto device selection, memory management.

Provides a centralised interface for all GPU-related decisions:
  - Detect available GPUs (CUDA, MPS/Apple Silicon)
  - Query real-time VRAM usage
  - Auto-select best device for a given model
  - Warn if VRAM is insufficient
  - Garbage-collect GPU memory between model loads
  - Track model-to-device assignments
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ════════════════════════════════════════════════════════════════════

class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"          # Apple Silicon
    ROCM = "rocm"        # AMD ROCm


@dataclass
class GPUInfo:
    """Static + dynamic info about a single GPU device."""
    index: int = 0
    name: str = "Unknown"
    device_type: DeviceType = DeviceType.CUDA
    total_vram_mb: int = 0
    free_vram_mb: int = 0
    used_vram_mb: int = 0
    temperature_c: int = 0
    utilisation_pct: int = 0          # GPU compute utilisation %
    cuda_version: str = ""
    driver_version: str = ""
    compute_capability: str = ""

    @property
    def vram_usage_pct(self) -> float:
        if self.total_vram_mb <= 0:
            return 0.0
        return (self.used_vram_mb / self.total_vram_mb) * 100.0

    @property
    def free_vram_gb(self) -> float:
        return self.free_vram_mb / 1024.0

    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_mb / 1024.0

    @property
    def short_name(self) -> str:
        """Shortened GPU name for UI display."""
        name = self.name
        # Strip common prefixes
        for prefix in ("NVIDIA ", "AMD ", "Intel "):
            if name.startswith(prefix):
                name = name[len(prefix):]
        # Truncate long names
        if len(name) > 30:
            name = name[:27] + "…"
        return name


@dataclass
class DeviceRecommendation:
    """Result of smart device selection for a model."""
    device: str                     # "cuda:0", "cuda:1", "cpu", "mps"
    device_type: DeviceType
    gpu: Optional[GPUInfo]
    reason: str                     # Human-readable explanation
    vram_sufficient: bool
    vram_headroom_mb: int           # Free VRAM after model load estimate
    estimated_model_vram_mb: int


# ════════════════════════════════════════════════════════════════════
#  GPU MANAGER
# ════════════════════════════════════════════════════════════════════

class GPUManager:
    """Centralised GPU detection, monitoring, and smart device selection.

    Usage::

        gm = GPUManager()
        gm.refresh()
        print(gm.summary())
        rec = gm.recommend_device(vram_needed_gb=4.0)
        gm.release_memory()
    """

    def __init__(self):
        self.gpus: list[GPUInfo] = []
        self.has_cuda: bool = False
        self.has_mps: bool = False
        self.has_rocm: bool = False
        self.cuda_version: str = ""
        self.torch_available: bool = False
        self.torch_version: str = ""
        self._torch_module = None

    # ── Detection ──────────────────────────────────────────────────

    def refresh(self):
        """Re-detect all GPU devices and refresh VRAM stats."""
        self.gpus.clear()
        self._detect_torch()
        self._detect_cuda()
        self._detect_mps()
        self._detect_rocm()
        self._query_nvidia_smi()

        if not self.gpus:
            logger.info("No GPU detected — CPU-only mode")
        else:
            for gpu in self.gpus:
                logger.info(
                    "GPU %d: %s  %.0f MB VRAM (%.0f MB free)  %s",
                    gpu.index, gpu.name, gpu.total_vram_mb,
                    gpu.free_vram_mb, gpu.driver_version,
                )

    def _detect_torch(self):
        """Check if PyTorch is importable and which version."""
        try:
            import torch
            self._torch_module = torch
            self.torch_available = True
            self.torch_version = torch.__version__
            logger.debug("PyTorch %s available", self.torch_version)
        except ImportError:
            self.torch_available = False
            self.torch_version = ""

    def _detect_cuda(self):
        """Check CUDA availability via PyTorch."""
        if not self.torch_available:
            self.has_cuda = False
            return
        try:
            torch = self._torch_module
            self.has_cuda = torch.cuda.is_available()
            if self.has_cuda:
                self.cuda_version = torch.version.cuda or ""
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    # Only add if not already populated by nvidia-smi
                    if not any(g.index == i for g in self.gpus):
                        gpu = GPUInfo(
                            index=i,
                            name=torch.cuda.get_device_name(i),
                            device_type=DeviceType.CUDA,
                            total_vram_mb=torch.cuda.get_device_properties(i).total_mem // (1024 * 1024),
                            cuda_version=self.cuda_version,
                        )
                        gpu.free_vram_mb, gpu.used_vram_mb = self._query_torch_vram(i, gpu.total_vram_mb)
                        self.gpus.append(gpu)
        except Exception as e:
            logger.debug("CUDA detection error: %s", e)
            self.has_cuda = False

    def _detect_mps(self):
        """Check Apple Silicon MPS availability."""
        if not self.torch_available:
            return
        try:
            torch = self._torch_module
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.has_mps = True
                # MPS doesn't expose VRAM info — use system memory as proxy
                gpu = GPUInfo(
                    index=len(self.gpus),
                    name="Apple Silicon (MPS)",
                    device_type=DeviceType.MPS,
                    total_vram_mb=self._get_mac_memory_mb(),
                    free_vram_mb=self._get_mac_memory_mb(),  # Shared memory
                )
                self.gpus.append(gpu)
                logger.info("Apple MPS detected — shared memory architecture")
        except Exception:
            pass

    def _detect_rocm(self):
        """Check AMD ROCm availability."""
        if not self.torch_available:
            return
        try:
            torch = self._torch_module
            if self.has_cuda and hasattr(torch.version, "hip"):
                self.has_rocm = True
                for gpu in self.gpus:
                    if gpu.device_type == DeviceType.CUDA:
                        gpu.device_type = DeviceType.ROCM
        except Exception:
            pass

    def _query_nvidia_smi(self):
        """Use nvidia-smi to get detailed GPU info (works without PyTorch)."""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.free,memory.used,"
                 "temperature.gpu,utilization.gpu,driver_version,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return

            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 9:
                    continue
                idx = int(parts[0])
                # Update existing or create new
                existing = next((g for g in self.gpus if g.index == idx), None)
                if existing:
                    existing.name = parts[1]
                    existing.total_vram_mb = int(float(parts[2]))
                    existing.free_vram_mb = int(float(parts[3]))
                    existing.used_vram_mb = int(float(parts[4]))
                    existing.temperature_c = int(float(parts[5]))
                    existing.utilisation_pct = int(float(parts[6]))
                    existing.driver_version = parts[7]
                    existing.compute_capability = parts[8]
                else:
                    self.gpus.append(GPUInfo(
                        index=idx,
                        name=parts[1],
                        device_type=DeviceType.CUDA,
                        total_vram_mb=int(float(parts[2])),
                        free_vram_mb=int(float(parts[3])),
                        used_vram_mb=int(float(parts[4])),
                        temperature_c=int(float(parts[5])),
                        utilisation_pct=int(float(parts[6])),
                        driver_version=parts[7],
                        compute_capability=parts[8],
                    ))
                self.has_cuda = True
        except FileNotFoundError:
            pass  # nvidia-smi not found — non-NVIDIA system
        except Exception as e:
            logger.debug("nvidia-smi query failed: %s", e)

    @staticmethod
    def _query_torch_vram(device_idx: int, total_mb: int) -> tuple[int, int]:
        """Query VRAM via PyTorch CUDA API."""
        try:
            import torch
            free, total = torch.cuda.mem_get_info(device_idx)
            free_mb = free // (1024 * 1024)
            used_mb = (total - free) // (1024 * 1024)
            return free_mb, used_mb
        except Exception:
            return total_mb, 0

    @staticmethod
    def _get_mac_memory_mb() -> int:
        """Get total physical memory on macOS (for MPS shared memory)."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=3,
            )
            return int(result.stdout.strip()) // (1024 * 1024)
        except Exception:
            return 8192  # Assume 8 GB

    # ── VRAM Refresh ───────────────────────────────────────────────

    def refresh_vram(self):
        """Refresh VRAM stats without full re-detection."""
        # Try nvidia-smi first (fastest, no torch needed)
        self._query_nvidia_smi()
        # Also try torch if available
        if self.torch_available and self.has_cuda:
            try:
                torch = self._torch_module
                for gpu in self.gpus:
                    if gpu.device_type == DeviceType.CUDA:
                        free, total = torch.cuda.mem_get_info(gpu.index)
                        gpu.free_vram_mb = free // (1024 * 1024)
                        gpu.used_vram_mb = (total - free) // (1024 * 1024)
                        gpu.total_vram_mb = total // (1024 * 1024)
            except Exception:
                pass

    # ── Smart Device Selection ─────────────────────────────────────

    def recommend_device(self, vram_needed_gb: float = 0.0,
                         model_name: str = "",
                         allow_cpu_fallback: bool = True) -> DeviceRecommendation:
        """Select the best device for a model requiring *vram_needed_gb* VRAM.

        Strategy:
            1. Find the GPU with the most free VRAM that can fit the model
            2. If no GPU has enough VRAM, try the one with the most total VRAM
               (may work with quantisation / offloading)
            3. Fall back to CPU if allowed
            4. Prefer CUDA over MPS over CPU
        """
        vram_needed_mb = int(vram_needed_gb * 1024)
        # Add 20% overhead for PyTorch CUDA context, activations, etc.
        vram_with_overhead_mb = int(vram_needed_mb * 1.2)

        # Sort GPUs by free VRAM descending
        sorted_gpus = sorted(self.gpus, key=lambda g: g.free_vram_mb, reverse=True)

        best_gpu: Optional[GPUInfo] = None
        reason = ""

        for gpu in sorted_gpus:
            if gpu.free_vram_mb >= vram_with_overhead_mb:
                best_gpu = gpu
                reason = (
                    f"{gpu.short_name} has {gpu.free_vram_gb:.1f} GB free — "
                    f"sufficient for {model_name or 'this model'} "
                    f"({vram_needed_gb:.1f} GB needed)"
                )
                break

        if best_gpu is None and sorted_gpus:
            # No GPU has enough free VRAM — try the biggest one
            biggest = sorted_gpus[0]
            if biggest.total_vram_gb >= vram_needed_gb:
                best_gpu = biggest
                reason = (
                    f"WARNING: {biggest.short_name} only has {biggest.free_vram_gb:.1f} GB free, "
                    f"but {vram_needed_gb:.1f} GB needed. Model may use CPU offloading or quantisation."
                )
            else:
                reason = (
                    f"No GPU has enough VRAM for {model_name or 'this model'} "
                    f"({vram_needed_gb:.1f} GB). Largest GPU: {biggest.short_name} "
                    f"({biggest.total_vram_gb:.1f} GB total)."
                )

        if best_gpu:
            device_str = f"cuda:{best_gpu.index}" if best_gpu.device_type == DeviceType.CUDA else "mps"
            vram_sufficient = best_gpu.free_vram_mb >= vram_with_overhead_mb
            headroom = best_gpu.free_vram_mb - vram_with_overhead_mb
            return DeviceRecommendation(
                device=device_str,
                device_type=best_gpu.device_type,
                gpu=best_gpu,
                reason=reason,
                vram_sufficient=vram_sufficient,
                vram_headroom_mb=headroom,
                estimated_model_vram_mb=vram_with_overhead_mb,
            )

        # CPU fallback
        if allow_cpu_fallback:
            return DeviceRecommendation(
                device="cpu",
                device_type=DeviceType.CPU,
                gpu=None,
                reason=f"No GPU available — using CPU. Generation will be significantly slower.",
                vram_sufficient=False,
                vram_headroom_mb=0,
                estimated_model_vram_mb=0,
            )

        return DeviceRecommendation(
            device="cpu",
            device_type=DeviceType.CPU,
            gpu=None,
            reason="No suitable GPU found and CPU fallback is disabled.",
            vram_sufficient=False,
            vram_headroom_mb=0,
            estimated_model_vram_mb=0,
        )

    def get_device_for_config(self, config: dict) -> str:
        """Get the configured device string, with smart auto-detection if set to 'auto'."""
        device = config.get('hardware', {}).get('device', 'auto')
        if device == 'auto':
            rec = self.recommend_device()
            return rec.device
        # Validate that the configured device is actually available
        if device.startswith('cuda') and not self.has_cuda:
            logger.warning(f"Configured device '{device}' not available — falling back to CPU")
            return 'cpu'
        if device == 'mps' and not self.has_mps:
            logger.warning(f"Configured device 'mps' not available — falling back to CPU")
            return 'cpu'
        return device

    # ── Memory Management ──────────────────────────────────────────

    def release_memory(self):
        """Force-release unused GPU memory (PyTorch cache clear)."""
        if not self.torch_available or not self.has_cuda:
            return
        try:
            torch = self._torch_module
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
        except Exception as e:
            logger.debug("GPU cache clear failed: %s", e)

    def release_all_memory(self):
        """Aggressively release all GPU memory."""
        if not self.torch_available or not self.has_cuda:
            return
        try:
            torch = self._torch_module
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Full GPU memory release completed")
        except Exception as e:
            logger.debug("Full GPU release failed: %s", e)

    def get_memory_summary(self) -> str:
        """Return a one-line memory summary for all GPUs."""
        if not self.gpus:
            return "No GPU — CPU only"
        parts = []
        for gpu in self.gpus:
            pct = gpu.vram_usage_pct
            parts.append(f"{gpu.short_name}: {gpu.used_vram_mb}/{gpu.total_vram_mb} MB ({pct:.0f}%)")
        return "  |  ".join(parts)

    # ── Summary ────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a dict summary suitable for UI display."""
        return {
            'has_cuda': self.has_cuda,
            'has_mps': self.has_mps,
            'has_rocm': self.has_rocm,
            'torch_available': self.torch_available,
            'torch_version': self.torch_version,
            'cuda_version': self.cuda_version,
            'gpu_count': len(self.gpus),
            'gpus': [
                {
                    'index': g.index,
                    'name': g.name,
                    'short_name': g.short_name,
                    'total_vram_gb': g.total_vram_gb,
                    'free_vram_gb': g.free_vram_gb,
                    'used_vram_mb': g.used_vram_mb,
                    'usage_pct': g.vram_usage_pct,
                    'temperature': g.temperature_c,
                    'utilization': g.utilisation_pct,
                    'driver': g.driver_version,
                    'compute_cap': g.compute_capability,
                }
                for g in self.gpus
            ],
            'best_device': self.recommend_device().device,
        }


# ════════════════════════════════════════════════════════════════════
#  SINGLETON INSTANCE
# ════════════════════════════════════════════════════════════════════

_global_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Return the global GPUManager singleton (lazy-initialised)."""
    global _global_gpu_manager
    if _global_gpu_manager is None:
        _global_gpu_manager = GPUManager()
        _global_gpu_manager.refresh()
    return _global_gpu_manager


def refresh_gpu() -> GPUManager:
    """Force-refresh the global GPU manager and return it."""
    gm = get_gpu_manager()
    gm.refresh()
    return gm
