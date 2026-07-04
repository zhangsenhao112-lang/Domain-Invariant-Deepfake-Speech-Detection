"""Dataset and protocol handling without machine-specific paths."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .rawboost import process_Rawboost_feature


LABELS = {"spoof": 0, "fake": 0, "bonafide": 1, "bona-fide": 1, "real": 1}


@dataclass(frozen=True)
class Trial:
    path: str
    label: int | None


@dataclass
class RawBoostConfig:
    nBands: int = 5
    minF: int = 20
    maxF: int = 8000
    minBW: int = 100
    maxBW: int = 1000
    minCoeff: int = 10
    maxCoeff: int = 100
    minG: int = 0
    maxG: int = 0
    minBiasLinNonLin: int = 5
    maxBiasLinNonLin: int = 20
    N_f: int = 5
    P: int = 10
    g_sd: int = 2
    SNRmin: int = 10
    SNRmax: int = 40


def read_protocol(path: str | Path, protocol_format: str = "simple") -> list[Trial]:
    """Read ``relative/path label`` or an ASVspoof five-column protocol."""
    trials = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if protocol_format == "asvspoof":
                if len(fields) < 5:
                    raise ValueError(f"{path}:{line_number}: expected 5 columns")
                audio_path, label = fields[1], fields[4]
            else:
                if len(fields) not in (1, 2):
                    raise ValueError(
                        f"{path}:{line_number}: expected '<path> [label]'"
                    )
                audio_path = fields[0]
                label = fields[1] if len(fields) == 2 else None
            normalized = label.lower() if label is not None else None
            if normalized is not None and normalized not in LABELS:
                raise ValueError(f"{path}:{line_number}: unknown label '{label}'")
            trials.append(Trial(audio_path, LABELS.get(normalized)))
    return trials


def repeat_pad(waveform: np.ndarray, length: int) -> np.ndarray:
    if waveform.size == 0:
        raise ValueError("empty audio file")
    if waveform.size >= length:
        return waveform[:length]
    return np.tile(waveform, length // waveform.size + 1)[:length]


class AudioTrials(Dataset):
    def __init__(
        self,
        root: str | Path,
        trials: list[Trial],
        sample_rate: int = 16000,
        num_samples: int = 66800,
        rawboost_algo: int = 0,
        audio_extension: str = "",
    ):
        self.root = Path(root)
        self.trials = trials
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.rawboost_algo = rawboost_algo
        self.audio_extension = audio_extension
        self.rawboost = RawBoostConfig()

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, index: int):
        trial = self.trials[index]
        relative_path = trial.path
        if self.audio_extension and not Path(relative_path).suffix:
            relative_path += self.audio_extension
        path = self.root / relative_path
        waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if source_rate != self.sample_rate:
            divisor = np.gcd(source_rate, self.sample_rate)
            waveform = resample_poly(
                waveform, self.sample_rate // divisor, source_rate // divisor
            ).astype(np.float32)
        if self.rawboost_algo:
            waveform = process_Rawboost_feature(
                waveform, self.sample_rate, self.rawboost, self.rawboost_algo
            )
        waveform = torch.from_numpy(repeat_pad(waveform, self.num_samples).copy())
        if trial.label is None:
            return waveform, trial.path
        return waveform, trial.label, trial.path
