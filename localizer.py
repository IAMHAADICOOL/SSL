"""
PassiveTriangularLocalizer – Real-World Ready Version
----------------------------------------------------
Accepts either theoretical *or* calibrated hydrophone positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from array_geometry import TriangularArray, CalibratedTriangularArray, tdoa_to_angles
from signal_processing import gcc_phase_delay

__all__ = ["PassiveTriangularLocalizer"]

@dataclass
class PassiveTriangularLocalizer:
    fs: float                                   # ADC sampling rate (Hz)
    sound_speed: float                          # Local sound speed (m/s)
    array: object                               # TriangularArray *or* CalibratedTriangularArray
    f_band: Tuple[float, float] = (300.0, 800.0)  # Band where source is strong

    # Optional: Store last result
    last_result: dict = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------
    def locate(self, sig0: np.ndarray, sig1: np.ndarray, sig2: np.ndarray) -> dict:
        """Estimate azimuth, pitch, and raw TDOAs for a data snapshot."""
        fmin, fmax = self.f_band

        # 1. Cross-spectrum phase TDOAs
        τ01 = gcc_phase_delay(sig0, sig1, self.fs, fmin, fmax)
        τ02 = gcc_phase_delay(sig0, sig2, self.fs, fmin, fmax)
        τ12 = gcc_phase_delay(sig1, sig2, self.fs, fmin, fmax)

        # 2. Convert TDOAs to angles
        d = self.array.d  # effective side length
        φ, θ = tdoa_to_angles(τ01, τ02, τ12, d, self.sound_speed)

        # Package result
        self.last_result = {
            'tau01': τ01,
            'tau02': τ02,
            'tau12': τ12,
            'azimuth_rad': φ,
            'pitch_rad': θ
        }
        return self.last_result

    # ------------------------------------------------------------------
    @classmethod
    def from_measured_positions(cls, fs: float, sound_speed: float,
                                L0_pos: list, L1_pos: list, L2_pos: list,
                                f_band=(300.0, 800.0)) -> 'PassiveTriangularLocalizer':
        """Convenience constructor with three measured hydrophone positions."""
        array = CalibratedTriangularArray(np.array(L0_pos), np.array(L1_pos), np.array(L2_pos))
        return cls(fs=fs, sound_speed=sound_speed, array=array, f_band=f_band)
