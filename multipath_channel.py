"""
Enhanced Multipath Channel Model for Real Underwater Environments
----------------------------------------------------------------
This module implements the virtual source ray-theory model with real-world
environmental parameters and seafloor characterization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

__all__ = [
    "VirtualSource",
    "UnderwaterMultipathChannel",
    "SeafloorType",
]

class SeafloorType:
    """Common seafloor reflection coefficients for different bottom types"""
    SAND = 0.3
    MUD = 0.2  
    CLAY = 0.15
    ROCK = 0.8
    CORAL = 0.75
    GRAVEL = 0.4

@dataclass
class VirtualSource:
    """Metadata for a single virtual source path."""
    position: np.ndarray  # (x, y, z) coordinates (m)
    coeff: float          # reflection coefficient / amplitude scaling
    order: int            # reflection order (0 = direct)
    label: str            # descriptive label e.g. "direct", "surface"

    def distance_to(self, rx_pos: np.ndarray) -> float:
        """Euclidean distance from virtual source to receiver (m)."""
        return float(np.linalg.norm(self.position - rx_pos))

class UnderwaterMultipathChannel:
    """Enhanced multipath channel for real underwater deployment."""

    def __init__(self,
                 sea_depth: float,                    # MEASURE: Site bathymetry
                 array_depth: float,                  # MEASURE: Hydrophone depth  
                 sound_speed: float,                  # MEASURE: CTD/sound velocity probe
                 bottom_type: float = SeafloorType.SAND,  # ESTIMATE: From sonar/samples
                 max_order: int = 2,
                 surface_loss_db: float = 0.5):      # Sea surface absorption

        self.B = sea_depth
        self.b = array_depth  
        self.c = sound_speed
        self.bottom_coeff = bottom_type
        self.max_order = max_order
        self.surface_coeff = -10**(-surface_loss_db/20)  # Convert dB to linear

        # Validate parameters
        if sea_depth <= array_depth:
            raise ValueError(f"Sea depth ({sea_depth}m) must be > array depth ({array_depth}m)")
        if sound_speed < 1400 or sound_speed > 1600:
            print(f"Warning: Sound speed {sound_speed} m/s is outside typical range 1400-1600 m/s")

    @classmethod
    def from_site_survey(cls, 
                        site_depth: float,
                        hydrophone_depth: float, 
                        water_temp_c: float,
                        salinity_ppt: float = 35.0,
                        bottom_type: float = SeafloorType.SAND) -> 'UnderwaterMultipathChannel':
        """Create channel from measured environmental parameters."""

        # UNESCO sound speed formula (simplified)
        c = 1449.2 + 4.6*water_temp_c - 0.055*water_temp_c**2 + 0.00029*water_temp_c**3
        c += (1.34 - 0.010*water_temp_c) * (salinity_ppt - 35)
        c += 0.016 * hydrophone_depth  # Pressure effect

        return cls(site_depth, hydrophone_depth, c, bottom_type)

    def _generate_virtual_sources(self, src_pos: np.ndarray) -> List[VirtualSource]:
        """Generate virtual sources with realistic reflection coefficients."""
        x, y, z = src_pos
        vs: List[VirtualSource] = []

        # Direct path
        vs.append(VirtualSource(
            position=np.array([x, y, z]), 
            coeff=1.0, 
            order=0, 
            label="direct"
        ))

        # 1st order reflections
        vs.append(VirtualSource(
            position=np.array([x, y, -z]), 
            coeff=self.surface_coeff,  # Phase reversal + small loss
            order=1, 
            label="surface"
        ))

        vs.append(VirtualSource(
            position=np.array([x, y, 2*self.B - z]),
            coeff=self.bottom_coeff,
            order=1,
            label="bottom"
        ))

        # 2nd order (if enabled)  
        if self.max_order >= 2:
            vs.append(VirtualSource(
                position=np.array([x, y, -(2*self.B - z)]),
                coeff=self.surface_coeff * self.bottom_coeff,
                order=2,
                label="surface-bottom"
            ))

            vs.append(VirtualSource(
                position=np.array([x, y, 2*self.B + z]),
                coeff=self.surface_coeff * self.bottom_coeff, 
                order=2,
                label="bottom-surface"
            ))

        return vs

    def H(self, src_pos: np.ndarray, rx_pos: np.ndarray, freq: float) -> complex:
        """Complex channel transfer function H(f) at single frequency."""
        if freq == 0:
            return 0+0j

        H_f = 0+0j
        for vs in self._generate_virtual_sources(src_pos):
            r = vs.distance_to(rx_pos)
            if r < 0.01:  # Avoid singularity for very close sources
                continue

            # Geometric spreading loss + phase delay
            amplitude = vs.coeff / r
            phase = 2 * np.pi * freq * r / self.c

            # Optional: Add frequency-dependent absorption
            # alpha = 0.1 * freq**2 / 1e6  # dB/km at freq in Hz
            # amplitude *= 10**(-alpha * r / 1000 / 20)  # Apply absorption

            H_f += amplitude * np.exp(-1j * phase)

        return H_f

    def H_vec(self, src_pos: np.ndarray, rx_pos: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Vectorized transfer function for frequency array."""
        return np.array([self.H(src_pos, rx_pos, f) for f in freqs])

    def get_channel_info(self) -> Dict:
        """Return channel configuration for logging/debugging."""
        return {
            'sea_depth_m': self.B,
            'array_depth_m': self.b,
            'sound_speed_ms': self.c,
            'bottom_reflection_coeff': self.bottom_coeff,
            'surface_reflection_coeff': self.surface_coeff,
            'max_reflection_order': self.max_order
        }
