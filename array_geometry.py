"""
Enhanced Triangular Array Geometry with Real-World Calibration  
--------------------------------------------------------------
Supports both theoretical and measured hydrophone positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

__all__ = [
    "TriangularArray", 
    "CalibratedTriangularArray",
    "tdoa_to_angles",
    "create_calibrated_array",
]

@dataclass
class TriangularArray:
    """Theoretical equilateral triangular array."""
    d: float  # side length (m)

    def __post_init__(self):
        h = self.d / np.sqrt(3)
        self.L0 = np.array([0.0,  h, 0.0])          # top vertex
        self.L1 = np.array([-self.d/2, -h/2, 0.0])  # bottom-left  
        self.L2 = np.array([ self.d/2, -h/2, 0.0])  # bottom-right
        self.elements = (self.L0, self.L1, self.L2)

    def get_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.L0, self.L1, self.L2

class CalibratedTriangularArray:
    """Array using actual measured hydrophone positions."""

    def __init__(self, L0_pos: np.ndarray, L1_pos: np.ndarray, L2_pos: np.ndarray):
        self.L0 = np.array(L0_pos)
        self.L1 = np.array(L1_pos)  
        self.L2 = np.array(L2_pos)

        # Calculate actual array characteristics
        self.d01 = np.linalg.norm(self.L1 - self.L0)
        self.d02 = np.linalg.norm(self.L2 - self.L0)
        self.d12 = np.linalg.norm(self.L2 - self.L1)

        # Use average side length for algorithms expecting single 'd' parameter
        self.d = (self.d01 + self.d02 + self.d12) / 3

        # Check if array is reasonably triangular
        sides = [self.d01, self.d02, self.d12]
        if max(sides) / min(sides) > 1.2:
            print(f"Warning: Array is not equilateral. Side ratios: {max(sides)/min(sides):.2f}")

        self.elements = (self.L0, self.L1, self.L2)

    def get_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.L0, self.L1, self.L2

    def get_array_stats(self) -> dict:
        """Return array geometry statistics for validation."""
        return {
            'side_lengths_m': [self.d01, self.d02, self.d12],
            'average_side_m': self.d,
            'side_length_std': np.std([self.d01, self.d02, self.d12]),
            'array_center': (self.L0 + self.L1 + self.L2) / 3
        }

def create_calibrated_array(L0_pos: list, L1_pos: list, L2_pos: list) -> CalibratedTriangularArray:
    """Factory function to create calibrated array from measured positions.

    Args:
        L0_pos, L1_pos, L2_pos: [x, y, z] coordinates in meters

    Returns:
        CalibratedTriangularArray object

    Example:
        # GPS + depth sounder measurements  
        array = create_calibrated_array(
            L0_pos=[0.002, 0.108, -169.85],   # slight measurement errors
            L1_pos=[-0.096, -0.055, -169.87],
            L2_pos=[0.093, -0.056, -169.88]
        )
    """
    return CalibratedTriangularArray(
        np.array(L0_pos), np.array(L1_pos), np.array(L2_pos)
    )

def tdoa_to_angles(tau01: float, tau02: float, tau12: float, 
                   d: float, c: float) -> Tuple[float, float]:
    """Convert TDOAs to azimuth & pitch angles using equations 12-14.

    Returns (azimuth, pitch) in radians with robust error handling.
    """
    Δ01, Δ02, Δ12 = (c * tau01, c * tau02, c * tau12)

    # Sanity check - TDOA magnitudes should be reasonable
    max_expected_delay = d / c  # Maximum possible delay for array size
    if any(abs(delta) > 2 * max_expected_delay for delta in [Δ01, Δ02, Δ12]):
        print(f"Warning: Large TDOAs detected. Check signal quality or array geometry.")

    angles = []

    try:
        # Equation 12 - using τ01 & τ12
        if abs(Δ12) > 1e-12:
            arg1 = (2*Δ01 - Δ12) / (np.sqrt(3) * d)
            arg2 = 2*np.sqrt(3)*Δ12 / (3*d)

            if abs(arg2) <= 1:  # Valid arcsin argument
                phi1 = np.arctan(arg1)
                theta1 = np.arcsin(arg2)
                angles.append((phi1, theta1))

        # Equation 13 - using τ01 & τ02  
        if abs(Δ01 + Δ02) > 1e-12:
            phi2 = np.arctan(np.sqrt(3)*(Δ01 - Δ02)/(Δ01 + Δ02))
            arg2 = np.sqrt(3)*(Δ01 + Δ02)/(3*d)

            if abs(arg2) <= 1:  # Valid arcsin argument
                theta2 = np.arcsin(arg2)
                angles.append((phi2, theta2))

        # Equation 14 - using τ02 & τ12
        if abs(Δ12) > 1e-12:
            arg1 = (2*Δ02 - Δ12) / (np.sqrt(3) * d)  
            arg2 = 2*np.sqrt(3)*Δ12 / (3*d)

            if abs(arg2) <= 1:  # Valid arcsin argument
                phi3 = np.arctan(arg1)
                theta3 = np.arcsin(arg2)
                angles.append((phi3, theta3))

    except (ValueError, ZeroDivisionError) as e:
        print(f"Warning in angle calculation: {e}")
        return 0.0, 0.0

    if not angles:
        print("Warning: No valid angle solutions found")
        return 0.0, 0.0

    # Average valid solutions for robustness
    phi_vals = [a[0] for a in angles if not np.isnan(a[0])]  
    theta_vals = [a[1] for a in angles if not np.isnan(a[1])]

    phi_avg = np.mean(phi_vals) if phi_vals else 0.0
    theta_avg = np.mean(theta_vals) if theta_vals else 0.0

    return float(phi_avg), float(theta_avg)
