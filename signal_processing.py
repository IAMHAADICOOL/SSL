"""
Cross-Spectrum Phase Time-Delay Estimation (Real-World Version)
--------------------------------------------------------------
Includes pre-filtering, windowing, and weighted least-squares for noisy data.
"""
from __future__ import annotations

import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import butter, sosfilt, windows
from scipy.signal import butter, sosfilt, windows, hilbert, find_peaks, savgol_filter

__all__ = [
    "gcc_phase_delay",
    "_detect_pulse",  # Export pulse detection function
]
def _detect_pulse(sig: np.ndarray, fs: float, min_pulse_duration: float = 0.008) -> tuple:
    """Detect and extract the strongest pulse in the signal.
    
    Args:
        sig: Input signal
        fs: Sampling frequency  
        min_pulse_duration: Minimum expected pulse duration (seconds)
        
    Returns:
        (pulse_start_idx, pulse_end_idx, pulse_present)
    """
    # Calculate envelope using Hilbert transform
    envelope = np.abs(hilbert(sig))
    
    # Smooth envelope to reduce noise
    from scipy.signal import savgol_filter
    if len(envelope) > 51:
        envelope_smooth = savgol_filter(envelope, 51, 3)
    else:
        envelope_smooth = envelope
    
    # Find peaks in envelope
    min_pulse_samples = int(min_pulse_duration * fs)
    peaks, properties = find_peaks(envelope_smooth, 
                                   height=np.max(envelope_smooth) * 0.3,  # 30% of max
                                   distance=min_pulse_samples)
    
    if len(peaks) == 0:
        return 0, len(sig), False  # No pulse found, use full signal
    
    # Use strongest peak
    strongest_peak_idx = peaks[np.argmax(envelope_smooth[peaks])]
    
    # Find pulse boundaries (where envelope drops to 10% of peak)
    peak_level = envelope_smooth[strongest_peak_idx]
    threshold = peak_level * 0.1
    
    # Search backwards for start
    pulse_start = strongest_peak_idx
    for i in range(strongest_peak_idx, max(0, strongest_peak_idx - min_pulse_samples * 2), -1):
        if envelope_smooth[i] < threshold:
            pulse_start = i
            break
    
    # Search forwards for end  
    pulse_end = strongest_peak_idx
    for i in range(strongest_peak_idx, min(len(sig), strongest_peak_idx + min_pulse_samples * 2)):
        if envelope_smooth[i] < threshold:
            pulse_end = i
            break
    
    # Ensure minimum pulse duration
    if pulse_end - pulse_start < min_pulse_samples:
        center = (pulse_start + pulse_end) // 2
        pulse_start = max(0, center - min_pulse_samples // 2)
        pulse_end = min(len(sig), center + min_pulse_samples // 2)
    
    return pulse_start, pulse_end, True

def _bandpass(sig: np.ndarray, fs: float, fmin: float, fmax: float) -> np.ndarray:
    """4th-order Butterworth band-pass filter"""
    sos = butter(4, [fmin, fmax], btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos, sig)

def gcc_phase_delay(sig_a: np.ndarray,
                    sig_b: np.ndarray,
                    fs: float,
                    fmin: float = 44000.0,
                    fmax: float = 46000.0) -> float:
    """Enhanced GCC-Phase for pulsed beacon signals."""
    
    # Step 1: Detect pulses in both signals
    start_a, end_a, pulse_a_found = _detect_pulse(sig_a, fs)
    start_b, end_b, pulse_b_found = _detect_pulse(sig_b, fs)
    
    if not (pulse_a_found and pulse_b_found):
        # Fall back to full signal if no pulses detected
        print("Warning: No clear pulse detected, using full signal")
        pulse_a = sig_a
        pulse_b = sig_b
    else:
        # Use overlapping window around both detected pulses
        overall_start = min(start_a, start_b)
        overall_end = max(end_a, end_b)
        
        # Ensure we have enough samples for FFT
        min_samples = int(0.02 * fs)  # 20ms minimum
        if overall_end - overall_start < min_samples:
            center = (overall_start + overall_end) // 2
            overall_start = max(0, center - min_samples // 2)
            overall_end = min(len(sig_a), center + min_samples // 2)
        
        pulse_a = sig_a[overall_start:overall_end]
        pulse_b = sig_b[overall_start:overall_end]
        
        print(f"Using pulse window: {overall_start} to {overall_end} "
              f"({len(pulse_a)} samples, {len(pulse_a)/fs*1000:.1f}ms)")
    
    # Step 2: Remove DC and band-pass filter the pulse segments
    try:
        pulse_a_filt = _bandpass(pulse_a - np.mean(pulse_a), fs, fmin, fmax)
        pulse_b_filt = _bandpass(pulse_b - np.mean(pulse_b), fs, fmin, fmax)
    except Exception as e:
        print(f"Warning: Bandpass filter failed: {e}, using unfiltered signals")
        pulse_a_filt = pulse_a - np.mean(pulse_a)
        pulse_b_filt = pulse_b - np.mean(pulse_b)
    
    # Step 3: Apply windowing (Tukey window is better for short pulses)
    if len(pulse_a_filt) > 10:
        window = windows.tukey(len(pulse_a_filt), alpha=0.25)  # Gentler than Hann
        pulse_a_filt *= window
        pulse_b_filt *= window
    
    # Step 4: Zero-pad for better frequency resolution
    N_orig = len(pulse_a_filt)
    N_padded = max(N_orig * 4, 1024)  # At least 4x padding
    
    pulse_a_padded = np.zeros(N_padded)
    pulse_b_padded = np.zeros(N_padded)
    pulse_a_padded[:N_orig] = pulse_a_filt
    pulse_b_padded[:N_orig] = pulse_b_filt
    
    # Step 5: FFT and cross-spectrum
    freqs = fftfreq(N_padded, 1/fs)
    X1 = fft(pulse_a_padded)
    X2 = fft(pulse_b_padded)
    CPS = X1 * np.conj(X2)
    phase = np.angle(CPS)
    
    # Step 6: Select frequency band
    idx = (freqs > 0) & (freqs >= fmin) & (freqs <= fmax)
    f_sel = freqs[idx]
    phi_sel = np.unwrap(phase[idx])
    weights = np.abs(CPS[idx])
    
    # Check if we have enough frequency points
    if len(f_sel) < 10:
        print(f"Warning: Only {len(f_sel)} frequency points in band {fmin}-{fmax} Hz")
        # Expand frequency band slightly
        fmin_expanded = max(0, fmin - 1000)
        fmax_expanded = min(fs/2, fmax + 1000)
        idx = (freqs > 0) & (freqs >= fmin_expanded) & (freqs <= fmax_expanded)
        f_sel = freqs[idx]
        phi_sel = np.unwrap(phase[idx])
        weights = np.abs(CPS[idx])
    
    # Step 7: Weighted least-squares with outlier rejection
    if len(f_sel) > 5:
        # Remove outliers (phase points that deviate too much from linear trend)
        initial_fit = np.polyfit(f_sel, phi_sel, 1)
        residuals = phi_sel - np.polyval(initial_fit, f_sel)
        std_residual = np.std(residuals)
        outlier_mask = np.abs(residuals) < 3 * std_residual
        
        if np.sum(outlier_mask) > 3:
            f_sel = f_sel[outlier_mask]
            phi_sel = phi_sel[outlier_mask]  
            weights = weights[outlier_mask]
    
    # Normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights) * len(weights)
    else:
        weights = np.ones_like(weights)
    
    # Final weighted least-squares
    try:
        A = np.vstack([-2*np.pi*f_sel, np.ones_like(f_sel)]).T * weights[:, None]
        b = phi_sel * weights
        slope, intercept = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Sanity check on delay estimate  
        max_expected_delay = 0.001  # 1ms max for reasonable array sizes
        if abs(slope) > max_expected_delay:
            print(f"Warning: Large delay estimate {slope*1e6:.1f}μs, may be unreliable")
            
        return float(slope)
        
    except Exception as e:
        print(f"Error in least-squares fit: {e}")
        return 0.0
