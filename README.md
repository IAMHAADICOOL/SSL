# Underwater Passive Localization System (45 kHz Beacon Edition)



**Real-time 3-hydrophone, 3-D direction-finding system** based on
“**Three-Dimensional Passive Localization Method for Underwater Target Using Regular Triangular Array**” (Sun et al., 2019).  
This repository is pre-configured for an **RJE ULB-362B/45 underwater acoustic beacon** (45 kHz, 1 pps, 10 ms pulse).

---

## ✨  Features

| Module | Purpose |
|--------|---------|
| `array_geometry.py` | Handles theoretical **and** calibrated triangular-array geometry (Eqs. 4-14) |
| `signal_processing.py` | Pulse-aware **GCC-Phase** time-delay estimator (Eqs. 15-17) |
| `multipath_channel.py` | Virtual-source multipath model (useful for simulation) |
| `localizer.py` | High-level azimuth/pitch estimator & API wrapper |
| `deploy_localizer.py` | Field-ready deployment script (hardware I/O, logging) |

---

## 🛠️ Hardware Requirements

1. **Hydrophones**: 3 × wideband (>50 kHz) elements.
2. **Array Geometry**: ~4–6 cm side length (≈1–2 λ at 45 kHz).
3. **Synchronized ADC**: ≥200 kHz *simultaneous* sampling on 3 channels.
4. **Anti-alias Filter**: Low-pass ≤60 kHz on each channel.
5. **CTD Probe**: Measure sound speed (**c**) on site.

> **Why 200 kHz?** Nyquist for 45 kHz is 90 kHz. 4× oversampling (≈200 kHz) gives
> sub-microsecond delay resolution and enough FFT bins inside the 44–46 kHz band.

---

## 📦 Installation

```bash
# Clone repository
$ git clone https://github.com/your-org/underwater-beacon-localizer.git
$ cd underwater-beacon-localizer

# Install Python dependencies
$ pip install -r requirements.txt
#  requirements.txt:
#  numpy scipy sounddevice pyyaml pandas
```

---

## ⚙️ Configuration

Create a project config once:

```bash
$ python deploy_localizer.py --create-config
```

Edit **`localization_config.yaml`**:

```yaml
sampling_rate: 200000          # ≥200 kHz
buffer_duration: 1.0           # seconds (captures 1×10 ms pulse)
source_freq_min: 44000.0       # Hz
source_freq_max: 46000.0       # Hz

hydrophone_positions:          # YOUR measured [x,y,z] in metres
  L0: [0.000, 0.040, -25.00]
  L1: [-0.035, -0.020, -25.05]
  L2: [0.035, -0.020, -24.95]

sound_speed: 1502.3            # m/s (CTD)
water_depth: 50.0              # m
array_depth: 25.0              # m
bottom_type: sand              # sand/mud/rock/coral
```

---

## 🚀 Quick Start

```bash
# run continuously, ctrl-c to quit
$ python deploy_localizer.py --config localization_config.yaml
```
Console output:
```
Source: Az=+32.1°, El= +8.4°, X=+0.85, Y=+0.53, Z=+0.15
```
*CSV log* saved as `beacon_localization.csv`.

---

## 🧮  Algorithm Overview

1. **Band-pass & Window** raw signals ⇨ 44–46 kHz.  
2. **Pulse detection** isolates 10 ms burst.  
3. **Cross-power spectrum** (Eq. 15) → phase (Eq. 16).  
4. **Weighted LSQ** fit of phase vs f → τ (Eq. 17).  
5. **Convert TDOAs** (τ₀₁,τ₀₂,τ₁₂) → azimuth φ & pitch θ using Eqs. 12-14.  
6. **Spherical → Cartesian** direction vector.

<p align="center"><img src="docs/flowchart.svg" width="650"></p>

---

## 🗂  Repository Structure

```
.
├── array_geometry.py          # geometry + eqs.12-14
├── signal_processing.py       # GCC-Phase (eqs.15-17) + pulse handling
├── multipath_channel.py       # virtual-source model (optional)
├── localizer.py               # high-level API
├── deploy_localizer.py        # main runtime script
├── requirements.txt           # pip deps
├── localization_config.yaml   # your site config
└── README.md                  # (this file)
```

---

## 📝  Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| **Delay estimate ~0** | Beacon not in band / bad filter | Check `source_freq_min/max`; verify ADC sample rate |
| **Random 180° jumps** | Pulse not detected correctly | Increase `buffer_duration` or tweak `_detect_pulse()` thresholds |
| **Aliasing/artifacts** | Array spacing too large for 45 kHz | Reduce spacing to ≤6 cm |
| **No device found** | `sounddevice` can’t open interface | Set `audio_device` index in config or implement `CustomADCInterface` |

---

## 📖  License

MIT License — see `LICENSE` for details.
