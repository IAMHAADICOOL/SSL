"""
Underwater Passive Localization Deployment Script
================================================
Complete real-world deployment interface for three-hydrophone triangular array system.

This script:
1. Interfaces with hydrophone hardware (ADC/audio interface)
2. Continuously acquires synchronized data from 3 hydrophones
3. Pre-processes signals (filtering, buffering)
4. Runs passive localization algorithm
5. Converts results to spatial coordinates
6. Logs results with timestamps

Usage:
    python deploy_localizer.py --config config.yaml

Requirements:
    - Three synchronized hydrophones connected to ADC/audio interface
    - Known hydrophone positions (GPS + depth measurements)
    - Target source frequency range
    - Local environmental parameters (sound speed, water depth)
"""

import numpy as np
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from collections import deque
import argparse
import yaml
import threading
import queue

# Audio/ADC interface options (choose based on your hardware)
try:
    import sounddevice as sd  # For USB audio interfaces
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("Warning: sounddevice not available. Implement custom ADC interface.")

# Import our localization modules
from localizer import PassiveTriangularLocalizer
from array_geometry import create_calibrated_array
from multipath_channel import UnderwaterMultipathChannel, SeafloorType

@dataclass
class DeploymentConfig:
    """Configuration parameters for deployment"""
    # Hardware settings
    # sampling_rate: int = 48000          # ADC sampling rate (Hz)
    # CHANGE TO:
    sampling_rate: int = 200000  # 200 kHz (4.4x Nyquist rate)
    # OR at minimum:
    # sampling_rate: int = 100000  # 100 kHz (2.2x Nyquist rate)
    # buffer_duration: float = 2.0        # Processing window duration (seconds)
    
    # CHANGE TO:  
    buffer_duration: float = 0.5  # seconds (capture at least one 10ms pulse)
    # OR for multiple pulses:
    # buffer_duration: float = 1.2  # seconds (capture 1+ pulses per analysis)
    overlap: float = 0.5               # Buffer overlap fraction

    # Hydrophone array geometry (measured positions in meters [x, y, z])
    hydrophone_positions: Dict[str, list] = None

    # Environmental parameters
    sound_speed: float = 1500.0         # m/s (measure with CTD)
    water_depth: float = 100.0          # m (bathymetry)
    array_depth: float = 50.0           # m (hydrophone deployment depth)
    bottom_type: str = "sand"           # "sand", "mud", "rock", etc.

    # Signal processing
    source_freq_min: float = 100.0      # Hz (your source frequency range)
    source_freq_max: float = 1000.0     # Hz

    # Output settings
    output_file: str = "localization_log.csv"
    log_level: str = "INFO"
    coordinate_system: str = "relative"  # "relative" or "absolute"

    # Hardware interface
    audio_device: Optional[int] = None   # Specific audio device ID
    adc_channels: list = None           # [0, 1, 2] for channels to use

class HydrophoneInterface:
    """Abstract interface for hydrophone hardware"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.fs = config.sampling_rate
        self.buffer_size = int(self.fs * config.buffer_duration)
        self.hop_size = int(self.buffer_size * (1 - config.overlap))

        # Data buffers for three channels
        self.buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.data_queue = queue.Queue()
        self.running = False

    def start_acquisition(self):
        """Start data acquisition thread"""
        self.running = True
        self.acquisition_thread = threading.Thread(target=self._acquisition_loop)
        self.acquisition_thread.start()

    def stop_acquisition(self):
        """Stop data acquisition"""
        self.running = False
        if hasattr(self, 'acquisition_thread'):
            self.acquisition_thread.join()

    def _acquisition_loop(self):
        """Main acquisition loop - override for specific hardware"""
        raise NotImplementedError("Implement for your specific hardware")

    def get_data_chunk(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get next synchronized data chunk from three hydrophones"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

class SoundDeviceInterface(HydrophoneInterface):
    """Interface using sounddevice library for USB audio interfaces"""

    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        if not HAS_SOUNDDEVICE:
            raise ImportError("sounddevice library required for this interface")

        # Configure sounddevice
        sd.default.samplerate = self.fs
        sd.default.device = config.audio_device
        sd.default.channels = 3  # Three hydrophone channels

    def _acquisition_loop(self):
        """Continuous audio acquisition"""
        def callback(indata, frames, time, status):
            if status:
                logging.warning(f"Audio callback status: {status}")

            # Add samples to buffers
            for ch in range(3):
                if ch < indata.shape[1]:
                    self.buffers[ch].extend(indata[:, ch])

            # Check if we have enough data for processing
            if all(len(buf) >= self.buffer_size for buf in self.buffers):
                # Extract synchronized chunks
                chunk = []
                for buf in self.buffers:
                    chunk.append(np.array(list(buf)[-self.buffer_size:]))

                # Queue for processing
                try:
                    self.data_queue.put_nowait(tuple(chunk))
                except queue.Full:
                    logging.warning("Data queue full, dropping samples")

        # Start stream
        with sd.InputStream(callback=callback, channels=3, dtype=np.float32):
            while self.running:
                time.sleep(0.1)

class CustomADCInterface(HydrophoneInterface):
    """Template for custom ADC hardware interface"""

    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        # Initialize your custom ADC hardware here
        # self.adc = YourADCClass(...)

    def _acquisition_loop(self):
        """Implement acquisition for your specific ADC"""
        while self.running:
            # Example structure - replace with your hardware calls
            try:
                # raw_data = self.adc.read_channels([0, 1, 2])  # Read 3 channels
                # Convert to numpy arrays and add to buffers
                # Process and queue data chunks

                # Placeholder - remove this when implementing
                time.sleep(0.1)

            except Exception as e:
                logging.error(f"ADC acquisition error: {e}")
                time.sleep(0.1)

class PassiveLocalizationSystem:
    """Complete passive localization system"""

    def __init__(self, config_file: str = None, config: DeploymentConfig = None):
        # Load configuration
        if config_file:
            self.config = self._load_config(config_file)
        elif config:
            self.config = config
        else:
            raise ValueError("Must provide either config_file or config object")

        # Setup logging
        self._setup_logging()

        # Initialize hardware interface
        self._setup_hardware()

        # Initialize localization algorithm
        self._setup_localizer()

        # Results storage
        self.results_log = []

    def _load_config(self, config_file: str) -> DeploymentConfig:
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return DeploymentConfig(**config_dict)

    def _setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"localization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        logging.info("Passive localization system initializing...")

    def _setup_hardware(self):
        """Initialize hardware interface"""
        if HAS_SOUNDDEVICE and self.config.audio_device is not None:
            self.hardware = SoundDeviceInterface(self.config)
            logging.info("Using SoundDevice interface")
        else:
            self.hardware = CustomADCInterface(self.config)
            logging.info("Using custom ADC interface - ensure you implement _acquisition_loop()")

    def _setup_localizer(self):
        """Initialize the triangular array localizer"""
        # Create calibrated array from measured positions
        if not self.config.hydrophone_positions:
            # Default positions if not provided (REPLACE WITH YOUR MEASUREMENTS)
            logging.warning("Using default hydrophone positions - MEASURE YOUR ACTUAL POSITIONS!")
            positions = {
                "L0": [0.0, 0.2, -self.config.array_depth],    # Top hydrophone
                "L1": [-0.17, -0.1, -self.config.array_depth], # Bottom left
                "L2": [0.17, -0.1, -self.config.array_depth]   # Bottom right
            }
        else:
            positions = self.config.hydrophone_positions

        # Create calibrated array
        self.array = create_calibrated_array(
            positions["L0"], positions["L1"], positions["L2"]
        )

        # Log array statistics
        stats = self.array.get_array_stats()
        logging.info(f"Array configuration: {stats}")

        # Initialize localizer
        self.localizer = PassiveTriangularLocalizer(
            fs=self.config.sampling_rate,
            sound_speed=self.config.sound_speed,
            array=self.array,
            f_band=(self.config.source_freq_min, self.config.source_freq_max)
        )

        logging.info(f"Localizer initialized with {self.config.sound_speed} m/s sound speed")
        logging.info(f"Source frequency band: {self.config.source_freq_min}-{self.config.source_freq_max} Hz")

    def _angles_to_coordinates(self, azimuth_rad: float, pitch_rad: float, 
                             range_m: float = None) -> Dict[str, float]:
        """Convert azimuth/pitch to Cartesian coordinates"""
        if range_m is None:
            # If no range provided, use unit vector (direction only)
            range_m = 1.0

        # Convert spherical to Cartesian (standard spherical coordinate system)
        # Note: pitch is elevation angle from horizontal plane
        x = range_m * np.cos(pitch_rad) * np.cos(azimuth_rad)
        y = range_m * np.cos(pitch_rad) * np.sin(azimuth_rad)
        z = range_m * np.sin(pitch_rad)

        return {
            "x": float(x),
            "y": float(y), 
            "z": float(z),
            "range": float(range_m),
            "azimuth_deg": float(np.degrees(azimuth_rad)),
            "pitch_deg": float(np.degrees(pitch_rad))
        }

    def process_data_chunk(self, sig0: np.ndarray, sig1: np.ndarray, sig2: np.ndarray) -> Dict[str, Any]:
        """Process one chunk of synchronized hydrophone data"""
        timestamp = time.time()

        try:
            # Run localization algorithm
            result = self.localizer.locate(sig0, sig1, sig2)

            # Convert to coordinates (assuming unit range for direction)
            coords = self._angles_to_coordinates(
                result["azimuth_rad"], 
                result["pitch_rad"]
            )

            # Package complete result
            complete_result = {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                **result,
                **coords
            }

            # Log result
            logging.info(
                f"Source detected: Az={coords['azimuth_deg']:+6.1f}°, "
                f"El={coords['pitch_deg']:+6.1f}°, "
                f"τ01={result['tau01']*1e6:+6.1f}μs"
            )

            return complete_result

        except Exception as e:
            logging.error(f"Processing error: {e}")
            return {
                "timestamp": timestamp,
                "error": str(e)
            }

    def save_results(self):
        """Save results to CSV file"""
        if not self.results_log:
            return

        import pandas as pd
        df = pd.DataFrame(self.results_log)
        df.to_csv(self.config.output_file, index=False)
        logging.info(f"Results saved to {self.config.output_file}")

    def run(self, duration: Optional[float] = None):
        """Main execution loop"""
        logging.info("Starting passive localization system...")

        # Start hardware acquisition
        self.hardware.start_acquisition()
        time.sleep(1.0)  # Allow hardware to stabilize

        start_time = time.time()
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break

                # Get data chunk
                chunk = self.hardware.get_data_chunk()
                if chunk is None:
                    time.sleep(0.01)  # No data available
                    continue

                sig0, sig1, sig2 = chunk

                # Process the chunk
                result = self.process_data_chunk(sig0, sig1, sig2)
                self.results_log.append(result)

                # Optional: Real-time coordinate output
                if "error" not in result:
                    print(f"\rSource: Az={result['azimuth_deg']:+6.1f}°, "
                          f"El={result['pitch_deg']:+6.1f}°, "
                          f"X={result['x']:+6.2f}, Y={result['y']:+6.2f}, Z={result['z']:+6.2f}", 
                          end="", flush=True)

        except KeyboardInterrupt:
            logging.info("Stopping on user request...")
        finally:
            # Cleanup
            self.hardware.stop_acquisition()
            self.save_results()
            print("\nShutdown complete.")

def create_example_config():
    """Create example configuration file"""
    config = {
        "sampling_rate": 48000,
        "buffer_duration": 2.0,
        "overlap": 0.5,

        # REPLACE WITH YOUR MEASURED POSITIONS
        "hydrophone_positions": {
            "L0": [0.000, 0.200, -25.0],     # GPS + depth measurements
            "L1": [-0.173, -0.100, -25.1],   # Slight real-world variations
            "L2": [0.174, -0.099, -24.9]
        },

        # MEASURE THESE FOR YOUR SITE
        "sound_speed": 1502.3,              # CTD measurement
        "water_depth": 50.0,                # Bathymetry
        "array_depth": 25.0,                # Deployment depth
        "bottom_type": "sand",

        # SET FOR YOUR TARGET SOURCE
        "source_freq_min": 400.0,           # Your source frequency range
        "source_freq_max": 800.0,

        "output_file": "localization_results.csv",
        "log_level": "INFO",
        "audio_device": None,               # Auto-select audio device
        "adc_channels": [0, 1, 2]
    }

    with open("localization_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("Created example configuration: localization_config.yaml")
    print("IMPORTANT: Update hydrophone positions and environmental parameters!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Underwater Passive Localization System")
    parser.add_argument("--config", default="localization_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--create-config", action="store_true",
                       help="Create example configuration file")
    parser.add_argument("--duration", type=float, default=None,
                       help="Run duration in seconds (None = continuous)")
    parser.add_argument("--test", action="store_true",
                       help="Run with simulated data for testing")

    args = parser.parse_args()

    if args.create_config:
        create_example_config()
    else:
        if args.test:
            # Test mode with simulated data
            print("Running in test mode with simulated data...")
            config = DeploymentConfig(
                hydrophone_positions={
                    "L0": [0.0, 0.2, -25.0],
                    "L1": [-0.173, -0.1, -25.0], 
                    "L2": [0.173, -0.1, -25.0]
                },
                # source_freq_min=400,
                # source_freq_max=800
                # CHANGE TO:
                source_freq_min= 44000.0,   # Hz (around your 45kHz beacon)
                source_freq_max= 46000.0   # Hz (narrow band around beacon)
            )
            # Run with short duration for testing
            system = PassiveLocalizationSystem(config=config)
            # Note: This will use CustomADCInterface which needs implementation
            # for real testing, implement the ADC interface or use SoundDevice
        else:
            # Production mode
            system = PassiveLocalizationSystem(config_file=args.config)

        try:
            system.run(duration=args.duration)
        except Exception as e:
            logging.error(f"System error: {e}")
            raise
