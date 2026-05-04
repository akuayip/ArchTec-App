"""
Environment sanity check — run once after installing dependencies.

Usage:
    python scripts/check_env.py
"""

from utils.device import print_device_info


def check_tensorflow() -> None:
    try:
        import tensorflow as tf
        print(f"\nTensorFlow version : {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"TF GPU devices     : {gpus}")
    except ImportError:
        print("\nTensorFlow: not installed (optional — only needed for .h5 models)")


if __name__ == "__main__":
    print("=== PyTorch / Device ===")
    print_device_info()
    check_tensorflow()
