import sys
sys.path.append(".")
from src.data.nsynth_dataset import NSynthBass
from torch.utils.data import DataLoader

# point this to your actual data dir once downloaded
# for now just test the class loads without errors
def test_dummy():
    print("NSynthBass class imported successfully")
    print("Waiting for data download to run full test")
    print("Expected output per sample:")
    print("  mel:         (128, time_frames)")
    print("  mel_shifted: (128, time_frames)")
    print("  pitch:       int 0-127")
    print("  shift:       int in pitch_shift_range")

if __name__ == "__main__":
    test_dummy()