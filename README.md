# Liveness Detection using MediaPipe & OpenCV

This Python-based project performs **real-time liveness detection** using webcam feed. It uses:

- **MediaPipe** for face mesh detection
- **OpenCV** for video processing and drawing
- **Skimage (LBP)** for texture analysis
- **Custom heuristics** for detecting eye movement, smile, head pose, and texture variance

---

## 👁️ Features

- Multi-face detection
- Eye Aspect Ratio (EAR) for blink detection
- Smile ratio for facial expression analysis
- Head depth estimation
- Texture analysis using both Laplacian and Local Binary Pattern (LBP)

---

## 🛠️ Installation

Make sure Python 3.7+ is installed.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install opencv-python mediapipe scikit-image numpy
