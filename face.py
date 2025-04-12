import cv2
import mediapipe as mp
import numpy as np
from skimage.feature import local_binary_pattern

# Initialize MediaPipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh

# Set up MediaPipe Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,  # Set the maximum number of faces to detect
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Define constants for eye, mouth, and pseudo-depth landmarks
LEFT_EYE = [33, 133, 160, 158, 153, 144]
RIGHT_EYE = [362, 263, 249, 338, 297, 334]
MOUTH = [61, 291, 81, 78, 13, 14, 87, 317]
NOSE_TIP = 1
CHIN = 152

EAR_THRESHOLD = 0.2
SMILE_THRESHOLD = 0.55
LAPLACIAN_TEXTURE_THRESHOLD = 300  
depth_threshold = 0.3
LBP_THRESHOLD = 110

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (A + B) / (2.0 * C)

# Function to calculate smile ratio
def calculate_smile(mouth_points):
    vertical_distance = np.linalg.norm(np.array(mouth_points[3]) - np.array(mouth_points[7]))
    horizontal_distance = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[4]))
    return vertical_distance / horizontal_distance

# Function to estimate head pose (pseudo-depth consistency)
def estimate_head_pose(face_landmarks, frame_shape):
    nose_tip = np.array([face_landmarks[NOSE_TIP].x * frame_shape[1], face_landmarks[NOSE_TIP].y * frame_shape[0]])
    chin = np.array([face_landmarks[CHIN].x * frame_shape[1], face_landmarks[CHIN].y * frame_shape[0]])
    return np.linalg.norm(nose_tip - chin)

# Function to detect texture consistency using Laplacian variance
def detect_texture_laplacian(frame, roi):
    y_min, y_max, x_min, x_max = roi
    y_min, y_max = max(0, y_min - 10), min(frame.shape[0], y_max + 10)
    x_min, x_max = max(0, x_min - 10), min(frame.shape[1], x_max + 10)
    roi_frame = frame[y_min:y_max, x_min:x_max]

    if roi_frame.size == 0:
        return 0
    
    # Optional: Downscale the ROI resolution
    roi_frame = cv2.resize(roi_frame, (roi_frame.shape[1] // 2, roi_frame.shape[0] // 2))

    # Convert to grayscale
    roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur (if necessary)
    roi_frame = cv2.GaussianBlur(roi_frame, (3, 3), 0)

    # Calculate Laplacian
    laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F, ksize=3)

    # Convert Laplacian to absolute value to avoid negative results
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # Calculate the variance and normalize it
    variance = np.var(laplacian_abs) / roi_gray.size

    # Optionally scale the result to fit within the desired range
    scaled_variance = variance * 1000  # Adjust scale factor if necessary

    return scaled_variance

# Function to detect texture using Local Binary Pattern (LBP) with tunable parameters
def detect_texture_lbp(frame, roi, P=32, R=4):
    y_min, y_max, x_min, x_max = roi
    y_min, y_max = max(0, y_min), min(frame.shape[0], y_max)
    x_min, x_max = max(0, x_min), min(frame.shape[1], x_max)
    roi_frame = frame[y_min:y_max, x_min:x_max]
    
    if roi_frame.size == 0:
        return 0

    # Convert to grayscale
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Compute the Local Binary Pattern (LBP) with tunable P and R
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")

    # Calculate the LBP variance as a texture measure
    lbp_var = np.var(lbp)
    return lbp_var

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_index, landmarks in enumerate(results.multi_face_landmarks, start=1):
            face_coords = [
                (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                for landmark in landmarks.landmark
            ]
            left_eye_points = [face_coords[i] for i in LEFT_EYE]
            right_eye_points = [face_coords[i] for i in RIGHT_EYE]
            mouth_points = [face_coords[i] for i in MOUTH]

            ear_left = calculate_ear(left_eye_points)
            ear_right = calculate_ear(right_eye_points)
            smile_ratio = calculate_smile(mouth_points)
            head_depth = estimate_head_pose(landmarks.landmark, frame.shape)

            # ROI for texture detection
            x_min = min(coord[0] for coord in face_coords)
            y_min = min(coord[1] for coord in face_coords)
            x_max = max(coord[0] for coord in face_coords)
            y_max = max(coord[1] for coord in face_coords)
            
            # Detect both texture measures
            texture_variance_laplacian = detect_texture_laplacian(frame, (y_min, y_max, x_min, x_max))

            texture_variance_lbp = detect_texture_lbp(frame, (y_min, y_max, x_min, x_max))
            texture_variance_lbp_8_1 = detect_texture_lbp(frame, (y_min, y_max, x_min, x_max), P=8, R=1)
            texture_variance_lbp_16_2 = detect_texture_lbp(frame, (y_min, y_max, x_min, x_max), P=16, R=2)
            texture_variance_lbp_24_3 = detect_texture_lbp(frame, (y_min, y_max, x_min, x_max), P=24, R=3)
            texture_variance_lbp_24_4 = detect_texture_lbp(frame, (y_min, y_max, x_min, x_max), P=32, R=4)


            # Normalize head depth relative to face size
            face_size = np.linalg.norm([x_max - x_min, y_max - y_min])
            head_depth_normalized = head_depth / face_size

            # # Liveness Check
            # is_real_face = (
            #     (ear_left < EAR_THRESHOLD and ear_right < EAR_THRESHOLD) or
            #     (smile_ratio > SMILE_THRESHOLD)
            # ) and (head_depth > 50) and (texture_variance > TEXTURE_THRESHOLD)


            # Weighted liveness scoring
            score = 0
            if ear_left < EAR_THRESHOLD or ear_right < EAR_THRESHOLD:
                score += 1
            if smile_ratio > SMILE_THRESHOLD:
                score += 1
            if head_depth_normalized > depth_threshold and head_depth > 70:  
                score += 1
            if texture_variance_laplacian > LAPLACIAN_TEXTURE_THRESHOLD:
                score += 1
            if texture_variance_lbp < LBP_THRESHOLD:
                score += 1

            is_real_face = score >= 3 and head_depth > 70 and texture_variance_laplacian < LAPLACIAN_TEXTURE_THRESHOLD and texture_variance_lbp < LBP_THRESHOLD

            print("\n")
            print(f"Face {face_index}:")
            print(f"  EAR Left: {ear_left:.2f}, EAR Right: {ear_right:.2f}")
            print(f"  Smile Ratio: {smile_ratio:.2f}")
            print(f"  Head Depth: {head_depth:.2f}")
            print(f"  Texture Variance (Laplacian): {texture_variance_laplacian:.2f}")
            print(f"  Texture Variance (LBP): {texture_variance_lbp:.2f}")

            # print("\n")
            # print(f"LBP Variance (P=8, R=1): {texture_variance_lbp_8_1:.2f}")
            # print(f"LBP Variance (P=16, R=2): {texture_variance_lbp_16_2:.2f}")
            # print(f"LBP Variance (P=24, R=3): {texture_variance_lbp_24_3:.2f}")
            # print(f"LBP Variance (P=32, R=4): {texture_variance_lbp_24_4:.2f}")


            color = (0, 255, 0) if is_real_face else (0, 0, 255)
            label = f"Face {face_index}: Real" if is_real_face else f"Face {face_index}: Spoof"

            # Draw bounding box and label for each face
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
