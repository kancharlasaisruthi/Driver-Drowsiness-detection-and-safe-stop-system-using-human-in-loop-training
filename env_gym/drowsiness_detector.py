"""
drowsiness_detector.py
----------------------
Real-time drowsiness detection using Eye Aspect Ratio (EAR).
Uses OpenCV + dlib facial landmark detector.

Dependencies:
    pip install opencv-python dlib scipy

The 68-point dlib landmark model must be downloaded separately:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Unzip and place shape_predictor_68_face_landmarks.dat in the same folder.
"""

import threading
import time
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist


def shape_to_np(shape: dlib.full_object_detection, dtype: type = "int") -> np.ndarray:
    """Convert dlib shape object to a NumPy array."""
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# ─────────────────────────────────────────────
#  EAR Helper
# ─────────────────────────────────────────────

def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    A low EAR (< threshold) sustained over several frames indicates a closed eye.
    """
    A = dist.euclidean(eye[1], eye[5])  # vertical distance 1
    B = dist.euclidean(eye[2], eye[4])  # vertical distance 2
    C = dist.euclidean(eye[0], eye[3])  # horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear


# ─────────────────────────────────────────────
#  DrowsinessDetector class
# ─────────────────────────────────────────────

class DrowsinessDetector:
    """
    Runs in a background thread.
    Sets self.is_drowsy = True when drowsiness is detected.

    Parameters
    ----------
    ear_threshold : float
        EAR below this value is treated as "eye closed". Default 0.25.
    consec_frames : int
        Number of consecutive frames EAR must be below threshold
        before drowsiness is flagged. Default 20 (~0.67 s at 30 fps).
    camera_index : int
        Which camera to open. Default 0 (built-in webcam).
    landmark_path : str
        Path to dlib's 68-point .dat file.
    show_window : bool
        If True, opens a cv2 debug window showing EAR overlaid on the feed.
    """

    # dlib landmark indices for left/right eye
    _LEFT_EYE_IDX  = slice(42, 48)
    _RIGHT_EYE_IDX = slice(36, 42)

    def __init__(
        self,
        ear_threshold: float = 0.25,
        consec_frames: int   = 20,
        camera_index: int    = 0,
        landmark_path: str   = "shape_predictor_68_face_landmarks.dat",
        show_window: bool    = False,
    ):
        self.ear_threshold  = ear_threshold
        self.consec_frames  = consec_frames
        self.camera_index   = camera_index
        self.landmark_path  = landmark_path
        self.show_window    = show_window

        # Public state — read from the env
        self.is_drowsy: bool  = False
        self.current_ear: float = 1.0   # last measured EAR value

        # Internal
        self._counter  = 0              # consecutive below-threshold frames
        self._running  = False
        self._thread: threading.Thread | None = None

        # Load dlib models
        self._detector  = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(landmark_path)

    # ── Public API ──────────────────────────────

    def start(self):
        """Start the background detection thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[DrowsinessDetector] Started.")

    def stop(self):
        """Stop the background detection thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[DrowsinessDetector] Stopped.")

    def reset(self):
        """Reset drowsiness state (call at each episode reset)."""
        self.is_drowsy = False
        self._counter  = 0

    # ── Internal loop ───────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[DrowsinessDetector] ERROR: Cannot open camera.")
            self._running = False
            return

        failed_reads = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                failed_reads += 1
                if failed_reads > 10:
                    print("[DrowsinessDetector] ERROR: Too many failed frame reads. Stopping detector.")
                    self._running = False
                    break
                time.sleep(0.01)
                continue
            else:
                failed_reads = 0  # Reset on success

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detector(gray, 0)

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detector(gray, 0)

            if len(faces) == 0:
                # No face detected — don't change drowsy state
                if self.show_window:
                    cv2.putText(frame, "No face", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                face  = faces[0]                        # use first detected face
                shape = self._predictor(gray, face)
                shape = shape_to_np(shape)

                left_eye  = shape[self._LEFT_EYE_IDX]
                right_eye = shape[self._RIGHT_EYE_IDX]

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                self.current_ear = ear

                if ear < self.ear_threshold:
                    self._counter += 1
                    if self._counter >= self.consec_frames:
                        self.is_drowsy = True
                else:
                    self._counter  = 0
                    self.is_drowsy = False

                if self.show_window:
                    self._draw_debug(frame, shape, left_eye, right_eye, ear)

            if self.show_window:
                cv2.imshow("Drowsiness Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if self.show_window:
            cv2.destroyAllWindows()

    def _draw_debug(self, frame, shape, left_eye, right_eye, ear):
        """Draw eye contours and EAR value on the debug frame."""
        for eye in (left_eye, right_eye):
            hull = cv2.convexHull(eye)
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

        color  = (0, 0, 255) if self.is_drowsy else (0, 255, 0)
        label  = "DROWSY!" if self.is_drowsy else "Awake"
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


# ─────────────────────────────────────────────
#  Quick standalone test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    detector = DrowsinessDetector(show_window=True)
    detector.start()
    print("Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(0.5)
            if detector.is_drowsy:
                print(f"  EAR={detector.current_ear:.3f}  drowsy={detector.is_drowsy}")
    except KeyboardInterrupt:
        detector.stop()
