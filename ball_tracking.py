# ============================================================
# DotLumen Challenge: Ball Tracking
# Author: Andrada Terteci-Popescu
# ============================================================
#
# OpenCV: video I/O, drawing, feature tracking, homographies and perspective transforms.
import cv2
# YOLO provides the object detector used to locate the sports ball in each RGB frame.
from ultralytics import YOLO
import math
import os
# NumPy provides vectors, matrices, norms, least-squares algebra and array operations.
import numpy as np
# deque stores only the most recent samples required by the smoothing/prediction windows.
from collections import deque
from openpyxl import Workbook
from openpyxl.styles import Font
import matplotlib.pyplot as plt

try:
# scikit-learn builds the online polynomial Ridge-regression trajectory predictor.
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
except ImportError as exc:
    raise ImportError(
        "This script requires scikit-learn for ML trajectory prediction. "
        "Install it with: pip install scikit-learn"
    ) from exc


# ============================================================
# RGB BALL + CAMERA GROUND TRACKING
# Intel RealSense D435i RGB video
# ============================================================
#
# OUTPUTS
# ------------------------------------------------------------
# 1. rgb_out_ground_tracking.avi
#
#    Annotated original RGB video containing:
#       - ball detection
#       - approximate ball ground-contact point
#       - RED ground-fixed measured ball trajectory
#
#    Each ball point is stored in the fixed ground coordinate
#    system and then reprojected into the current camera image.
#
#    The trajectory is therefore intended to remain fixed
#    relative to the ground as the camera moves.
#
#
# 2. ball_camera_ground_map.avi
#
#       RED     = measured ball ground trajectory
#       MAGENTA = ML/fallback predicted future ground trajectory
#       BLUE    = reconstructed camera ground trajectory
#
#
# 3. Excel trajectory data, including per-frame camera-relative
#    position and velocity, ground position and velocity,
#    prediction method/fit, prediction endpoint, camera position,
#    and homography information
#
#
# LIMITATION
# ----------
# No physical terrace dimensions are known.
#
# Ground coordinates therefore have arbitrary global scale.
#
# ============================================================

# ============================================================
# FILES
# ============================================================

# Input/output filenames. Only the RGB video is required as input.
INPUT_VIDEO = "rgb.avi"
OUTPUT_VIDEO = "rgb_out_ground_tracking.avi"
OUTPUT_TOP_VIEW = "ball_camera_ground_map.avi"
OUTPUT_EXCEL = "ball_camera_ground_tracking.xlsx"


# ============================================================
# YOLO
# ============================================================

# YOLO model, COCO sports-ball class ID, and minimum accepted detection confidence.
MODEL = "yolo11n.pt"
SPORTS_BALL_CLASS = 32
CONFIDENCE = 0.20

# ============================================================
# FOOTBALL SIZE
# ============================================================

# Known physical ball diameter; used with apparent pixel size for monocular depth.
BALL_DIAMETER_M = 0.220

# ============================================================
# SMOOTHING
# ============================================================

# Median-filter window lengths. Larger windows suppress noise but increase temporal lag.
BALL_SMOOTH_WINDOW = 5
CAMERA_SMOOTH_WINDOW = 5

# Velocity is calculated from consecutive smoothed 3-D positions.
# A short median window suppresses frame-to-frame depth noise.
VELOCITY_SMOOTH_WINDOW = 5

# Ground velocity is still estimated for velocity reporting and as a
# robust fallback predictor.
# Number of recent ground samples used to estimate the fallback ground velocity.
GROUND_VELOCITY_WINDOW = 10
MIN_PREDICTION_POINTS = 4

# ------------------------------------------------------------
# MACHINE-LEARNING TRAJECTORY PREDICTION
# ------------------------------------------------------------
# The ML predictor is trained online on the most recent FIXED-GROUND
# ball positions. It learns X(t) and Z(t) jointly using polynomial
# features plus Ridge regression. Quadratic features allow short-term
# curvature while Ridge regularization reduces sensitivity to noise.
# ML settings: recent training window, minimum samples, polynomial degree and Ridge strength.
ML_PREDICTION_WINDOW = 20
MIN_ML_PREDICTION_POINTS = 8
ML_POLYNOMIAL_DEGREE = 2
ML_RIDGE_ALPHA = 0.05

# Future time span and number of samples drawn along every predicted trajectory.
PREDICTION_HORIZON_S = 1.0
PREDICTION_STEPS = 20

# ============================================================
# CAMERA MOTION ESTIMATION
# ============================================================

# Parameters for Shi-Tomasi feature detection and pyramidal Lucas-Kanade optical flow.
MAX_FEATURES = 1500
FEATURE_QUALITY = 0.01
FEATURE_MIN_DISTANCE = 10
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
# RANSAC/inlier thresholds reject unreliable frame-to-frame homographies.
RANSAC_REPROJECTION_THRESHOLD = 3.0
MIN_HOMOGRAPHY_INLIERS = 25
CAMERA_FEATURE_MASK_MARGIN = 40

# ============================================================
# HOMOGRAPHY SANITY CHECKS
# ============================================================

# Homography sanity limits reject implausible scale changes or large image translations.
MAX_FRAME_TRANSLATION_PX = 150.0
MIN_HOMOGRAPHY_DETERMINANT = 0.20
MAX_HOMOGRAPHY_DETERMINANT = 5.00

# ============================================================
# TRAJECTORY DRAWING
# ============================================================

RGB_TRAJECTORY_THICKNESS = 4
PREDICTION_TRAJECTORY_THICKNESS = 3

# Only connect consecutive trajectory points if the projected
# points are not absurdly far apart in the image.
#
# This helps suppress visual explosions after a bad homography.

MAX_DRAW_SEGMENT_PX = 250

# ============================================================
# VIDEO
# ============================================================

# FOURCC codec used by OpenCV when writing the AVI output videos.
VIDEO_CODEC = "XVID"
BOX_THICKNESS = 2

# ============================================================
# TOP VIEW
# ============================================================

# Top-view canvas geometry and extra padding around all trajectories.
MAP_WIDTH = 1100
MAP_HEIGHT = 850
MAP_MARGIN = 70
MAP_PADDING_FRACTION = 0.12

# ============================================================
# CAMERA INTRINSICS
# ============================================================

# Read the video size and derive approximate pinhole-camera intrinsics from the nominal FOV.
# Returns focal lengths (fx, fy) and principal point (cx, cy), all in pixels.
def get_intrinsics():
    video_in = cv2.VideoCapture(INPUT_VIDEO)

    W = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_in.get(cv2.CAP_PROP_FPS)

    video_in.release()

    print("Resolution:", W, "x", H)
    print("FPS:", fps)

    # Nominal D435i RGB field of view
    hfov_deg = 69.4
    vfov_deg = 42.5

    # Convert degrees to radians because NumPy trigonometric functions expect radians.
    hfov = np.deg2rad(hfov_deg)
    vfov = np.deg2rad(vfov_deg)

    # Pinhole geometry: tan(FOV/2) = (image_size/2) / focal_length.
    # Rearranging gives f = image_size / (2*tan(FOV/2)).
    fx = W / (2.0 * np.tan(hfov / 2.0))
    fy = H / (2.0 * np.tan(vfov / 2.0))

    # Assume the optical principal point is at the image centre.
    cx = W / 2.0
    cy = H / 2.0

    return fx, fy, cx, cy

# ============================================================
# BALL GEOMETRY
# ============================================================

# Approximate the ball's image diameter from the YOLO bounding box.
# Averaging box width and height reduces sensitivity to a slightly non-square box.
def ball_diameter_from_box(box):
    x1, y1, x2, y2 = box

    # Clamp each dimension to at least 1 px to avoid zero-size geometry.
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))

    return 0.5 * (w + h)


# Approximate the point where the ball meets/projects onto the ground.
# The bottom-centre of the bounding box is used instead of the box centre.
def ball_ground_contact_from_box(box):
    """
    Approximate the ball's ground-contact / ground-projection point.

    For a ball on the floor, bottom-center of the detection box
    is more appropriate than the center of the bounding box.
    """

    x1, y1, x2, y2 = box

    # Horizontal midpoint of the box; y2 is its bottom edge.
    gx = float(0.5 * (x1 + x2))
    gy = float(y2)

    return (gx,gy)


# Back-project a detected ball from image coordinates into camera-relative 3D.
# Uses the known ball diameter to resolve monocular scale approximately.
def pixel_to_camera(u, v, diameter_px, fx, fy, cx, cy):
    """
    Approximate camera-relative 3-D position.

    u, v   = detected ball-center pixel coordinates
    cx, cy = camera principal point
    fx, fy = focal lengths in pixels

    +X = camera right
    +Y = camera down
    +Z = camera forward
    """

    diameter_px = max(1.0, float(diameter_px))

    # Estimate depth from the known physical ball diameter
    z = fx * BALL_DIAMETER_M / diameter_px

    # Inverse pinhole projection
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy

    return np.array([x, y, z])


# ============================================================
# SMOOTHING
# ============================================================

# Apply a component-wise median filter to the most recent vectors in a history buffer.
# Median filtering is robust to occasional detection/depth outliers.
def median_smooth(history, window):

    if len(history) == 0:
        return None

    # Slice only the newest 'window' samples; older values do not influence the result.
    array = np.asarray(list(history)[-window:])

    # axis=0 computes one median for each vector component.
    return np.median(array, axis=0)


# ============================================================
# VELOCITY + TRAJECTORY PREDICTION
# ============================================================

# Estimate instantaneous velocity by a first-order finite difference between two positions.
def finite_difference_velocity(previous_position, previous_time, current_position, current_time):
    """
    Estimate velocity from two measured positions.

    v = (p_k - p_(k-1)) / (t_k - t_(k-1))

    The position can be any NumPy vector. The returned units are
    the position units per second.
    """

    if (previous_position is None or previous_time is None):
        return None

    # Delta time between the current and previous valid position measurements.
    dt = float(current_time - previous_time)

    # Reject zero/near-zero time intervals to avoid numerical division problems.
    if dt <= 1e-9:
        return None

    # v = Δp / Δt, evaluated component-wise.
    return (np.asarray(current_position) - np.asarray(previous_position)) / dt


# Fit a straight line to recent fixed-ground positions to estimate smoothed ground velocity.
# X(t) and Z(t) are fitted simultaneously with ordinary least squares.
def estimate_ground_velocity(timed_ground_history, window=GROUND_VELOCITY_WINDOW):
    """
    Estimate camera-motion-compensated ground velocity with a least-squares line fit:

        X(t) = X0 + Vx * t
        Z(t) = Z0 + Vz * t

    Because the history is stored in the FIXED ground coordinate system, camera image motion does not directly enter this velocity.

    Returns:
        velocity = [Vx, Vz] in ground-units / second
        fit_rmse = RMSE residual in ground units
    """

    samples = list(timed_ground_history)[-window:]

    if len(samples) < 2:
        return None, None

    # Separate the sample times and 2D ground targets [X, Z].
    times = np.asarray([sample[0] for sample in samples])
    points = np.asarray([sample[1] for sample in samples])

    # Shift the newest sample to t=0. The fit is unchanged but numerically better conditioned.
    # Shift the time origin for numerical conditioning.
    times = times - times[-1]

    if np.ptp(times) <= 1e-9: #peak-to-peak
        return None, None

    # Design matrix for p(t)=v*t+p0: first column is time, second is the intercept term.
    A = np.column_stack((times, np.ones_like(times)))

    try:
        # Least-squares solves A @ coefficients ≈ points.
        # Row 0 of 'coefficients' is the velocity vector [Vx, Vz].
        coefficients, _, _, _ = np.linalg.lstsq(A, points, rcond=None) #least-squares solution
    except np.linalg.LinAlgError:
        return None, None

    # slope of the fitted line = ground velocity.
    velocity = coefficients[0]
    # fitted positions = A x fitted coefficients.
    fitted = A @ coefficients
    # residual = measured position - fitted position.
    residual = points - fitted

    # 2D RMSE: Euclidean residual magnitude is squared, averaged, then square-rooted.
    fit_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

    if not np.isfinite(velocity).all():
        return None, None

    return (np.asarray(velocity), fit_rmse)

# Deterministic fallback predictor used before the ML model has enough data or if ML fails.
def predict_ground_trajectory_constant_velocity(
    current_ground_point,
    ground_velocity,
    horizon_s=PREDICTION_HORIZON_S,
    steps=PREDICTION_STEPS
):
    """
    Fallback predictor using constant ground velocity.

        p(t + tau) = p(t) + v_ground * tau

    This is used before the ML model has enough samples or if the ML fit fails.
    """

    if (current_ground_point is None or ground_velocity is None or steps < 1 or horizon_s <= 0.0):
        return []

    p0 = np.asarray(current_ground_point)

    v = np.asarray(ground_velocity)

    if (not np.isfinite(p0).all() or not np.isfinite(v).all()):
        return []

    # Start the predicted polyline exactly at the current measured ground point.
    prediction = [(float(p0[0]), float(p0[1]))]

    for step in range(1, steps + 1):
        # Prediction time τ progresses uniformly from 0 to the requested horizon.
        tau = (horizon_s * step / steps)
        # Constant-velocity motion model: p(t+τ) = p(t) + v*τ.
        p = p0 + v * tau

        prediction.append((float(p[0]), float(p[1])))

    return prediction


# Train a small polynomial Ridge-regression model online from recent fixed-ground samples.
# Input: relative time. Targets: ground coordinates [X, Z]. Output: future ground path.
def predict_ground_trajectory_ml(
    timed_ground_history,
    current_ground_point,
    horizon_s=PREDICTION_HORIZON_S,
    steps=PREDICTION_STEPS,
    window=ML_PREDICTION_WINDOW,
    degree=ML_POLYNOMIAL_DEGREE,
    ridge_alpha=ML_RIDGE_ALPHA
):
    """
    Online machine-learning trajectory prediction in the FIXED ground coordinate system.

    Training data:
        input  = relative time t
        target = [X_ground, Z_ground]

    Model:
        PolynomialFeatures(degree=2) + Ridge regression

    For degree 2, the learned trajectory has the form

        X(t) = a0 + a1*t + a2*t^2
        Z(t) = b0 + b1*t + b2*t^2

    but the coefficients are learned from the recent measured trajectory
    rather than fixed manually. Ridge regularization stabilizes the fit in
    the presence of noisy detections and homography jitter.

    Returns:
        prediction : list of (X, Z) future points
        fit_rmse   : training-fit RMS error in ground units
        n_samples  : number of trajectory samples used to train the model
    """

    # Restrict training to the most recent samples so the model follows current motion.
    samples = list(timed_ground_history)[-window:]
    n_samples = len(samples)

    if (n_samples < MIN_ML_PREDICTION_POINTS or current_ground_point is None or steps < 1 or horizon_s <= 0.0):
        return [], None, n_samples

    # Build the 1-D time input and 2-D target matrix [X, Z].
    times = np.asarray([sample[0] for sample in samples],)
    points = np.asarray([sample[1] for sample in samples])

    if (not np.isfinite(times).all() or not np.isfinite(points).all()):
        return [], None, n_samples

    # Make the latest observation t=0; past samples are negative and predictions use t>0.
    # Put the current frame at t = 0. This keeps the numerical values small
    # and makes future prediction simply t > 0.
    relative_times = (times - times[-1]).reshape(-1, 1)

    # A degree-d polynomial requires at least d+1 distinct time values.
    # A polynomial of degree d needs at least d+1 distinct time values.
    if (np.unique(relative_times).size < degree + 1):
        return [], None, n_samples

    # Pipeline: [t] -> polynomial features [t, t², ...] -> Ridge regression.
    # Ridge adds L2 regularization, discouraging excessively large coefficients.
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=ridge_alpha))

    try:
        # Fit one multi-output regression: the same time features predict X and Z.
        model.fit(relative_times, points)
        fitted_points = model.predict(relative_times)

        # Training residuals quantify how closely the polynomial follows recent measurements.
        residual = (points - fitted_points)

        # RMSE summarizes the 2-D fitting error in ground-coordinate units.
        fit_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        # Uniform future times from the current instant (0) to the prediction horizon.
        future_times = np.linspace(0.0, horizon_s, steps + 1).reshape(-1, 1)

        # Extrapolate the learned polynomial to those future times.
        predicted_points = model.predict(future_times)

    except Exception:
        return [], None, n_samples

    if not np.isfinite(predicted_points).all():
        return [], None, n_samples

    # Make the predicted curve start exactly from the latest measured
    # ground point. This removes a small visual jump caused by regression
    # smoothing while preserving the learned future curvature.
    current_ground_point = np.asarray(current_ground_point)

    if not np.isfinite(current_ground_point).all():
        return [], None, n_samples

    # Regression can slightly miss the latest point; translate the whole predicted curve
    # so its first point exactly equals the latest measured ground position.
    correction = (current_ground_point - predicted_points[0])
    predicted_points = (predicted_points + correction)

    prediction = [(float(point[0]), float(point[1])) for point in predicted_points]

    return prediction, fit_rmse, n_samples

# ============================================================
# CAMERA MOTION
# ============================================================

# Estimate camera motion from static background features between consecutive grayscale frames.
# The returned homography maps CURRENT-frame pixels back to PREVIOUS-frame pixels.
def estimate_camera_homography(prev_gray, gray, previous_ball_box):
    """
    Estimate frame-to-frame background homography.
    Returned H maps:
        CURRENT FRAME -> PREVIOUS FRAME
    """

    # 255 means features may be detected; the ball region will be painted 0 (excluded).
    mask = np.full(prev_gray.shape, 255, dtype=np.uint8)

    # --------------------------------------------------------
    # Remove ball region from camera-motion features
    # --------------------------------------------------------

    # Exclude an expanded region around the previous ball so ball motion is not treated as camera motion.
    if previous_ball_box is not None:

        x1, y1, x2, y2 = (previous_ball_box)

        x1 -= CAMERA_FEATURE_MASK_MARGIN
        y1 -= CAMERA_FEATURE_MASK_MARGIN

        x2 += CAMERA_FEATURE_MASK_MARGIN
        y2 += CAMERA_FEATURE_MASK_MARGIN

        cv2.rectangle(mask,
                      (max(0, int(x1)), max(0, int(y1))),
                      (min(mask.shape[1] - 1, int(x2)),
                      min(mask.shape[0] - 1, int(y2))), 0, -1)

    # --------------------------------------------------------
    # Background features
    # --------------------------------------------------------

    # Shi-Tomasi-style corner detection chooses trackable background features.
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=MAX_FEATURES,
        qualityLevel=FEATURE_QUALITY,
        minDistance=FEATURE_MIN_DISTANCE,
        mask=mask,
        blockSize=7)

    if p0 is None:
        return None, 0

    if len(p0) < 40:
        return None, 0

    # --------------------------------------------------------
    # Optical flow
    # --------------------------------------------------------

    # Pyramidal Lucas-Kanade optical flow tracks each previous feature into the current frame.
    p1, status, error = (
        cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            p0,
            None,
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL))

    if p1 is None or status is None:
        return None, 0

    # Keep only features for which optical flow reports successful tracking.
    good = (status.ravel() == 1)

    if error is not None:
        # Also reject tracks with large Lucas-Kanade error.
        good &= (error.ravel() < 20.0)

    # Reshape accepted feature coordinates into N×2 point arrays.
    previous_points = (p0[good].reshape(-1, 2))
    current_points = (p1[good].reshape(-1, 2))

    if len(previous_points) < 40:
        return None, 0

    # --------------------------------------------------------
    # CURRENT -> PREVIOUS
    # --------------------------------------------------------

    # Robustly fit CURRENT -> PREVIOUS homography with RANSAC.
    # RANSAC ignores feature correspondences that do not agree with the dominant camera motion.
    H, inlier_mask = cv2.findHomography(
        current_points,
        previous_points,

        cv2.RANSAC,

        RANSAC_REPROJECTION_THRESHOLD)

    if H is None or inlier_mask is None:
        return None, 0

    # Number of RANSAC correspondences supporting the estimated homography.
    inliers = int(inlier_mask.sum())

    if inliers < MIN_HOMOGRAPHY_INLIERS:
        return None, inliers

    # A projective homography is defined up to scale; H[2,2] must be usable for normalization.
    if abs(H[2, 2]) < 1e-12:
        return None, inliers

    # Normalize so H[2,2]=1. This changes only homogeneous scale, not the mapping.
    H = (H / H[2, 2])

    # Determinant of the 2×2 upper-left block is used as a coarse scale/orientation sanity check.
    determinant = np.linalg.det(H[:2, :2])

    if not np.isfinite(determinant):
        return None, inliers

    if not (MIN_HOMOGRAPHY_DETERMINANT <= abs(determinant) <= MAX_HOMOGRAPHY_DETERMINANT):
        return None, inliers

    # Magnitude of the homography's image-plane translation components.
    translation = math.hypot(H[0, 2], H[1, 2])

    if translation > MAX_FRAME_TRANSLATION_PX:
        return None, inliers

    return H, inliers


# ============================================================
# HOMOGRAPHY UTILITIES
# ============================================================

# Lightweight validation before a homography is accumulated into the camera-motion chain.
def homography_valid(H):

    if H is None:
        return False

    # Reject NaN or infinite entries.
    if not np.isfinite(H).all():
        return False

    # Reject matrices that cannot be safely normalized by H[2,2].
    if abs(H[2, 2]) < 1e-12:
        return False

    return True


# Transform one 2-D point with a 3×3 homography using OpenCV's homogeneous projection.
def transform_point(H, point):
    """
    Apply a homography to one 2D point.
    """

    # OpenCV perspectiveTransform expects shape (N,1,2), hence the nested brackets.
    p = np.array([[[float(point[0]), float(point[1])]]])
    # Internally: q_h ~ H p_h, followed by division by the homogeneous third coordinate.
    q = cv2.perspectiveTransform(p, H)[0, 0]

    return np.array([float(q[0]), float(q[1])])

# ============================================================
# INITIAL GROUND COORDINATE SYSTEM
# ============================================================

# Define the reference image-to-ground mapping from four manually chosen ground-plane points.
# The resulting top-view coordinates are consistent but have arbitrary physical scale.
def create_initial_ground_coordinate_system(width, height):
    """
    Establish a fixed coordinate system on the visible ground.

    No terrace dimension is known, therefore these are arbitrary ground units, NOT meters.
    """

    # --------------------------------------------------------
    # Visible ground rectangle in reference frame.
    #
    # Defined for 1280 x 720, then scaled automatically.
    # --------------------------------------------------------

    # Source quadrilateral in a 1280×720 reference frame: far-left, far-right, near-right, near-left.
    # These points are manually chosen to delimit a visible planar ground region.
    base_src = np.array(
        [
            [180.0, 430.0],
            [1080.0, 430.0],
            [1240.0, 710.0],
            [40.0, 710.0]
        ],
        dtype=np.float32
    )

    # Scale the manually chosen reference pixels if the actual video resolution differs.
    sx = width / 1280.0
    sy = height / 720.0

    src = base_src.copy()

    # Scale x coordinates by sx and y coordinates by sy.
    src[:, 0] *= sx
    src[:, 1] *= sy

    # --------------------------------------------------------
    # Arbitrary fixed ground coordinates
    # --------------------------------------------------------

    # Destination rectangle in user-defined ground coordinates [X,Z].
    # Values are not metres because no measured terrace dimensions are available.
    dst = np.array(
        [
            [-3.0, 8.0],
            [7.0, 8.0],
            [7.0, 0.0],
            [-3.0, 0.0]
        ],
        dtype=np.float32
    )

    # Solve the four-point projective mapping H_reference->ground.
    return cv2.getPerspectiveTransform(src, dst)

# ============================================================
# CURRENT IMAGE --> FIXED GROUND
# ============================================================

# Map one current image point into the fixed ground coordinate system with the current homography.
def current_image_to_ground(image_point, H_current_to_ground):
    return transform_point(H_current_to_ground, image_point)

# ============================================================
# FIXED GROUND --> CURRENT IMAGE
# ============================================================

# Reproject a stored fixed-ground point back into the current camera image for drawing.
def ground_to_current_image(ground_point, H_current_to_ground):
    """
    Reproject a FIXED ground coordinate into the current RGB frame.

    This is the key operation used to draw the realistic ground-fixed football trajectory on rgb_out_ground_tracking.avi.

    Because the stored trajectory lives on the ground rather than in
    screen pixels, the trajectory follows the terrace when the camera moves.
    """

    try:
        # Invert image->ground to obtain ground->current-image.
        # A singular matrix has no inverse, so the function safely returns None.
        H_ground_to_current = np.linalg.inv(H_current_to_ground)
    except np.linalg.LinAlgError:
        return None

    if abs(H_ground_to_current[2, 2]) < 1e-12:
        return None

    # Normalize homogeneous scale before applying the inverse homography.
    H_ground_to_current /= H_ground_to_current[2, 2]
    image_point = transform_point(H_ground_to_current, ground_point)

    if not np.isfinite(image_point).all():
        return None

    return image_point

# ============================================================
# DRAW GROUND-FIXED TRAJECTORY ON RGB FRAME
# ============================================================

# Draw the measured ground-fixed ball history after reprojecting every stored point into this frame.
def draw_ground_trajectory_on_frame(frame, ground_trajectory, H_current_to_ground):
    """
    Draw the ball's historical FIXED-GROUND trajectory in the CURRENT camera image.

    This is fundamentally different from drawing historical YOLO pixel centers.

    Historical ground points:
         ground P0, P1, P2 ...

    are transformed:
         FIXED GROUND -> CURRENT CAMERA IMAGE

    on every frame.

    Therefore a stationary trajectory on the terrace remains attached to the terrace while the camera moves.
    """

    if len(ground_trajectory) < 2:
        return frame

    # Current image dimensions are used only for visibility checks.
    height, width = frame.shape[:2]
    previous_visible = None

    for ground_point in ground_trajectory:
        image_point = (ground_to_current_image(ground_point, H_current_to_ground))

        if image_point is None:
            previous_visible = None
            continue

        x = float(image_point[0])
        y = float(image_point[1])

        # ----------------------------------------------------
        # Only draw visible / near-visible projected points.
        # ----------------------------------------------------

        # Allow a small off-screen margin so lines do not disappear abruptly at the border.
        margin = 100

        visible = (
            -margin <= x < width + margin
            and
            -margin <= y < height + margin
        )

        if not visible:
            previous_visible = None
            continue

        current = (int(round(x)), int(round(y)))

        if previous_visible is not None:

            # Euclidean pixel distance between consecutive projected trajectory points.
            segment_length = math.hypot(current[0] - previous_visible[0],
                                        current[1] - previous_visible[1])

            if (segment_length <= MAX_DRAW_SEGMENT_PX):
                cv2.line(
                    frame,
                    previous_visible,
                    current,
                    (0, 0, 255),
                    RGB_TRAJECTORY_THICKNESS,
                    cv2.LINE_AA)

        previous_visible = current

    return frame

# ============================================================
# DRAW PREDICTED GROUND TRAJECTORY ON RGB FRAME
# ============================================================

# Draw the predicted fixed-ground path in the current RGB frame as a dashed magenta polyline.
def draw_ground_prediction_on_frame(
    frame,
    predicted_ground_trajectory,
    H_current_to_ground
):
    """
    Reproject the future fixed-ground prediction into the current RGB frame. The prediction is drawn as a dashed MAGENTA line.
    """

    if len(predicted_ground_trajectory) < 2:
        return frame

    height, width = frame.shape[:2]
    previous_visible = None

    for index, ground_point in enumerate(predicted_ground_trajectory):
        image_point = ground_to_current_image(ground_point, H_current_to_ground)

        if image_point is None:
            previous_visible = None
            continue

        x = float(image_point[0])
        y = float(image_point[1])

        margin = 100

        visible = (
            -margin <= x < width + margin
            and
            -margin <= y < height + margin
        )

        if not visible:
            previous_visible = None
            continue

        current = (int(round(x)), int(round(y)))

        if previous_visible is not None:

            segment_length = math.hypot(
                current[0] - previous_visible[0],
                current[1] - previous_visible[1]
            )

            # Draw every other predicted segment to make a dashed line.
            if (
                index % 2 == 1
                and segment_length <= MAX_DRAW_SEGMENT_PX
            ):
                cv2.line(
                    frame,
                    previous_visible,
                    current,
                    (255, 0, 255),
                    PREDICTION_TRAJECTORY_THICKNESS,
                    cv2.LINE_AA)

        previous_visible = current

    return frame

# ============================================================
# CAMERA POSITION FROM GROUND HOMOGRAPHY
# ============================================================

# Recover an approximate camera ground position from the plane-induced homography and K.
# This decomposes the ground->image homography into rotation/translation up to plane scale.
def camera_position_from_ground_homography(H_image_to_ground, K):
    """
    Recover the camera center from the plane-induced homography.

    Ground coordinates:
        X, Z

    Ground plane:
        Y = 0

    H_image_to_ground:
        image -> ground

    inverse:
        ground -> image

    For a ground plane:

        H_ground_to_image ~ K [r_X r_Z t]

    Camera center:

        C = -R^T t

    Position is returned in the same arbitrary ground scale used by the top-view map.
    """

    try:
        # The pose relation is expressed ground->image, so invert the stored image->ground H.
        H_ground_to_image = np.linalg.inv(H_image_to_ground)
    except np.linalg.LinAlgError:
        return None

    if abs(H_ground_to_image[2, 2]) < 1e-12:
        return None

    # Homogeneous normalization: force the bottom-right entry to 1.
    H_ground_to_image /= H_ground_to_image[2, 2]

    try:
        # K^-1 removes camera intrinsics, leaving the extrinsic homography structure.
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        return None

    # B = K^-1 H ≈ [r_X  r_Z  t] up to one common scale factor.
    # '@' denotes matrix multiplication.
    B = K_inv @ H_ground_to_image

    # Columns correspond approximately to two ground-plane rotation axes and translation.
    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    # Rotation columns should have unit norm; their observed norms reveal the unknown scale.
    norm1 = np.linalg.norm(b1)
    norm2 = np.linalg.norm(b2)

    if (norm1 < 1e-10 or norm2 < 1e-10):
        return None

    # Use the average of both column norms for a symmetric scale estimate.
    base_scale = (2.0 / (norm1 + norm2))
    candidates = []

    # Homography decomposition has a sign ambiguity, so evaluate both ± scale solutions.
    for sign in (1.0, -1.0):
        scale = sign * base_scale
        r_x = scale * b1
        r_z = scale * b2
        t = scale * b3

        # X / Y(normal) / Z coordinate system.

    # The missing axis is perpendicular to the two ground-plane axes: r_y = r_z × r_x.
        r_y = np.cross(r_z, r_x)

    # Assemble an approximate 3×3 rotation matrix from the three axes.
        R_approx = np.column_stack((r_x, r_y, r_z))

        try:
            # SVD projects the approximate matrix onto the nearest orthonormal rotation matrix.
            U, _, Vt = np.linalg.svd(R_approx)
        except np.linalg.LinAlgError:
            continue

        # R = U V^T is the orthogonal factor from the SVD.
        R = U @ Vt

        # Enforce a proper rotation with det(R)=+1 rather than a reflection.
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # Camera centre in world/ground coordinates: C = -R^T t.
        C = -R.T @ t

        if np.isfinite(C).all():
            candidates.append(C)

    if len(candidates) == 0:
        return None

    # Prefer camera above ground.

    # Prefer the physically plausible solution with positive height above the ground plane.
    above_ground = [C for C in candidates if C[1] > 0]

    if len(above_ground) > 0:
        C = max(above_ground, key=lambda candidate: candidate[1])
    else:
        C = candidates[0]
    # Return only horizontal ground coordinates [X,Z]; height C[1] is omitted.
    return np.array([float(C[0]), float(C[2])])


# ============================================================
# MAP BOUNDS
# ============================================================

# Compute top-view bounds that contain ball, camera and predicted paths with padding.
def calculate_map_bounds(ball_trajectory, camera_trajectory, prediction_points=None):
    points = []

    for point in ball_trajectory:
        if point is None:
            continue

        point = np.asarray(point)

        if np.isfinite(point).all():

            points.append([point[0], point[1]])

    for point in camera_trajectory:

        if point is None:
            continue

        point = np.asarray(point)

        if np.isfinite(point).all():
            points.append([point[0], point[1]])

    if prediction_points is not None:

        for point in prediction_points:

            if point is None:
                continue

            point = np.asarray(point)

            if np.isfinite(point).all():
                points.append([point[0], point[1]])

    if len(points) == 0:
        return -1.0, 1.0, -1.0, 1.0

    points = np.asarray(points)

    # Axis-aligned extrema of every valid 2-D ground point.
    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    z_min = float(np.min(points[:, 1]))
    z_max = float(np.max(points[:, 1]))

    # Clamp spans away from zero so later normalization never divides by zero.
    dx = max(x_max - x_min, 0.1)
    dz = max(z_max - z_min, 0.1)

    # Padding

    # Expand each axis by a fixed fraction so trajectories do not touch the map border.
    x_min -= (dx * MAP_PADDING_FRACTION)
    x_max += (dx * MAP_PADDING_FRACTION)

    z_min -= (dz * MAP_PADDING_FRACTION)
    z_max += (dz * MAP_PADDING_FRACTION)

    # --------------------------------------------------------
    # Preserve equal visual scale
    # --------------------------------------------------------

    # Drawable canvas size after removing equal margins on both sides.
    usable_width = (MAP_WIDTH - 2 * MAP_MARGIN)
    usable_height = (MAP_HEIGHT - 2 * MAP_MARGIN)

    # Required width/height ratio of the drawable map region.
    target_aspect = (usable_width / usable_height)

    dx = (x_max - x_min)
    dz = (z_max - z_min)

    # Current physical-coordinate aspect ratio.
    current_aspect = (dx / dz)

    # Expand only the shorter axis range so one ground unit has the same visual scale on x and z.
    if (current_aspect < target_aspect):
        required_dx = (dz * target_aspect)
        extra = (required_dx - dx) / 2.0

        x_min -= extra
        x_max += extra

    else:
        required_dz = (dx / target_aspect)
        extra = (required_dz - dz) / 2.0

        z_min -= extra
        z_max += extra

    return x_min, x_max, z_min, z_max

# ============================================================
# GROUND -> MAP
# ============================================================

# Convert a ground coordinate (X,Z) into an integer pixel on the top-view canvas.
def ground_to_map_pixel(x, z, bounds):
    (x_min, x_max, z_min, z_max) = bounds

    usable_width = (MAP_WIDTH - 2 * MAP_MARGIN)
    usable_height = (MAP_HEIGHT - 2 * MAP_MARGIN)

    # Affine normalization maps X from [x_min,x_max] into the drawable horizontal pixel range.
    px = (MAP_MARGIN + ((x - x_min) / (x_max - x_min)) * usable_width)
    # Z is vertically inverted because image y increases downward while map Z increases upward.
    py = (MAP_HEIGHT - MAP_MARGIN - ((z - z_min) / (z_max - z_min)) * usable_height)

    # Clip numerical edge cases so the result stays inside the map rectangle.
    px = np.clip(px, MAP_MARGIN, MAP_WIDTH  - MAP_MARGIN)
    py = np.clip(py, MAP_MARGIN, MAP_HEIGHT - MAP_MARGIN)

    return (int(round(px)), int(round(py)))

# ============================================================
# TOP-VIEW MAP
# ============================================================

# Render one complete top-view frame: grid, historical paths, prediction and current markers.
def draw_ground_top_view(
    ball_ground,
    camera_ground,
    ball_trajectory,
    camera_trajectory,
    predicted_trajectory,
    frame_number,
    time_s,
    homography_inliers,
    bounds):

    # Start with a light RGB canvas.
    canvas = np.full((MAP_HEIGHT, MAP_WIDTH, 3), 245, dtype=np.uint8)

    (x_min, x_max, z_min, z_max) = bounds

    # ========================================================
    # GRID
    # ========================================================

    GRID_X = 10
    GRID_Z = 8

    for i in range(1, GRID_X):

        # Interpolate equally spaced vertical grid lines in ground X.
        x = (x_min + (i / GRID_X) * (x_max - x_min))

        p1 = ground_to_map_pixel(x, z_min, bounds)
        p2 = ground_to_map_pixel(x, z_max, bounds)

        cv2.line(canvas, p1, p2,
            (215, 215, 215), 1)

    for i in range(1, GRID_Z):
        # Interpolate equally spaced horizontal grid lines in ground Z.
        z = (z_min + (i / GRID_Z) * (z_max - z_min))

        p1 = ground_to_map_pixel(x_min, z, bounds)
        p2 = ground_to_map_pixel(x_max, z, bounds)

        cv2.line(canvas, p1, p2,
            (215, 215, 215), 1)

    # ========================================================
    # BLACK RECTANGLE
    # ========================================================

    cv2.rectangle(
        canvas,
        (MAP_MARGIN, MAP_MARGIN),
        (MAP_WIDTH - MAP_MARGIN, MAP_HEIGHT - MAP_MARGIN),
        (30, 30, 30), 3)

    # ========================================================
    # BALL TRAJECTORY
    # ========================================================

    if len(ball_trajectory) >= 2:
        for i in range(1, len(ball_trajectory)):

            # Convert each pair of consecutive measured ground points to map pixels before drawing.
            p1 = ground_to_map_pixel(
                ball_trajectory[i - 1][0],
                ball_trajectory[i - 1][1],
                bounds)

            p2 = ground_to_map_pixel(
                ball_trajectory[i][0],
                ball_trajectory[i][1],
                bounds)

            cv2.line(
                canvas,
                p1, p2,
                (0, 0, 230),
                4, cv2.LINE_AA)

    # ========================================================
    # PREDICTED BALL TRAJECTORY
    # ========================================================

    if len(predicted_trajectory) >= 2:
        for i in range(1, len(predicted_trajectory)):

            if i % 2 == 0:
                continue

            # Predicted points use the same ground-to-map conversion as measured points.
            p1 = ground_to_map_pixel(
                predicted_trajectory[i - 1][0],
                predicted_trajectory[i - 1][1],
                bounds
            )

            p2 = ground_to_map_pixel(
                predicted_trajectory[i][0],
                predicted_trajectory[i][1],
                bounds
            )

            cv2.line(
                canvas,
                p1, p2,
                (255, 0, 255),
                3, cv2.LINE_AA)

    # ========================================================
    # CAMERA TRAJECTORY
    # ========================================================

    if len(camera_trajectory) >= 2:

        for i in range(1, len(camera_trajectory)):

            # Camera history is drawn in the same fixed coordinate system as the ball.
            p1 = ground_to_map_pixel(
                camera_trajectory[i - 1][0],
                camera_trajectory[i - 1][1],
                bounds)

            p2 = ground_to_map_pixel(
                camera_trajectory[i][0],
                camera_trajectory[i][1],
                bounds)

            cv2.line(
                canvas,
                p1, p2,
                (220, 80, 0),
                3, cv2.LINE_AA)

    # ========================================================
    # CURRENT BALL
    # ========================================================

    if ball_ground is not None:

        bp = ground_to_map_pixel(
            ball_ground[0],
            ball_ground[1],
            bounds)

        cv2.circle(
            canvas,
            bp, 10,
            (0, 0, 230),
            -1)

        cv2.circle(
            canvas,
            bp, 14,
            (255, 255, 255),
            2)

        cv2.putText(
            canvas,
            "BALL",
            (bp[0] + 15, bp[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 180),
            2, cv2.LINE_AA)

    # ========================================================
    # CURRENT CAMERA
    # ========================================================

    if camera_ground is not None:

        cp = ground_to_map_pixel(
            camera_ground[0],
            camera_ground[1],
            bounds)

        cv2.circle(
            canvas,
            cp, 10,
            (220, 80, 0),
            -1)

        cv2.circle(
            canvas,
            cp, 14,
            (255, 255, 255),
            2)

        cv2.putText(
            canvas,
            "CAMERA",
            (cp[0] + 15, cp[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 60, 0),
            2, cv2.LINE_AA)

    # ========================================================
    # TEXT
    # ========================================================

    cv2.putText(
        canvas,
        "GROUND-FIXED TOP VIEW",
        (25, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (20, 20, 20),
        2, cv2.LINE_AA)

    cv2.putText(
        canvas,
        "RED = BALL    BLUE = CAMERA    MAGENTA = ML/FALLBACK PREDICTION",
        (25, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (40, 40, 40),
        1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        (f"time={time_s:.2f}s   "
         f"frame={frame_number}   "
          f"inliers={homography_inliers}"),
        (25, MAP_HEIGHT - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (40, 40, 40),
        1, cv2.LINE_AA)

    return canvas

# ============================================================
# EXCEL
# ============================================================

# Export all per-frame measurements, velocities and prediction diagnostics to an Excel workbook.
def write_excel(rows, filename):

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = ("Ground tracking")

    headers = [
        "Frame",
        "Time_s",
        "Ball_Detected",
        "Confidence",
        "Pixel_X",
        "Pixel_Y",
        "BBox_Width_px",
        "BBox_Height_px",
        "Ball_Diameter_px",
        "Ball_X_Camera_m",
        "Ball_Y_Camera_m",
        "Ball_Z_Camera_m",
        "Ball_VX_Camera_m_s",
        "Ball_VY_Camera_m_s",
        "Ball_VZ_Camera_m_s",
        "Ball_Speed_Camera_m_s",
        "Velocity_Current_Measurement",
        "Ball_Distance_Camera_m",
        "Ball_Ground_X",
        "Ball_Ground_Z",
        "Ball_Ground_VX_units_s",
        "Ball_Ground_VZ_units_s",
        "Ball_Ground_Speed_units_s",
        "Ground_Velocity_Fit_RMSE",
        "Prediction_Valid",
        "Prediction_Method",
        "ML_Training_Samples",
        "ML_Fit_RMSE",
        "Prediction_Horizon_s",
        "Predicted_Ground_X_Horizon",
        "Predicted_Ground_Z_Horizon",
        "Camera_Ground_X",
        "Camera_Ground_Z",
        "Camera_Motion_Estimated",
        "Homography_Inliers"]

    # First row contains column names matching the keys stored in each per-frame dictionary.
    sheet.append(headers)

    for cell in sheet[1]:
        cell.font = Font(bold=True)

    for row in rows:
        sheet.append(
            [row["frame"],
             row["time"],
             row["detected"],
             row["confidence"],
             row["pixel_x"],
             row["pixel_y"],
             row["box_width"],
             row["box_height"],
             row["diameter_px"],
             row["ball_x_camera"],
             row["ball_y_camera"],
             row["ball_z_camera"],
             row["ball_vx_camera"],
             row["ball_vy_camera"],
             row["ball_vz_camera"],
             row["ball_speed_camera"],
             row["velocity_current_measurement"],
             row["ball_distance"],
             row["ball_ground_x"],
             row["ball_ground_z"],
             row["ball_ground_vx"],
             row["ball_ground_vz"],
             row["ball_ground_speed"],
             row["ground_velocity_fit_rmse"],
             row["prediction_valid"],
             row["prediction_method"],
             row["ml_training_samples"],
             row["ml_fit_rmse"],
             row["prediction_horizon"],
             row["predicted_ground_x_horizon"],
             row["predicted_ground_z_horizon"],
             row["camera_ground_x"],
             row["camera_ground_z"],
             row["camera_motion"],
             row["homography_inliers"]])

    # Keep the header visible while scrolling through long trajectories.
    sheet.freeze_panes = "A2"

    # Auto-size each column from its longest displayed value, capped for readability.
    for column in sheet.columns:
        maximum = 0
        letter = (column[0].column_letter)

        for cell in column:
            value = ("" if cell.value is None else str(cell.value))
            maximum = max(maximum, len(value))

        sheet.column_dimensions[letter].width = min(maximum + 2, 30)

    workbook.save(filename)


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

# Instantiate the YOLO detector once; the same model is reused for every frame.
model = YOLO(MODEL)

# ============================================================
# OPEN VIDEO
# ============================================================

# Open the input video stream and read its metadata.
video_in = cv2.VideoCapture(INPUT_VIDEO)
if not video_in.isOpened():
    raise RuntimeError(f"Could not open {INPUT_VIDEO}")

fps = video_in.get(cv2.CAP_PROP_FPS)
width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

# Use 30 FPS only as a safety fallback when the file does not report a valid frame rate.
if fps <= 0:
    fps = 30.0

# ============================================================
# INTRINSICS
# ============================================================

# Approximate intrinsics used by depth back-projection and camera-pose decomposition.
fx, fy, px, py = get_intrinsics()

# Standard pinhole intrinsic matrix K = [[fx,0,cx],[0,fy,cy],[0,0,1]].
K = np.array(
    [   [fx, 0.0, px],
        [0.0, fy, py],
        [0.0, 0.0, 1.0]])

# ============================================================
# INITIAL GROUND SYSTEM
# ============================================================

# Fixed reference-image -> ground homography established once from the chosen ground quadrilateral.
H_reference_to_ground = (create_initial_ground_coordinate_system(width, height))

# ============================================================
# OUTPUT WRITER
# ============================================================

# Convert the four-character codec string (e.g. 'XVID') into OpenCV's integer FOURCC identifier.
fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

# Writer for the annotated RGB video; frame size must match the input video.
rgb_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

if not rgb_writer.isOpened():
    raise RuntimeError("Could not create annotated RGB video.")

# ============================================================
# STATE
# ============================================================

prev_gray = None
last_ball_center = None
last_ball_box = None

# Current image -> first/reference image

# cumulative_H maps the current frame back to the first/reference frame.
# Identity means the first frame initially coincides with itself.
cumulative_H = np.eye(3, dtype=np.float64)

# Short rolling histories used by the median filters.
ball_xyz_history = deque(maxlen=BALL_SMOOTH_WINDOW)
ball_velocity_history = deque(maxlen=VELOCITY_SMOOTH_WINDOW)

# Ground history must be long enough for both velocity fitting and ML training.
timed_ground_history = deque(maxlen=max(GROUND_VELOCITY_WINDOW, ML_PREDICTION_WINDOW))
camera_position_history = deque(maxlen=CAMERA_SMOOTH_WINDOW)

# Last valid state is retained so finite differences and fallback values can be computed.
last_velocity_position = None
last_velocity_time = None
last_ball_velocity = np.zeros(3, dtype=np.float64)
have_ball_velocity = False
last_ground_velocity = np.zeros(2, dtype=np.float64)
have_ground_velocity = False

# IMPORTANT:
#
# This trajectory is stored in FIXED GROUND COORDINATES.
#
# It is NOT a list of historical screen pixels.

# Persistent trajectories are stored in ground coordinates, not changing screen pixels.
ball_ground_trajectory = []
camera_ground_trajectory = []
ball_xyz_trajectory = []

# Prediction points are collected so the automatic map bounds include
# the complete future path and keep it inside the black rectangle.
# Collect all predicted points so final map bounds can include the complete forecast paths.
all_prediction_ground_points = []
rows = []
map_frame_data = []
frame_number = 0

# ============================================================
# PROCESS VIDEO
# ============================================================

print()
print("Tracking ball, ground and camera...")
print()

# Process the video sequentially; each iteration corresponds to one frame.
while True:
    ret, frame = video_in.read()
    if not ret:
        break

    # Frame time in seconds, assuming constant frame spacing of 1/fps.
    frame_number += 1
    time_s = (frame_number - 1) / fps
    # Camera-motion estimation operates on grayscale intensity rather than RGB colour.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ========================================================
    # CAMERA MOTION
    # ========================================================

    camera_motion_estimated = False
    homography_inliers = 0

    # Camera motion requires two frames, so it starts from the second video frame.
    if prev_gray is not None:

        # Estimate H_step: current frame -> previous frame from tracked background features.
        H_step, homography_inliers = (estimate_camera_homography(prev_gray, gray, last_ball_box))

        if H_step is not None:
            # Compose projective mappings: current->previous followed by previous->reference.
            candidate_H = cumulative_H @ H_step

            if homography_valid(candidate_H):
                # Normalize projective scale after matrix composition.
                candidate_H /= (candidate_H[2, 2])
                cumulative_H = (candidate_H)
                camera_motion_estimated = True

    # ========================================================
    # CURRENT IMAGE -> FIXED GROUND
    # ========================================================

    # Compose current->reference with reference->ground:
    # H_current->ground = H_reference->ground @ H_current->reference.
    H_current_to_ground = H_reference_to_ground @ cumulative_H

    # Homographies are scale-equivalent; normalize to keep values numerically stable.
    if abs(H_current_to_ground[2, 2]) > 1e-12:
        H_current_to_ground /= H_current_to_ground[2, 2]

    # ========================================================
    # CAMERA GROUND POSITION
    # ========================================================

    # Decompose the current image->ground homography to estimate the camera's ground position.
    camera_raw = (
        camera_position_from_ground_homography(H_current_to_ground, K))

    current_camera = None
    if camera_raw is not None:
        # Smooth reconstructed camera positions to reduce homography jitter.
        camera_position_history.append(camera_raw)

        current_camera = median_smooth(
            camera_position_history,
            CAMERA_SMOOTH_WINDOW)

        camera_ground_trajectory.append((float(current_camera[0]), float(current_camera[1])))

    # If this frame's pose fails, reuse the most recent valid camera ground position.
    elif len(camera_ground_trajectory) > 0:
        current_camera = np.array(camera_ground_trajectory[-1])

    # ========================================================
    # YOLO BALL DETECTION
    # ========================================================

    # Run YOLO only for the sports-ball class; each result contains boxes and confidence scores.
    results = model(
        frame,
        verbose=False,
        conf=CONFIDENCE,
        classes=[SPORTS_BALL_CLASS])

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(
                box.conf[0])

            # Convert the selected box tensor from the model device to a NumPy [x1,y1,x2,y2] vector.
            coords = (box.xyxy[0].cpu().numpy())

            x1 = int(coords[0])
            y1 = int(coords[1])
            x2 = int(coords[2])
            y2 = int(coords[3])

            # Bounding-box centre in image pixels.
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Store all ball candidates before selecting the temporal best match.
            detections.append(
                {"box": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence":confidence})

    # ========================================================
    # SELECT MOST LIKELY BALL
    # ========================================================

    selected = None

    if len(detections) > 0:

        # With no previous ball, initialize tracking from the highest-confidence candidate.
        if last_ball_center is None:
            selected = max(detections,
                key=lambda detection:
                    detection["confidence"])

        else:
            # Score balances temporal proximity against detector confidence; lower is better.
            def detection_score(detection):
                center = (detection["center"])
                # Pixel displacement from the previously selected ball centre.
                distance = math.hypot(center[0] - last_ball_center[0],
                                      center[1] - last_ball_center[1])

                # Confidence is rewarded by subtracting 100*confidence from spatial distance.
                return (distance - 100.0 * detection["confidence"])

            selected = min(detections, key=detection_score)

    # ========================================================
    # DEFAULT DATA
    # ========================================================

    # Defaults ensure every video frame produces a complete Excel row even if detection fails.
    current_ball_ground = None
    current_prediction = []
    prediction_method = "none"
    ml_fit_rmse = None
    ml_training_samples = 0

    # Every Excel row receives velocity fields. If the ball is not
    # detected in the current frame, the last valid estimate is carried
    # forward and Velocity_Current_Measurement is False.
    # Carry forward the last valid camera-relative velocity when this frame has no new measurement.
    default_camera_velocity = (
        last_ball_velocity
        if have_ball_velocity
        else np.zeros(3, dtype=np.float64))

    # Same carry-forward rule for fixed-ground velocity.
    default_ground_velocity = (
        last_ground_velocity
        if have_ground_velocity
        else np.zeros(2, dtype=np.float64))

    # Per-frame record; fields are overwritten below when a valid ball is processed.
    row = {      "frame": frame_number,
                  "time": time_s,
              "detected": False,
            "confidence": None,
               "pixel_x": None,
               "pixel_y": None,
             "box_width": None,
            "box_height": None,
           "diameter_px": None,
         "ball_x_camera": None,
         "ball_y_camera": None,
         "ball_z_camera": None,
        "ball_vx_camera": float(default_camera_velocity[0]),
        "ball_vy_camera": float(default_camera_velocity[1]),
        "ball_vz_camera": float(default_camera_velocity[2]),
     "ball_speed_camera": float(np.linalg.norm(default_camera_velocity)),
     "velocity_current_measurement": False,
         "ball_distance": None,
         "ball_ground_x": None,
         "ball_ground_z": None,
        "ball_ground_vx": float(default_ground_velocity[0]),
        "ball_ground_vz": float(default_ground_velocity[1]),
     "ball_ground_speed": float(np.linalg.norm(default_ground_velocity)),
     "ground_velocity_fit_rmse": None,
      "prediction_valid": False,
     "prediction_method": prediction_method,
     "ml_training_samples": ml_training_samples,
           "ml_fit_rmse": ml_fit_rmse,
    "prediction_horizon": PREDICTION_HORIZON_S,
    "predicted_ground_x_horizon": None,
    "predicted_ground_z_horizon": None,
       "camera_ground_x":
            (None if current_camera is None
                else float(current_camera[0])),
       "camera_ground_z":
            (None if current_camera is None
                else float(current_camera[1])),
         "camera_motion": camera_motion_estimated,
    "homography_inliers": homography_inliers}

    # ========================================================
    # PROCESS BALL
    # ========================================================

    # Continue geometric/kinematic processing only when a ball candidate was selected.
    if selected is not None:
        (x1, y1, x2, y2) = selected["box"]
        (cx, cy) = selected["center"]

        confidence = (selected["confidence"])

        # ----------------------------------------------------
        # Apparent diameter
        # ----------------------------------------------------

        # Pixel diameter is required for the monocular size-based depth estimate.
        diameter_px = (ball_diameter_from_box(selected["box"]))

        # ----------------------------------------------------
        # Camera-relative 3-D
        # ----------------------------------------------------

        # Reconstruct raw camera-relative [X,Y,Z] from image centre and apparent ball diameter.
        xyz_raw = pixel_to_camera(
            cx, cy,
            diameter_px,
            fx, fy,
            px, py)

        # Smooth 3-D position before using it for velocity.
        ball_xyz_history.append(xyz_raw)
        xyz = median_smooth(ball_xyz_history, BALL_SMOOTH_WINDOW)
        ball_xyz_trajectory.append(xyz.copy())

        # ----------------------------------------------------
        # Camera-relative 3-D velocity
        # ----------------------------------------------------

        # Finite-difference velocity uses the previous smoothed position/time pair.
        raw_velocity = finite_difference_velocity(
            last_velocity_position,
            last_velocity_time,
            xyz,
            time_s)

        velocity_current_measurement = False

        if raw_velocity is not None:
            ball_velocity_history.append(
                raw_velocity)

            # Median-filter raw velocity to suppress frame-to-frame depth noise.
            velocity_camera = median_smooth(
                ball_velocity_history,
                VELOCITY_SMOOTH_WINDOW)

            last_ball_velocity = (velocity_camera.copy())

            have_ball_velocity = True
            velocity_current_measurement = True

        elif have_ball_velocity:
            velocity_camera = (last_ball_velocity.copy())

        else:
            velocity_camera = np.zeros(
                3, dtype=np.float64)

        # Update the finite-difference reference state for the next valid ball measurement.
        last_velocity_position = xyz.copy()
        last_velocity_time = time_s

        # Euclidean speed magnitude ||v|| = sqrt(Vx²+Vy²+Vz²).
        speed_camera = float(
            np.linalg.norm(velocity_camera))

        # Camera-to-ball distance ||p|| = sqrt(X²+Y²+Z²).
        distance_camera = float(
            np.linalg.norm(xyz))

        # ----------------------------------------------------
        # BALL GROUND POSITION
        # ----------------------------------------------------

        # Ground mapping uses the bottom-centre of the ball box as its ground-contact image point.
        contact_pixel = (
            ball_ground_contact_from_box(
                selected["box"]))

        # Project that pixel through H_current->ground into fixed [X,Z] coordinates.
        ground_candidate = (
            current_image_to_ground(
                contact_pixel,
                H_current_to_ground))

        # Reject NaN/Inf projections before adding them to any trajectory/history.
        if np.isfinite(ground_candidate).all():
            current_ball_ground = (ground_candidate)
            ball_ground_trajectory.append(
                (float(ground_candidate[0]),
                 float(ground_candidate[1])))

            # Store timestamp + fixed-ground position for velocity fitting and online ML.
            timed_ground_history.append((
                    time_s,
                    np.asarray(
                        current_ball_ground,
                        dtype=np.float64
                    ).copy()))

            # Fit recent ground positions to obtain a smoothed fallback velocity and fit RMSE.
            ground_velocity, ground_fit_rmse = (
                estimate_ground_velocity(
                    timed_ground_history,
                    GROUND_VELOCITY_WINDOW))

            if ground_velocity is not None:
                last_ground_velocity = (ground_velocity.copy())
                have_ground_velocity = True
            else:
                ground_fit_rmse = None

            # ------------------------------------------------
            # ML TRAJECTORY PREDICTION
            # ------------------------------------------------
            # Train an online polynomial Ridge model on recent fixed-ground
            # positions. Because the training coordinates are ground-fixed,
            # camera motion has already been compensated before learning.
            # Attempt the polynomial Ridge predictor first; it returns path, fit error and sample count.
            (ml_prediction, ml_fit_rmse, ml_training_samples) = predict_ground_trajectory_ml(
                timed_ground_history, current_ball_ground, PREDICTION_HORIZON_S, PREDICTION_STEPS)

            # Prefer the ML prediction whenever it produced at least two valid path points.
            if len(ml_prediction) >= 2:
                current_prediction = ml_prediction
                prediction_method = "ML_Polynomial_Ridge"

            # Otherwise use the constant-velocity predictor once enough ground samples exist.
            elif ground_velocity is not None and len(timed_ground_history) >= MIN_PREDICTION_POINTS:

                # Early in the video there may not yet be enough samples to
                # train the ML model. Keep a deterministic fallback so the
                # output still contains a prediction.
                current_prediction = (
                    predict_ground_trajectory_constant_velocity(
                        current_ball_ground,
                        ground_velocity,
                        PREDICTION_HORIZON_S,
                        PREDICTION_STEPS))

                if len(current_prediction) >= 2:
                    prediction_method = ("Constant_Velocity_Fallback")

            # Preserve every forecast point so final automatic map limits include predictions.
            if len(current_prediction) >= 2:
                all_prediction_ground_points.extend(current_prediction)

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        # Replace default row values with measurements computed for this detected ball.
        row.update(
            {"detected": True,
           "confidence": confidence,
              "pixel_x": cx,
              "pixel_y": cy,
            "box_width": x2 - x1,
           "box_height": y2 - y1,
          "diameter_px": diameter_px,
        "ball_x_camera": float(xyz[0]),
        "ball_y_camera": float(xyz[1]),
        "ball_z_camera": float(xyz[2]),
       "ball_vx_camera": float(velocity_camera[0]),
       "ball_vy_camera": float(velocity_camera[1]),
       "ball_vz_camera": float(velocity_camera[2]),
    "ball_speed_camera": speed_camera,
    "velocity_current_measurement": velocity_current_measurement,
        "ball_distance": distance_camera,
        "ball_ground_x": (
                        None
                        if current_ball_ground
                        is None
                        else float(
                            current_ball_ground[0])),
        "ball_ground_z": (
                        None
                        if current_ball_ground
                        is None
                        else float(
                            current_ball_ground[1])),
        "ball_ground_vx":
                    float(
                        last_ground_velocity[0]
                        if have_ground_velocity
                        else 0.0),
        "ball_ground_vz":
                    float(
                        last_ground_velocity[1]
                        if have_ground_velocity
                        else 0.0),
    "ball_ground_speed":
                    float(
                        # Ground speed magnitude ||v_ground|| = sqrt(Vx²+Vz²).
                        np.linalg.norm(
                            last_ground_velocity
                            if have_ground_velocity
                            else np.zeros(2))),
        "ground_velocity_fit_rmse":
                    ground_fit_rmse
                    if current_ball_ground is not None
                    else None,
       "prediction_valid": len(current_prediction) >= 2,
      "prediction_method": prediction_method,
    "ml_training_samples": ml_training_samples,
            "ml_fit_rmse": ml_fit_rmse,
     "prediction_horizon": PREDICTION_HORIZON_S,
        # Store only the final prediction point as the horizon endpoint in Excel.
        "predicted_ground_x_horizon":
                    (   None
                        if len(current_prediction) < 2
                        else float(
                            current_prediction[-1][0])),
        "predicted_ground_z_horizon":
                    (   None
                        if len(current_prediction) < 2
                        else float(
                            current_prediction[-1][1]))
            }
        )

        # These values guide temporal candidate selection and feature masking in the next frame.
        last_ball_center = (cx, cy)
        last_ball_box = (x1, y1, x2, y2)

    # ========================================================
    # *** KEY CHANGE ***
    #
    # DRAW THE FIXED-GROUND BALL TRAJECTORY IN CURRENT IMAGE
    # ========================================================

    # Historical and future paths are stored in ground coordinates, then reprojected into this frame.
    frame = draw_ground_trajectory_on_frame(
        frame,
        ball_ground_trajectory,
        H_current_to_ground)

    frame = draw_ground_prediction_on_frame(
        frame,
        current_prediction,
        H_current_to_ground)

    # ========================================================
    # DRAW CURRENT BALL ON TOP OF TRAJECTORY
    # ========================================================

    if selected is not None:
        (x1, y1, x2, y2) = selected["box"]
        (cx, cy) = selected["center"]

        # Draw YOLO bounding box around the currently selected ball.
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            BOX_THICKNESS)

        cv2.circle(
            frame,
            (cx, cy),
            4,
            (0, 255, 255),
            -1)

        contact_pixel = ball_ground_contact_from_box(selected["box"])

        # Convert floating-point contact pixel to integer coordinates required by OpenCV drawing.
        contact_int = (int(round(contact_pixel[0])),
                       int(round(contact_pixel[1])))

        # Current ground point

        cv2.circle(
            frame,
            contact_int,
            7,
            (0, 0, 255),
            -1)

        cv2.circle(
            frame,
            contact_int,
            10,
            (255, 255, 255),
            2)

        cv2.putText(
            frame,
            "BALL GROUND POINT",
            (contact_int[0] + 12,
             contact_int[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2, cv2.LINE_AA)

        # Recover camera-relative [X,Y,Z] from the row for on-screen annotation.
        xyz = np.array(
            [
                row["ball_x_camera"],
                row["ball_y_camera"],
                row["ball_z_camera"]
            ]
        )

        # Human-readable position/distance overlay.
        info = (
            f"t={time_s:.2f}s  "
            f"X={xyz[0]:+.2f}m  "
            f"Y={xyz[1]:+.2f}m  "
            f"Z={xyz[2]:.2f}m  "
            f"D={row['ball_distance']:.2f}m")

        cv2.putText(
            frame,
            info,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2, cv2.LINE_AA)

        # Human-readable velocity overlay.
        velocity_info = (
            f"Vx={row['ball_vx_camera']:+.2f}m/s  "
            f"Vy={row['ball_vy_camera']:+.2f}m/s  "
            f"Vz={row['ball_vz_camera']:+.2f}m/s  "
            f"speed={row['ball_speed_camera']:.2f}m/s")

        cv2.putText(
            frame,
            velocity_info,
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 255, 255),
            2, cv2.LINE_AA)

        # Show whether this frame used ML, fallback prediction, or no predictor yet.
        prediction_info = (
            f"predictor={row['prediction_method']}  "
            f"ML_samples={row['ml_training_samples']}")

        cv2.putText(
            frame,
            prediction_info,
            (20, 89),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 0, 255),
            2, cv2.LINE_AA)

    # ========================================================
    # EXPLANATION ON VIDEO
    # ========================================================

    cv2.putText(
        frame,
        "RED = measured trajectory",
        (20, height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2, cv2.LINE_AA)

    cv2.putText(
        frame,
        "MAGENTA = ML/fallback prediction",
        (20, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2, cv2.LINE_AA)

    # ========================================================
    # STORE MAP DATA
    # ========================================================

    # Cache minimal per-frame ground data; it will be rendered later after global map bounds are known.
    map_frame_data.append(
        {
            "frame": frame_number,
             "time": time_s,
             "ball":
                (   None
                    if current_ball_ground
                    is None
                    else (float(current_ball_ground[0]),
                          float(current_ball_ground[1]))
                ),
            "camera":
                (   None
                    if current_camera
                    is None
                    else (float(current_camera[0]),
                          float(current_camera[1]))
                ),
            "prediction": list(current_prediction),
            "inliers": homography_inliers
        }
    )

    # Append the completed measurement row for the later Excel export.
    rows.append(row)

    # ========================================================
    # WRITE ANNOTATED RGB FRAME
    # ========================================================

    # Encode the annotated RGB frame into the output video.
    rgb_writer.write(frame)

    # ========================================================
    # NEXT FRAME
    # ========================================================

    # Current grayscale image becomes the 'previous' image on the next iteration.
    prev_gray = (gray.copy())

    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 30 == 0:
        # Progress percentage, protected against a zero frame-count metadata value.
        progress = (100.0 * frame_number / max(frame_count, 1))
        print(
            f"Tracking: "
            f"{progress:.1f}%"
        )

# ============================================================
# CLOSE INPUT / RGB OUTPUT
# ============================================================

# Release video handles before creating the independent top-view output.
video_in.release()
rgb_writer.release()

# ============================================================
# AUTOMATIC TOP-VIEW BOUNDS
# ============================================================

# Compute one common set of map limits from the complete measured, camera and predicted trajectories.
map_bounds = calculate_map_bounds(
    ball_ground_trajectory,
    camera_ground_trajectory,
    all_prediction_ground_points)

print()
print("Automatic ground-map bounds:")
print(
    f"X = "
    f"{map_bounds[0]:.3f} "
    f"to "
    f"{map_bounds[1]:.3f}")
print(
    f"Z = "
    f"{map_bounds[2]:.3f} "
    f"to "
    f"{map_bounds[3]:.3f}")
print()

# ============================================================
# TOP-VIEW VIDEO
# ============================================================

# Create the second video writer for the fixed-size top-view map.
top_writer = cv2.VideoWriter(
    OUTPUT_TOP_VIEW,
    fourcc,
    fps,
    (MAP_WIDTH, MAP_HEIGHT)
)

if not top_writer.isOpened():
    raise RuntimeError("Could not create top-view video.")

# ============================================================
# SECOND PASS FOR MAP
# ============================================================

print("Rendering top-view map...")
ball_map_history = []
camera_map_history = []

# Second pass: replay cached ground data now that final map bounds are known.
for index, data in enumerate(map_frame_data):
    if data["ball"] is not None:
        ball_map_history.append(
            data["ball"])

    if data["camera"] is not None:
        camera_map_history.append(data["camera"])

    # Render one map frame using histories accumulated up to this time step.
    top_map = draw_ground_top_view(data["ball"],
        data["camera"],
        ball_map_history,
        camera_map_history,
        data["prediction"],
        data["frame"],
        data["time"],
        data["inliers"],
        map_bounds)

    # Encode the top-view frame.
    top_writer.write(top_map)

    if (index + 1) % 30 == 0:
        progress = (100.0 * (index + 1) / max(len(map_frame_data), 1))

        print(
            f"Top view: "
            f"{progress:.1f}%")

top_writer.release()

# ============================================================
# EXCEL
# ============================================================

# Write the accumulated per-frame dictionaries to Excel.
write_excel(rows, OUTPUT_EXCEL)

# ============================================================
# DONE
# ============================================================

print()
print("==========================================")
print("DONE")
print("==========================================")
print()
print("Annotated RGB video:")
print(os.path.abspath(OUTPUT_VIDEO))
print()
print("Ground top-view:")
print(os.path.abspath(OUTPUT_TOP_VIEW))
print()
print("Excel:")
print(os.path.abspath(OUTPUT_EXCEL))
print()
print("rgb_out_ground_tracking.avi:")
print("The RED trajectory is stored on the fixed ground plane and reprojected into each current camera frame.")
print("It is therefore compensated for estimated camera motion.")
print()
print("ball_camera_ground_map.avi:")
print("RED = measured ball ground trajectory")
print("MAGENTA = ML-predicted ball ground trajectory")
print("BLUE = camera ground trajectory")
print("Both use the same fixed ground coordinate system.")
print()
print("NOTE:")
print("Ground scale is arbitrary because no physical terrace distance is known.")