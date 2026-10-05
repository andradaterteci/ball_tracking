\# ============================================================

\# README.md

\# Ball tracking and ML trajectory prediction

\# Intel RealSense D435i RGB video

\# ============================================================

\#

\# INPUT: rgb.avi

\#

\# OUTPUTS

\# ------------------------------------------------------------

\# 1. rgb\_out\_ground\_tracking.avi

\#

\#    Annotated original RGB video containing:

\#       - ball detection

\#       - approximate ball ground-contact point

\#       - RED ground-fixed measured ball trajectory

\#

\#    Each ball point is stored in the fixed ground coordinate

\#    system and then reprojected into the current camera image.

\#

\#    The trajectory is therefore intended to remain fixed

\#    relative to the ground as the camera moves.

\#

\#

\# 2. ball\_camera\_ground\_map.avi

\#

\#       RED     = measured ball ground trajectory

\#       MAGENTA = ML/fallback predicted future ground trajectory

\#       BLUE    = reconstructed camera ground trajectory

\#

\#

\# 3. Excel trajectory data, including per-frame camera-relative

\#    position and velocity, ground position and velocity,

\#    prediction method/fit, prediction endpoint, camera position,

\#    and homography information

