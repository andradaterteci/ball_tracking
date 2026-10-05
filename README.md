INITIALIZE YOLO detector, camera intrinsics, reference-image-to-ground homography, histories, buffers, and output writers

FOR each RGB frame:

&#x20;   compute frame time

&#x20;   convert the current frame to grayscale

&#x20;   estimate current-to-previous camera-motion homography

&#x20;       using background feature tracking + RANSAC

&#x20;   IF the estimated camera-motion homography is valid:

&#x20;       accumulate the current-to-reference homography

&#x20;   combine the current-to-reference homography with the

&#x20;       reference-to-ground homography to obtain H\_current\_to\_ground

&#x20;   reconstruct the current camera position in the fixed-ground coordinate system

&#x20;   detect the ball with YOLO

&#x20;   select the most plausible ball detection

&#x20;   IF a ball is detected:

&#x20;       estimate camera-relative 3D ball position from

&#x20;           image position + apparent ball diameter

&#x20;       smooth the camera-relative 3D position

&#x20;       compute camera-relative 3D velocity

&#x20;       smooth the camera-relative velocity

&#x20;       approximate the ball-ground contact point using the

&#x20;           bottom-center of the bounding box

&#x20;       transform the ball-ground contact point from the current image into 

&#x20;           fixed-ground coordinates using H\_current\_to\_ground

&#x20;       append the fixed-ground position to the trajectory history

&#x20;       estimate recent fixed-ground velocity using linear least-squares fitting

&#x20;       TRY polynomial-feature + Ridge regression trajectory prediction

&#x20;       IF the ML prediction is valid:

&#x20;           use ML\_Polynomial\_Ridge trajectory

&#x20;       ELSE IF constant-velocity prediction is available:

&#x20;           use Constant\_Velocity\_Fallback trajectory

&#x20;       ELSE:

&#x20;           no future trajectory is available yet

&#x20;   reproject the measured and predicted fixed-ground trajectories

&#x20;       into the current RGB frame

&#x20;   store the numerical output row

&#x20;   store the information required for the top-view frame

END FOR

compute global top-view bounds from the stored trajectories

render the top-view video in a second pass

write the Excel output

save the camera-relative 3D trajectory plot

