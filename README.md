# kommuai_challenge
kommuai_challenge

1) run code
write down the directory:
kommu_ai/
→batch_process.py
→video_input
→output_schematics //automatically created

install: pip install opencv-python numpy
execute: python batch_process.py

// for yolo
install: pip install ultralytics
execute: python batch_process.py

2) METHOD
Pre-process:
downscale the image (lessen the burden for the cpu, the cpu used is too weak for the whole process).
gaussian blur and converted the image into grayscale.
low pass filter to eliminate high-frequency noise.

segmentation:   
computes the threshold value for each pixel's neighbourhood
isolates the road markings

region of interest:
detect the peak intensity positions when the lane enters the frame.
dynamically build custom coordinate bounds.

3)ASSUMPTION
- the mapping of the lane dependant on the camera position as it mounted to the car.
- different terrain will affect the visibility of the road on the camera.
- distance between right boundaries and left might not be constant.
- pc does not have the computational power to carry out the processing of the video.
- downscaling the image to allow pc to carry out the task at the cost of accuracy.

4) CAMERA
- the camera's center is calculated by taking the width and the height of the grid (grid of the whole image) and divide it by 2.
- check the image for the intensity of the target (road), a double lane would show a split in its intensity.
- measure the most left and most right of the road relatively to the center at the bottom row (the closest part of the car to the road).

5) LIMITATIONS
- Changes in lighting, blurring the lines between the road and the edges.
- Noise created by factor such as lighting and the quality of the visual.
- Changes in the landscape would distort the recognition of the lane.
- Not enough computational power to carry out task.

improving by adding filter that allows the shadow and dead pixels to be estimated.
deep learning to predict the boundaries and recognise the change in topography.
adding datasets to the deep learning systems, allow the module to accurately recognise the lanes and its changes.
use pc with higher processing/computational power.



