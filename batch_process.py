import cv2
import numpy as np
import os
import json

# --- Configurations ---
INPUT_DIR = "video_input"
OUTPUT_DIR = "output_schematics"

# TARGET RESOLUTION FOR SPEED: 
# Downscaling to 640x360 reduces the CPU load by over 85%!
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

# CAP SELECTION: Only extract a total maximum of 10-15 screenshots per video 
# to ensure your processor does not freeze.
MAX_TOTAL_SNAPSHOTS = 15

script_dir = os.path.dirname(os.path.abspath(__file__))
abs_input_path = os.path.join(script_dir, INPUT_DIR)
abs_output_path = os.path.join(script_dir, OUTPUT_DIR)

if not os.path.exists(abs_output_path):
    os.makedirs(abs_output_path)


def dynamically_extract_lane(img):
    """Ultra-fast lightweight layout extraction."""
    h, w = img.shape[:2]
    
    # Preprocessing on lightweight downscaled image matrix
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Calculate base column histogram boundaries
    bottom_half_hist = np.sum(thresh[int(h*0.8):h, :], axis=0)
    midpoint = w // 2
    
    left_base = np.argmax(bottom_half_hist[:midpoint])
    right_base = np.argmax(bottom_half_hist[midpoint:]) + midpoint
    
    if bottom_half_hist[left_base] < 300: left_base = int(w * 0.15)
    if bottom_half_hist[right_base] < 300: right_base = int(w * 0.85)

    src = np.float32([
        [w * 0.43, h * 0.62], [w * 0.57, h * 0.62],
        [right_base, h], [left_base, h]
    ])
    dst = np.float32([
        [w * 0.25, 0], [w * 0.75, 0],
        [w * 0.75, h], [w * 0.25, h]
    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(thresh, M, (w, h))
    
    # Minimalist Single-Pass Tracking Window Search (Optimized for slow CPUs)
    # Instead of doing 10 heavy loop windows, we use 4 fast slices
    nwindows = 4
    window_height = h // nwindows
    nonzero = warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    
    left_lane_inds, right_lane_inds = [], []
    cx_left, cx_right, margin = int(w * 0.25), int(w * 0.75), int(w * 0.10)
    
    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= cx_left - margin) & (nonzerox < cx_left + margin)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= cx_right - margin) & (nonzerox < cx_right + margin)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > 20: cx_left = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > 20: cx_right = int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 1) # Use linear fit for speed
        right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 1)
        
        ploty = np.linspace(0, h-1, h)
        left_fitx = left_fit[0]*ploty + left_fit[1]
        right_fitx = right_fit[0]*ploty + right_fit[1]
        visible_lanes = 2
    except:
        ploty = np.linspace(0, h-1, h)
        left_fitx = np.linspace(w * 0.25, w * 0.25, h)
        right_fitx = np.linspace(w * 0.75, w * 0.75, h)
        visible_lanes = 0

    # Real-World metric scaling approximations
    camera_center = w / 2
    lane_width_px = (right_fitx[-1] - left_fitx[-1])
    pixel_offset = ((left_fitx[-1] + right_fitx[-1]) / 2) - camera_center
    xm_per_pix = 3.7 / lane_width_px if lane_width_px != 0 else 0.01
    meter_offset = pixel_offset * xm_per_pix
    
    left_edge_relative = (left_fitx[-1] - camera_center) * xm_per_pix
    right_edge_relative = (right_fitx[-1] - camera_center) * xm_per_pix

    # Render schematic layout drawing
    road_schematic = np.full((h, w, 3), 255, dtype=np.uint8) 
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    road_corridor = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_schematic, np.int_([road_corridor]), (235, 235, 235))
    
    pts_left_solid = np.int_([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right_solid = np.int_([np.transpose(np.vstack([right_fitx, ploty]))])
    cv2.polylines(road_schematic, pts_left_solid, isClosed=False, color=(40, 40, 40), thickness=4)
    cv2.polylines(road_schematic, pts_right_solid, isClosed=False, color=(40, 40, 40), thickness=4)

    # Simplified text metadata HUD
    offset_text = f"Offset: {abs(meter_offset):.2f}m " + ("Right" if meter_offset > 0 else "Left")
    cv2.putText(road_schematic, f"Lanes: {visible_lanes}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    cv2.putText(road_schematic, offset_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    
    frame_metrics = {
        "visible_lanes": visible_lanes,
        "lane_center_offset_meters": round(meter_offset, 3),
        "left_edge_relative_meters": round(left_edge_relative, 3),
        "right_edge_relative_meters": round(right_edge_relative, 3)
    }
    
    return road_schematic, frame_metrics


# --- Universal Folder Loop Execution ---
supported_extensions = ('.ts', '.mp4', '.avi', '.mkv')

if not os.path.exists(abs_input_path):
    os.makedirs(abs_input_path)
    print(f"Created empty directory '{INPUT_DIR}'. Please put your video segments or drive.mp4 inside it.")

video_files = [f for f in os.listdir(abs_input_path) if f.lower().endswith(supported_extensions)]

if video_files:
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        input_path = os.path.join(abs_input_path, video_file)
        video_output_dir = os.path.join(abs_output_path, video_name)
        
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
            
        print(f"Reading and sub-sampling: '{video_file}'...")
        cap = cv2.VideoCapture(input_path)
        
        # Calculate exactly how many frames are in the video to plan a smart step jump
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 900 # Fallback safety estimation
        
        # Determine step gap interval dynamically to extract exactly 12-15 total snapshots
        step_interval = max(total_frames // MAX_TOTAL_SNAPSHOTS, 1)
        
        frame_count = 0
        telemetry_log = {}
        
        while cap.isOpened():
            # Instruct your CPU to skip processing frames completely until it lands on a sample node
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret: break
            
            # Instantly downscale the matrix canvas size to preserve CPU resources
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                
            schematic_drawing, metrics = dynamically_extract_lane(frame)
            
            # Save artifacts
            output_png_path = os.path.join(video_output_dir, f"frame_{frame_count:05d}_schematic.png")
            cv2.imwrite(output_png_path, schematic_drawing)
            
            if len(telemetry_log) == 0:
                cv2.imwrite(os.path.join(script_dir, "output_lane_drawing.png"), schematic_drawing)
            
            telemetry_log[f"frame_{frame_count}"] = metrics
            
            # Increment by the calculated large step gap instead of reading sequentially
            frame_count += step_interval
            if len(telemetry_log) >= MAX_TOTAL_SNAPSHOTS:
                break
                
        cap.release()
        
        # Serialize generated metrics datasets smoothly
        with open(os.path.join(video_output_dir, "lane_lines.json"), "w") as f:
            json.dump(telemetry_log, f, indent=4)
        with open(os.path.join(script_dir, "lane_lines.json"), "w") as f:
            json.dump(telemetry_log, f, indent=4)
            
        print(f"Success! Generated {len(telemetry_log)} snapshots without crashing your hardware.")
else:
    print(f"No video files found inside '{INPUT_DIR}/' directory.")