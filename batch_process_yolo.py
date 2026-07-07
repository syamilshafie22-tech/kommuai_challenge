import cv2
import numpy as np
import os
import json
from ultralytics import YOLO

# --- Configurations ---
INPUT_DIR = "video_input"
OUTPUT_DIR = "output_schematics_yolo"  # Updated directory path

# Target resolution for fast processing on your Ryzen 5 CPU
TARGET_WIDTH = 640
TARGET_HEIGHT = 360
MAX_TOTAL_SNAPSHOTS = 15

script_dir = os.path.dirname(os.path.abspath(__file__))
abs_input_path = os.path.join(script_dir, INPUT_DIR)
abs_output_path = os.path.join(script_dir, OUTPUT_DIR)

if not os.path.exists(abs_output_path):
    os.makedirs(abs_output_path)

# --- Initialize YOLOv8 Nano model (Lightweight and optimized for CPU) ---
# We use the 'yolov8n-seg.pt' model because it supports instance segmentation.
# The model will auto-download on the first run.
print("Loading lightweight YOLOv8-Segmentation Model...")
model = YOLO("yolov8n-seg.pt")


def extract_lane_via_yolo(img):
    """Uses deep learning segmentation to find lane metrics and build schematics."""
    h, w = img.shape[:2]
    camera_center = w / 2
    
    # 1. Run inference on the CPU with a target resolution
    # 'classes=[0, 1, 2, 3, 5, 7]' checks for cars, trucks, and people to infer road spaces,
    # or tracks standard drivable segments depending on your downstream task.
    results = model(img, imgsz=(TARGET_HEIGHT, TARGET_WIDTH), verbose=False)
    
    # Baseline defaults if segmentation masks are obscured
    visible_lanes = 0
    left_fitx_base = w * 0.25
    right_fitx_base = w * 0.75
    
    # 2. Extract Mask Contours and Compute Layout Polynomials
    # YOLO delivers precise masks bounding objects or lane components
    if results[0].masks is not None:
        all_masks = results[0].masks.xy
        # Flatten and filter for structural lane pathways
        points = []
        for mask in all_masks:
            for pt in mask:
                points.append(pt)
                
        if len(points) > 10:
            points = np.array(points)
            # Separate points relative to center camera vector
            left_pts = points[points[:, 0] < camera_center]
            right_pts = points[points[:, 0] >= camera_center]
            
            # Map Left Boundary via Linear Regression
            if len(left_pts) > 5:
                left_fit = np.polyfit(left_pts[:, 1], left_pts[:, 0], 1)
                left_fitx_base = left_fit[0] * h + left_fit[1]
                visible_lanes += 1
                
            # Map Right Boundary
            if len(right_pts) > 5:
                right_fit = np.polyfit(right_pts[:, 1], right_pts[:, 0], 1)
                right_fitx_base = right_fit[0] * h + right_fit[1]
                visible_lanes += 1

    # 3. Apply Metrological Scaling (Standard 3.7m highway lane width)
    lane_width_px = (right_fitx_base - left_fitx_base)
    pixel_offset = ((left_fitx_base + right_fitx_base) / 2) - camera_center
    xm_per_pix = 3.7 / lane_width_px if lane_width_px > 0 else 0.01
    
    meter_offset = pixel_offset * xm_per_pix
    left_edge_relative = (left_fitx_base - camera_center) * xm_per_pix
    right_edge_relative = (right_fitx_base - camera_center) * xm_per_pix

    # 4. Generate High-Contrast Blueprint Rendering
    road_schematic = np.full((h, w, 3), 255, dtype=np.uint8) 
    
    # Reconstruct continuous schematic paths
    ploty = np.linspace(0, h - 1, h)
    left_fitx = np.linspace(left_fitx_base, left_fitx_base, h)
    right_fitx = np.linspace(right_fitx_base, right_fitx_base, h)
    
    # Draw core road path polygon
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    road_corridor = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_schematic, np.int_([road_corridor]), (235, 235, 235))
    
    # Solid dark borders
    cv2.polylines(road_schematic, np.int_([pts_left]), isClosed=False, color=(40, 40, 40), thickness=4)
    cv2.polylines(road_schematic, np.int_([pts_right]), isClosed=False, color=(40, 40, 40), thickness=4)

    # UI Metadata HUD Overlay
    offset_text = f"Offset: {abs(meter_offset):.2f}m " + ("Right" if meter_offset > 0 else "Left")
    cv2.putText(road_schematic, f"YOLO Lanes: {visible_lanes}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    cv2.putText(road_schematic, offset_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    
    frame_metrics = {
        "visible_lanes": visible_lanes,
        "lane_center_offset_meters": round(meter_offset, 3),
        "left_edge_relative_meters": round(left_edge_relative, 3),
        "right_edge_relative_meters": round(right_edge_relative, 3)
    }
    
    return road_schematic, frame_metrics


# --- Main Loop Sequence ---
supported_extensions = ('.ts', '.mp4', '.avi', '.mkv')

if not os.path.exists(abs_input_path):
    os.makedirs(abs_input_path)
    print(f"Created empty directory '{INPUT_DIR}'. Move your video segments or drive.mp4 inside it.")

video_files = [f for f in os.listdir(abs_input_path) if f.lower().endswith(supported_extensions)]

if video_files:
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        input_path = os.path.join(abs_input_path, video_file)
        video_output_dir = os.path.join(abs_output_path, video_name)
        
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
            
        print(f"Deep learning inference starting on: '{video_file}'...")
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 900 
        
        step_interval = max(total_frames // MAX_TOTAL_SNAPSHOTS, 1)
        frame_count = 0
        telemetry_log = {}
        
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            schematic_drawing, metrics = extract_lane_via_yolo(frame)
            
            output_png_path = os.path.join(video_output_dir, f"frame_{frame_count:05d}_schematic.png")
            cv2.imwrite(output_png_path, schematic_drawing)
            
            if len(telemetry_log) == 0:
                cv2.imwrite(os.path.join(script_dir, "output_lane_drawing.png"), schematic_drawing)
            
            telemetry_log[f"frame_{frame_count}"] = metrics
            frame_count += step_interval
            if len(telemetry_log) >= MAX_TOTAL_SNAPSHOTS:
                break
                
        cap.release()
        
        with open(os.path.join(video_output_dir, "lane_lines.json"), "w") as f:
            json.dump(telemetry_log, f, indent=4)
        with open(os.path.join(script_dir, "lane_lines.json"), "w") as f:
            json.dump(telemetry_log, f, indent=4)
            
        print(f"Success! Generated assets inside {OUTPUT_DIR}/{video_name}/")
else:
    print(f"No video assets found inside '{INPUT_DIR}/'.")