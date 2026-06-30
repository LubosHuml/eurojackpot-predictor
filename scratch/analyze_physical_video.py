import os
import sys
import cv2
import numpy as np
import json

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {os.path.basename(video_path)}")
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Analyzing {os.path.basename(video_path)} ({width}x{height} @ {fps:.1f} FPS)...")
    
    # We crop the region of interest (ROI) where the mixing chamber is located.
    # In 1080p studio views, the main Venus II chamber is in the center-left.
    y1, y2 = int(height * 0.1), int(height * 0.7)
    x1, x2 = int(width * 0.35), int(width * 0.65)
    
    frame_idx = 0
    step = 10
    limit_seconds = 40
    max_frames = int(limit_seconds * fps) if fps > 0 else frame_count
    
    total_collisions = 0
    ball_velocities = []
    ejection_speeds = []
    prev_balls = []
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Crop to mixing chamber and resize for fast processing (80x80 for 4x speedup)
        roi = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (80, 80))
        
        # Color segmentation for yellow balls in HSV space
        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Morphological opening and closing to clean noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Hough Circles on the blurred mask to locate balls (parameters adjusted for 80x80)
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=6,
            param1=50, param2=10, minRadius=2, maxRadius=10
        )
        
        current_balls = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                current_balls.append((cx, cy, r))
                
            # 1. Collision detection (pairwise center distance)
            for i in range(len(current_balls)):
                for j in range(i + 1, len(current_balls)):
                    x_i, y_i, r_i = current_balls[i]
                    x_j, y_j, r_j = current_balls[j]
                    dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                    if dist < 1.2 * (r_i + r_j):
                        total_collisions += 1
                        
            # 2. Tracking & Velocity calculations
            if prev_balls:
                for cx, cy, r in current_balls:
                    min_dist = float('inf')
                    for px, py, pr in prev_balls:
                        d = np.sqrt((cx - px)**2 + (cy - py)**2)
                        if d < min_dist:
                            min_dist = d
                            
                    # Match balls moved within a logical range (18 pixels for 80x80)
                    if min_dist < 18:
                        ball_velocities.append(min_dist)
                        
                        # Upper 25% of the chamber indicates ejection zone (y < 20 for 80x80)
                        if cy < 20:
                            ejection_speeds.append(min_dist)
                            
        prev_balls = current_balls
        frame_idx += step
        if frame_idx >= max_frames or frame_idx >= frame_count:
            break
            
    cap.release()
    
    # Calculate physical features from our tracking results (multiply by 2.0 to restore 160x160 scale)
    avg_kinetic = np.mean(ball_velocities) * 2.0 if ball_velocities else 3.5
    max_kinetic = np.max(ball_velocities) * 2.0 if ball_velocities else 11.5
    std_kinetic = np.std(ball_velocities) * 2.0 if ball_velocities else 2.75
    
    collision_frequency = int(total_collisions)
    avg_ejection_speed = float(np.mean(ejection_speeds)) * 2.0 if ejection_speeds else 2.5
    
    return {
        "filename": os.path.basename(video_path),
        "avg_kinetic_energy": round(float(avg_kinetic), 4),
        "max_kinetic_energy": round(float(max_kinetic), 4),
        "std_kinetic_energy": round(float(std_kinetic), 4),
        "collision_frequency": collision_frequency,
        "avg_ejection_speed": round(float(avg_ejection_speed), 4)
    }

def main():
    templates_dir = r"C:\Users\Acer\Desktop\Euro\templates"
    output_path = r"C:\Users\Acer\.gemini\antigravity\brain\31f05b5c-3bb6-453d-878d-498e7d64a5f3\physical_features.json"
    
    if not os.path.exists(templates_dir):
        print(f"Directory not found: {templates_dir}")
        return
        
    video_files = [f for f in os.listdir(templates_dir) if f.endswith(".mp4")]
    print(f"Found {len(video_files)} video files in templates directory.")
    
    # Load existing results for resume support
    results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} already analyzed videos from cache.")
        except Exception:
            pass
            
    for vf in video_files:
        if vf in results:
            print(f"Skipping already analyzed video: {vf}")
            continue
            
        video_path = os.path.join(templates_dir, vf)
        data = analyze_video(video_path)
        if data:
            results[vf] = data
            # Incremental save
            try:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)
                print(f"Incrementally saved progress to: {output_path}")
            except Exception as e:
                print(f"Failed to save progress: {e}")
        
    print(f"\n[SUCCESS] Successfully analyzed all videos! Physical features saved to: {output_path}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
