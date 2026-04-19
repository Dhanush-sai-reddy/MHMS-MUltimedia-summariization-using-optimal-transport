import os
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
import cv2
import glob
import numpy as np
from sklearn.cluster import KMeans

def variance_of_laplacian(image):
    """
    Computes the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian.
    Higher variance means the image is sharper.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_summary_for_segment(video_path, save_dir, save_fname, n_clusters=3):
    """
    Implements the Unsupervised Visual Summarization from the paper:
    1. Extracts frames from the segment.
    2. Clusters the frames using image histograms via K-Means.
    3. Selects the best frame from the clusters based on highest variance of laplacian.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    histograms = []
    
    max_retries = 20
    retry_count = 0
    total_frames_target = 100 # limit frames to keep memory low and speed up
    frame_count = 0
    
    while len(frames) < total_frames_target:
        ret, frame = cap.read()
        if not ret:
            retry_count += 1
            if retry_count > max_retries:
                break
            continue
            
        retry_count = 0 # reset on success
        frame_count += 1
        
        # Skip some frames to move faster through the segment
        if frame_count % 5 != 0:
            continue

        try:
            # Resize to process histogram generation faster
            small_frame = cv2.resize(frame, (320, 240))
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            
            # Compute color histogram
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            frames.append(frame)
            histograms.append(hist)
        except Exception:
            continue
        
    cap.release()
    
    if not frames:
        return
        
    # If video segment is too short, just save the first frame
    if len(frames) < n_clusters:
        cv2.imwrite(os.path.join(save_dir, save_fname), frames[0])
        return

    # 1. K-Means clustering of frames by image histogram
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(histograms)
    
    # 2. To get a single representative summary frame for the segment, we pick the most common cluster
    counts = np.bincount(labels)
    largest_cluster = np.argmax(counts)
    
    best_frame = None
    max_laplacian = -1
    
    # 3. Select frame inside cluster with the max variance of laplacian (sharpest image)
    for i, label in enumerate(labels):
        if label == largest_cluster:
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            lap = variance_of_laplacian(gray)
            if lap > max_laplacian:
                max_laplacian = lap
                best_frame = frames[i]
                
    if best_frame is not None:
        cv2.imwrite(os.path.join(save_dir, save_fname), best_frame)

def main():
    base_dir = "cnn_data"
    if not os.path.exists(base_dir):
        print(f"Cannot find {base_dir} folder.")
        return
        
    for case_folder in os.listdir(base_dir):
        case_path = os.path.join(base_dir, case_folder)
        video_dir = os.path.join(case_path, "video")
        if os.path.isdir(video_dir):
            ts_files = glob.glob(os.path.join(video_dir, "*.ts"))
            
            for ts_file in ts_files:
                basename = os.path.basename(ts_file)
                name, _ = os.path.splitext(basename)
                
                # e.g. segment1.ts -> segment1_summary.jpg
                save_fname = f"{name}_summary.jpg"
                
                # Check if it already exists to skip
                if os.path.exists(os.path.join(case_path, save_fname)):
                    print(f"Skipping {case_folder}/{basename} - summary already exists.")
                    continue
                
                try:
                    print(f"Processing {case_folder}/{basename}...")
                    extract_summary_for_segment(ts_file, case_path, save_fname)
                    print(f"Generated visual summary for {case_folder}/{basename} -> {save_fname}")
                except Exception as e:
                    print(f"Error processing {ts_file}: {e}")

if __name__ == "__main__":
    main()