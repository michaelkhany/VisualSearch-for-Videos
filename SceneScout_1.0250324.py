#!/usr/bin/env python3
import os
import json
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests
from ultralytics import YOLO  # Ensure this matches your YOLO v11 implementation

# -----------------------
# Configuration Variables
# -----------------------
MODEL_FILENAME = "yolo11n.pt"
# Official YOLO11n model from Ultralytics GitHub release (v8.3.0)
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

def download_model(model_url, model_path):
    """Download the YOLO model weights from a given URL."""
    try:
        print(f"Downloading model from {model_url}...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model download completed.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

# Check if model file exists; if not, download it.
if not os.path.exists(MODEL_FILENAME):
    download_model(MODEL_URL, MODEL_FILENAME)

# Initialize the YOLOv11 model
model = YOLO(MODEL_FILENAME)

# -----------------------
# Video Processing Functions
# -----------------------
def process_video(video_path, model, frame_skip=30, save_frames_dir=None):
    """
    Processes a single video file:
      - Extracts frames (saving them to save_frames_dir if provided).
      - Runs YOLO detection on each processed frame.
      - Records metadata: timestamp, object label, bounding box, and confidence.
    """
    video_metadata = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return video_metadata

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS is unavailable
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_skip == 0:
            timestamp = frame_count / fps  # Current timestamp in seconds
            
            # Save the current frame as an image if a directory is provided.
            if save_frames_dir:
                os.makedirs(save_frames_dir, exist_ok=True)
                frame_filename = os.path.join(save_frames_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
            
            # Run detection on the current frame.
            results = model(frame)
            # Assuming each detection is formatted as:
            # [x1, y1, x2, y2, confidence, class]
            for detection in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = detection
                label = model.names[int(cls)]
                video_metadata.append({
                    "timestamp": timestamp,
                    "object": label,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })
        frame_count += 1
        success, frame = cap.read()
    cap.release()
    return video_metadata

def process_videos_in_directory(video_dir, metadata_dir, model, frame_skip=30):
    """
    Processes all video files in the specified directory.
      - For each video, creates a folder to store its extracted frames.
      - Runs YOLO detection on the frames.
      - Saves a JSON metadata file with details of detected objects.
    Returns a log string summarizing the processing.
    """
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    log = ""
    # Create a root folder for all extracted frames.
    frames_root_dir = os.path.join(metadata_dir, "frames")
    os.makedirs(frames_root_dir, exist_ok=True)
    for filename in os.listdir(video_dir):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(video_dir, filename)
            log += f"Processing video: {video_path}\n"
            # Create a subfolder for frames for this video.
            video_name = os.path.splitext(filename)[0]
            frames_dir = os.path.join(frames_root_dir, video_name)
            metadata = process_video(video_path, model, frame_skip, save_frames_dir=frames_dir)
            metadata_filename = video_name + ".json"
            metadata_path = os.path.join(metadata_dir, metadata_filename)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            log += f"Metadata saved to: {metadata_path}\n"
            log += f"Extracted frames saved to: {frames_dir}\n"
    return log

def search_metadata(query, metadata_dir):
    """
    Searches through JSON metadata files in the metadata directory for the specified object.
    Returns a list of detections with video name, timestamp, object, bounding box, and confidence.
    """
    results_found = []
    for filename in os.listdir(metadata_dir):
        if filename.endswith(".json"):
            path = os.path.join(metadata_dir, filename)
            with open(path, "r") as f:
                data = json.load(f)
                for detection in data:
                    if query.lower() in detection["object"].lower():
                        results_found.append({
                            "video": filename.replace(".json", ""),
                            "timestamp": detection["timestamp"],
                            "object": detection["object"],
                            "bbox": detection["bbox"],
                            "confidence": detection["confidence"]
                        })
    return results_found

# -----------------------
# GUI Code using Tkinter Forms
# -----------------------
class SceneScoutGUI:
    def __init__(self, master):
        self.master = master
        master.title("SceneScout")
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True)
        self.create_process_tab()
        self.create_search_tab()

    def create_process_tab(self):
        self.process_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.process_frame, text="Process Videos")

        # Video Directory Selection
        ttk.Label(self.process_frame, text="Video Directory:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.video_dir_var = tk.StringVar()
        self.video_dir_entry = ttk.Entry(self.process_frame, textvariable=self.video_dir_var, width=50)
        self.video_dir_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.process_frame, text="Browse", command=self.browse_video_dir).grid(row=0, column=2, padx=5, pady=5)

        # Metadata Directory Selection
        ttk.Label(self.process_frame, text="Metadata Directory:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.metadata_dir_var = tk.StringVar()
        self.metadata_dir_entry = ttk.Entry(self.process_frame, textvariable=self.metadata_dir_var, width=50)
        self.metadata_dir_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.process_frame, text="Browse", command=self.browse_metadata_dir).grid(row=1, column=2, padx=5, pady=5)

        # Frame Skip Entry
        ttk.Label(self.process_frame, text="Frame Skip (process every nth frame):").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.frame_skip_var = tk.IntVar(value=30)
        self.frame_skip_entry = ttk.Entry(self.process_frame, textvariable=self.frame_skip_var, width=10)
        self.frame_skip_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        # Process Button
        self.process_button = ttk.Button(self.process_frame, text="Start Processing", command=self.start_processing)
        self.process_button.grid(row=3, column=0, columnspan=3, pady=10)

        # Log Text Box to show progress
        self.process_log = tk.Text(self.process_frame, height=10, width=80)
        self.process_log.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

    def create_search_tab(self):
        self.search_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.search_frame, text="Search Metadata")

        # Metadata Directory Selection for Search
        ttk.Label(self.search_frame, text="Metadata Directory:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.search_metadata_dir_var = tk.StringVar()
        self.search_metadata_dir_entry = ttk.Entry(self.search_frame, textvariable=self.search_metadata_dir_var, width=50)
        self.search_metadata_dir_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.search_frame, text="Browse", command=self.browse_search_metadata_dir).grid(row=0, column=2, padx=5, pady=5)

        # Object Query Entry
        ttk.Label(self.search_frame, text="Object to Search:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.search_object_var = tk.StringVar()
        self.search_object_entry = ttk.Entry(self.search_frame, textvariable=self.search_object_var, width=50)
        self.search_object_entry.grid(row=1, column=1, padx=5, pady=5)

        # Search Button
        self.search_button = ttk.Button(self.search_frame, text="Search", command=self.start_search)
        self.search_button.grid(row=2, column=0, columnspan=3, pady=10)

        # Listbox to display search results
        self.search_results = tk.Listbox(self.search_frame, width=100, height=15)
        self.search_results.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

    def browse_video_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.video_dir_var.set(directory)

    def browse_metadata_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.metadata_dir_var.set(directory)

    def browse_search_metadata_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.search_metadata_dir_var.set(directory)

    def start_processing(self):
        video_dir = self.video_dir_var.get()
        metadata_dir = self.metadata_dir_var.get()
        frame_skip = self.frame_skip_var.get()
        if not video_dir or not metadata_dir:
            messagebox.showerror("Error", "Please select both video and metadata directories.")
            return
        self.process_log.insert(tk.END, "Starting processing...\n")
        self.master.update()  # Update UI

        # Process videos (extract frames and detect objects) and log the output.
        log = process_videos_in_directory(video_dir, metadata_dir, model, frame_skip)
        self.process_log.insert(tk.END, log + "\nProcessing complete.\n")

    def start_search(self):
        metadata_dir = self.search_metadata_dir_var.get()
        query = self.search_object_var.get()
        if not metadata_dir or not query:
            messagebox.showerror("Error", "Please select a metadata directory and enter an object to search for.")
            return

        self.search_results.delete(0, tk.END)  # Clear previous results
        results = search_metadata(query, metadata_dir)
        if results:
            for res in results:
                result_str = (
                    f"Video: {res['video']}, "
                    f"Time: {res['timestamp']} sec, "
                    f"Object: {res['object']}, "
                    f"BBox: {res['bbox']}, "
                    f"Confidence: {res['confidence']:.2f}"
                )
                self.search_results.insert(tk.END, result_str)
        else:
            self.search_results.insert(tk.END, f"No occurrences of '{query}' found.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SceneScoutGUI(root)
    root.mainloop()
