#!/usr/bin/env python3
import os
import cv2
import json
import argparse
from ultralytics import YOLO  # Assumes a YOLO implementation similar to YOLO v11

def process_video(video_path, model, frame_skip=30):
    """
    Processes a single video file:
      - Reads frames (skipping some for speed).
      - Runs object detection.
      - Records metadata: timestamp, detected object, bounding box, and confidence.
    """
    video_metadata = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return video_metadata

    # Get frames per second to calculate timestamp
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Fallback to 30 if fps not available
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_skip == 0:
            timestamp = frame_count / fps  # Time in seconds
            # Run detection on the current frame
            results = model(frame)
            # Process each detection (assuming each detection is in the format:
            # [x1, y1, x2, y2, confidence, class])
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
    Processes all video files in the specified directory and saves their metadata.
    """
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    for filename in os.listdir(video_dir):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(video_dir, filename)
            print(f"Processing video: {video_path}")
            metadata = process_video(video_path, model, frame_skip)
            metadata_filename = os.path.splitext(filename)[0] + ".json"
            metadata_path = os.path.join(metadata_dir, metadata_filename)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata saved to: {metadata_path}")

def search_metadata(query, metadata_dir):
    """
    Searches through JSON metadata files in the metadata directory for the specified object.
    Returns a list of occurrences with video name, timestamp, bounding box, and confidence.
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

def main():
    parser = argparse.ArgumentParser(
        description="YOLO v11 Video Object Detection and Metadata Search"
    )
    parser.add_argument("mode", choices=["process", "search"], help="Mode: process videos or search metadata")
    parser.add_argument("--video_dir", type=str, default="videos", help="Directory containing videos")
    parser.add_argument("--metadata_dir", type=str, default="metadata", help="Directory to save/load metadata")
    parser.add_argument("--frame_skip", type=int, default=30, help="Process every nth frame (default: 30)")
    parser.add_argument("--object", type=str, help="Object to search for in metadata (for search mode)")
    args = parser.parse_args()

    # Initialize the YOLO v11 model.
    # This line will download the weights if they are not present locally.
    model = YOLO("yolov11.pt")  # Replace with your actual model file if necessary

    if args.mode == "process":
        process_videos_in_directory(args.video_dir, args.metadata_dir, model, args.frame_skip)
    elif args.mode == "search":
        if not args.object:
            print("Please specify an object to search for using --object")
            return
        results = search_metadata(args.object, args.metadata_dir)
        if results:
            print(f"Found occurrences of '{args.object}':")
            for res in results:
                print(f"Video: {res['video']}, Time: {res['timestamp']} sec, "
                      f"Object: {res['object']}, BBox: {res['bbox']}, "
                      f"Confidence: {res['confidence']:.2f}")
        else:
            print(f"No occurrences of '{args.object}' found in the metadata.")

if __name__ == "__main__":
    main()
