# VisualSearch for Videos: SceneScout

SceneScout is a Python-based application for object detection and scene search in videos using YOLO. It allows you to extract metadata (such as timestamps, object labels, bounding boxes, and confidence scores) from videos and later search for specific objects across those videos.

The project is available in two versions:
- **CLI Version:** A command-line interface for processing videos and searching metadata.
- **GUI Version:** A user-friendly Tkinter-based graphical interface for the same tasks.

## Repository Structure [with two instance videos]

```
D:.
│   requirements.txt
│   SceneScout_0.0250324_cli.py       # CLI version of SceneScout
│   SceneScout_1.0250324.py           # GUI version of SceneScout
│   yolo11n.pt                      # YOLO model weights file (will be auto-downloaded if missing)
│
├── metadata                        # Folder for generated metadata JSON files
│   │   Living rooms with calm interiors _ One-minute videos _ Dezeen.json
│   │   NEVER TOO SMALL Melbourne Hotel Small Apartment Conversion - 50sqm_538sqft.json
│   │
│   └── frames                      # Folder for extracted video frames (organized by video)
│       ├── Living rooms with calm interiors _ One-minute videos _ Dezeen
│       │       frame_000000.jpg
│       │       frame_000030.jpg
│       │       ...
│       └── NEVER TOO SMALL Melbourne Hotel Small Apartment Conversion - 50sqm_538sqft
│               frame_000000.jpg
│               frame_000030.jpg
│               ...
│
└── videos                          # Folder for source video files
        Living rooms with calm interiors _ One-minute videos _ Dezeen.mp4
        NEVER TOO SMALL Melbourne Hotel Small Apartment Conversion - 50sqm_538sqft.mp4
```

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/michaelkhany/VisualSearch-for-Videos.git
   cd VisualSearch-for-Videos
   ```

2. **Install Dependencies**

   Install the required Python packages using the provided `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include (at a minimum) the following packages:
   - `opencv-python`
   - `ultralytics`
   - `requests`
   - `tkinter` (usually included with Python)

## Usage

### Automatic Model Download

If the YOLO model file (`yolo11n.pt`) is not found in the repository root, the scripts will automatically download it from a configured URL. Ensure you have a stable internet connection when running the scripts for the first time.

### CLI Version

Use the CLI version for a quick and scriptable experience.

- **Processing Videos**

  This command processes all videos in the `videos` directory, runs object detection on every nth frame (default is 30), and saves the metadata JSON files in the `metadata` directory.

  ```bash
  python SceneScout_0.0250324_cli.py process --video_dir videos --metadata_dir metadata --frame_skip 30
  ```

- **Searching Metadata**

  To search for a specific object (for example, "car") in the metadata, use:

  ```bash
  python SceneScout_0.0250324_cli.py search --metadata_dir metadata --object car
  ```

### GUI Version

The GUI version provides a more interactive experience with two tabs: one for processing videos and one for searching metadata.

- **Launching the GUI**

  Simply run:

  ```bash
  python SceneScout_1.0250324.py
  ```

  The GUI window will open with:
  
  - A **Process Videos** tab where you can browse for your video and metadata directories, adjust the frame skip value, and start processing.
  - A **Search Metadata** tab where you can browse for your metadata directory, enter the object name to search, and view the search results.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License.

## Repository Links

- **GitLab Repository:** [VisualSearch-for-Videos](https://github.com/michaelkhany/VisualSearch-for-Videos/tree/main)
