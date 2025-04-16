
# Mark Stations and Detect Persons

This Python script processes a video feed to identify and classify people (sitting or standing) near predefined station positions. It overlays visual markers on the video and exports frame-wise occupation data as a JSON file.

## Features

- Uses YOLO for person detection.
- Classifies people as sitting or standing based on bounding box ratio.
- Detects face orientation (front/back) using MediaPipe.
- Marks table and chair positions from a calibration CSV file.
- Tracks station occupation based on presence and duration.
- Outputs annotated video and frame-wise occupation data.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Required Python packages:**
- `opencv-python`
- `ultralytics`
- `mediapipe`
- `numpy`

Also ensure you have:
- YOLO model file (e.g., `yolo12x.pt`)
- Input video file
- Station calibration CSV file

## File Structure

```bash
project/
│
├── mark_stations_copy.py
├── input videos/
│   └── Video_data/
│       └── video6.mp4
├── SRC/
│   └── calibration_file/
│       ├── updated_station_calibration.csv
│       └── marked_video_with_persons_classified.mp4
├── yolo12x.pt
└── framewise_station_data.json  # Output JSON
```

## How to Run

1. Place your input video and calibration CSV in the appropriate folders.
2. Modify the file paths in the `__main__` section of `mark_stations_copy.py` if needed.
3. Run the script:

```bash
python mark_stations_copy.py
```

## Output

- **Video with visual annotations** at the path defined by `output_video_path`
- **JSON file** `framewise_station_data.json` with data per frame:
```json
{
  "frame_10": [
    {
      "station": "S1",
      "occupants": 1,
      "occupation Status": true
    },
    ...
  ]
}
```

## Notes

- A person is considered **sitting** if the height/width ratio of the bounding box is less than 2.
- Face orientation is marked if the person is sitting and within a certain threshold from a station.
- Stations are marked occupied if a person remains nearby for more than 20 seconds.

## License

MIT License
