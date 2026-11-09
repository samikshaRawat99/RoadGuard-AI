# RoadGuard AI 🚗🛡️

An AI-powered Road Safety and Vehicle Monitoring System built with Ultralytics YOLO, OpenCV, Roboflow Supervision, Streamlit, Pandas, and Plotly.

## Project Structure

```text
roadguard-ai/
├── app.py                     # Streamlit web application entry point
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── src/                       # Core modules
│   ├── __init__.py
│   ├── detection.py           # Object & vehicle detection (YOLO)
│   ├── tracking.py            # Vehicle tracking & unique ID assignment
│   ├── violations.py          # Traffic violation detection (speeding, lane departure, etc.)
│   ├── speed_estimation.py    # Speed calculation based on pixel displacement
│   └── analytics.py           # Data aggregation and Plotly visualizations
├── models/                    # Pretrained YOLO model weights (.pt files)
├── data/
│   ├── videos/                # Sample input video clips
│   └── images/                # Sample input test images
├── outputs/
│   ├── processed_videos/      # Output videos with detection overlays
│   └── violations/            # Cropped snapshots and violation logs
└── assets/                    # Static assets, logos, and UI icons
```

## Setup & Installation

```bash
# 1. Create a virtual environment (Python 3.10 - 3.12 recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit application
streamlit run app.py
```

