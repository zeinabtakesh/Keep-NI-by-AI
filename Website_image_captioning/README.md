# AI-Powered CCTV Monitoring System

This system provides real-time CCTV monitoring with AI-powered suspicious activity detection, using the Vit-GPT2 model for image captioning and ChatGPT for analysis.

## Features

- Real-time camera feed monitoring
- Automatic video captioning every 5 seconds
- Suspicious activity detection using ChatGPT
- Automatic footage saving for suspicious activities
- Daily report generation
- Historical event querying
- Metadata storage for future reference

## Prerequisites

- Python 3.8 or higher
- Webcam or CCTV camera
- OpenAI API key

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Live Monitoring
To start the live monitoring system:
```bash
python main.py --mode monitor
```
Press 'q' to quit the monitoring.

### Generate Daily Report
To generate a summary report of the day's events:
```bash
python main.py --mode report
```

### Query Past Events
To search for specific events in the past:
```bash
python main.py --mode query --query "your search query here"
```

Example queries:
- "When did someone with a red jacket appear?"
- "Show me all suspicious activities from yesterday"
- "Find footage of people entering through the back door"

## Data Storage

The system stores data in the following structure:
- `data/captions.csv`: Contains all captions and analysis results
- `data/metadata.txt`: Detailed metadata for all events
- `data/footage/`: Directory containing saved footage videos
- `data/daily_report.txt`: Generated daily reports

## Notes

- The system processes every 5 second video footage to reduce computational load
- Footage is automatically saved when suspicious activity is detected
- All data is stored locally for future reference
- The system uses ChatGPT for both suspicious activity detection and report generation 
