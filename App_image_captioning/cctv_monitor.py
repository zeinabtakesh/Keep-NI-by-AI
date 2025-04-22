import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import openai
import json
from pathlib import Path
from openai import OpenAI
from threading import Thread
import time
import threading
from PIL import Image

# Import the new ImageInferenceEngine instead of VideoInferenceEngine
from inference import ImageInferenceEngine


class CCTVMonitor:
    def __init__(self, storage_path=None):
        # Environment configuration
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '10000'

        # Load environment variables
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in .env file")

        # Storage configuration - either use provided path or default to user's home directory
        if storage_path:
            self.base_storage_path = Path(storage_path)
            # If relative path is provided, make it absolute
            if not self.base_storage_path.is_absolute():
                # Get the current working directory to resolve the relative path
                cwd = Path.cwd()
                self.base_storage_path = cwd / self.base_storage_path
        else:
            self.base_storage_path = Path.home() / "CCTV_Monitoring"

        print(f"[INIT] Base storage path: {self.base_storage_path}")
        self._create_storage_directories()

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)

        # Initialize image inference engine
        self.inference_engine = ImageInferenceEngine()

        # Initialize camera capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"[INIT] Using FPS: {self.fps}")

        # Frame counters and settings
        self.frame_interval = 100  # Process captioning every 100 frames
        self.frame_counter = 0  # Counter for frame buffering

        # Data tracking (log captions/analysis)
        self.captions_df = pd.DataFrame(columns=[
            'timestamp', 'camera_name', 'caption',
            'is_suspicious', 'reason', 'confidence',
            'image_path'
        ])

        # Load existing data if available
        self._load_existing_data()

    def _create_storage_directories(self):
        """Create required storage directories."""
        try:
            # Create main data directory
            self.data_dir = self.base_storage_path / "data"
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Create images directory instead of footage
            self.footage_dir = self.data_dir / "images"
            self.footage_dir.mkdir(parents=True, exist_ok=True)

            print(f"[STORAGE] Using storage path: {self.footage_dir.resolve()}")

            # Verify we can write to the directories
            test_file = self.data_dir / "test_write.txt"
            try:
                with open(test_file, 'w') as f:
                    f.write("Test write")
                os.remove(test_file)
            except Exception as e:
                print(f"[WARNING] Directory is not writable: {self.data_dir}")
                print(f"[WARNING] Error: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Failed to create storage directories: {str(e)}")

    def _load_existing_data(self):
        """Load existing data from CSV if available."""
        try:
            captions_csv = self.data_dir / "captions.csv"
            if captions_csv.exists():
                loaded_df = pd.read_csv(captions_csv)
                if not loaded_df.empty:
                    self.captions_df = loaded_df
                    print(f"[DATA] Loaded {len(self.captions_df)} existing records")
        except Exception as e:
            print(f"[WARNING] Failed to load existing data: {str(e)}")

    def _get_image_path(self, timestamp):
        """Generate image path based on timestamp."""
        # Ensure the footage directory exists
        if not self.footage_dir.exists():
            self.footage_dir.mkdir(parents=True, exist_ok=True)

        filename = f"image_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        return self.footage_dir / filename

    def save_image(self, frame, timestamp):
        """Save a single frame as an image."""
        try:
            image_path = self._get_image_path(timestamp)
            print(f"[SAVING] Saving image to: {image_path}")

            # Convert frame to BGR if it's RGB
            if frame.shape[2] == 3:  # If it's a 3-channel image
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = frame

            # Save image
            cv2.imwrite(str(image_path), bgr_frame)

            # Verify file was written
            if image_path.exists() and image_path.stat().st_size > 0:
                print(f"[SUCCESS] Saved image to {image_path}")
                return image_path
            else:
                print(f"[ERROR] Failed to save image to {image_path}")
                return None
        except Exception as e:
            print(f"[ERROR] Exception during image saving: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def on_caption_ready(self, image_id, caption):
        """Callback for when an image caption is ready."""
        if not caption or caption.startswith("Error"):
            print(f"[ERROR] Caption failed for {image_id}: {caption}")
            return

        # Extract timestamp from image_id
        # The image_id should be in format "image_YYYYMMDD_HHMMSS"
        try:
            timestamp_str = image_id.split('_', 1)[1]
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')

            # Get image path from saved images
            image_path = self._get_image_path(timestamp)

            print(f"[CAPTION] Generated caption for {image_id}: {caption}")

            # Analyze the caption
            analysis = self.analyze_suspicious_activity(caption)

            # Add to dataframe
            new_row = pd.DataFrame([{
                'timestamp': timestamp,
                'camera_name': 'Camera 1',
                'caption': caption,
                'is_suspicious': analysis.get('is_suspicious', False),
                'reason': analysis.get('reason', 'Normal activity'),
                'confidence': analysis.get('confidence', 0.0),
                'image_path': str(image_path)
            }])

            # Add the row and immediately save to disk
            self.captions_df = pd.concat([self.captions_df, new_row], ignore_index=True)
            self.save_data()

            is_suspicious = analysis.get('is_suspicious', False)
            print(f"[ANALYSIS] {'Suspicious' if is_suspicious else 'Normal'} activity detected: {caption}")

        except Exception as e:
            print(f"[ERROR] Processing caption callback: {str(e)}")
            import traceback
            traceback.print_exc()

    def analyze_suspicious_activity(self, caption):
        """Send caption to OpenAI Chat API to analyze suspicious activity."""
        prompt = f"""Analyze this CCTV caption and determine if it describes any suspicious activity:
Caption: {caption}

Respond with a JSON object containing:
- is_suspicious: boolean
- reason: string (if suspicious)
- confidence: float (0-1)
"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"[ERROR] Failed to parse analysis response: {e}")
            return {'is_suspicious': False, 'reason': 'Parsing error', 'confidence': 0.0}

    def process_image_directly(self, frame):
        """Process a frame directly without saving to file first."""
        try:
            timestamp = datetime.now()

            # Convert BGR to RGB for the inference engine if needed
            if frame.shape[2] == 3:  # If it's a 3-channel image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame

            # Save image to disk
            image_path = self.save_image(frame, timestamp)
            if not image_path:
                return None, None

            # Convert to PIL Image for the captioning model
            pil_image = Image.fromarray(rgb_frame)

            # Generate caption directly (blocking call)
            caption = self.inference_engine.caption_image(pil_image)
            print(f"[DIRECT_CAPTION] Generated: {caption}")

            # Skip GPT analysis for empty or error captions
            if caption.startswith("Error") or not caption:
                analysis = {'is_suspicious': False, 'reason': 'No valid caption', 'confidence': 0.0}
            else:
                # Analyze the caption
                analysis = self.analyze_suspicious_activity(caption)

            # Save the results immediately
            new_row = pd.DataFrame([{
                'timestamp': timestamp,
                'camera_name': 'Camera 1',
                'caption': caption,
                'is_suspicious': analysis.get('is_suspicious', False),
                'reason': analysis.get('reason', 'Normal activity'),
                'confidence': analysis.get('confidence', 0.0),
                'image_path': str(image_path)
            }])

            # Add the row and immediately save to disk
            self.captions_df = pd.concat([self.captions_df, new_row], ignore_index=True)
            self.save_data()

            return caption, analysis
        except Exception as e:
            print(f"[ERROR] Frame processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Error processing frame", {'is_suspicious': False}

    def run(self):
        """Main loop for capturing frames and processing them."""
        self.frame_counter = 0

        try:
            print("[INFO] Starting camera capture...")
            while True:
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to capture frame")
                    break

                # Increment frame counter
                self.frame_counter += 1

                # Process every frame_interval frames
                if self.frame_counter % self.frame_interval == 0:
                    print(f"[INFO] Processing frame {self.frame_counter}")

                    # Process the frame directly
                    caption, analysis = self.process_image_directly(frame)

                    if caption and analysis:
                        is_suspicious = analysis.get('is_suspicious', False)
                        confidence = analysis.get('confidence', 0.0)

                        # Display caption and analysis info on the frame
                        status = "SUSPICIOUS" if is_suspicious else "Normal"
                        cv2.putText(frame, f"Caption: {caption[:50]}...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Status: {status} ({confidence:.2f})", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255) if is_suspicious else (0, 255, 0), 2)

                # Display the resulting frame
                cv2.imshow('CCTV Monitor', frame)

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] Exception during monitoring: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Release the capture and close windows
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Camera released and windows closed")

    def save_data(self):
        """Persist the collected captions and analysis data."""
        try:
            # Make sure the data directory exists
            if not self.data_dir.exists():
                self.data_dir.mkdir(parents=True, exist_ok=True)

            # Save CSV data
            captions_csv = self.data_dir / "captions.csv"
            self.captions_df.to_csv(captions_csv, index=False)

            # Verify the file was written
            if captions_csv.exists() and captions_csv.stat().st_size > 0:
                print(f"[DATA] Saved {len(self.captions_df)} records to {captions_csv}")
            else:
                print(f"[ERROR] Failed to write data to {captions_csv}")

            # Also save a simple metadata text file
            metadata_file = self.data_dir / "metadata.txt"
            with open(metadata_file, "w") as f:
                for _, row in self.captions_df.iterrows():
                    f.write(f"Time: {row['timestamp']}\n")
                    f.write(f"Camera: {row['camera_name']}\n")
                    f.write(f"Caption: {row['caption']}\n")
                    f.write(f"Suspicious: {row['is_suspicious']}\n")
                    f.write(f"Reason: {row['reason']}\n")
                    f.write(f"Image: {row['image_path']}\n")
                    f.write("-" * 50 + "\n")
                f.flush()
                os.fsync(f.fileno())  # Ensure it's written to disk

        except Exception as e:
            print(f"[ERROR] Failed to save data: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_report(self):
        """Generate and save a summary report for today's activity using ChatGPT."""
        try:
            if self.captions_df.empty:
                return "No data found. No alerts to summarize."

            # Get today's date only
            today = datetime.now().date()

            # Filter DataFrame to keep only today’s alerts
            self.captions_df['timestamp'] = pd.to_datetime(self.captions_df['timestamp'])
            today_df = self.captions_df[self.captions_df['timestamp'].dt.date == today]

            if today_df.empty:
                return "No alerts were logged today."

            # Generate metadata string from today’s alerts
            metadata = ""
            for _, row in today_df.iterrows():
                metadata += f"Time: {row['timestamp']}\n"
                metadata += f"Camera: {row['camera_name']}\n"
                metadata += f"Caption: {row['caption']}\n"
                metadata += f"Suspicious: {row['is_suspicious']}\n"
                metadata += f"Reason: {row['reason']}\n"
                metadata += "-" * 50 + "\n"

            # Optional truncation if still too long
            if len(metadata) > 10000:
                metadata = metadata[-10000:]

            # Generate report prompt
            prompt = f"""Generate a summary report of today's CCTV monitoring data:

    {metadata}

    Please include:
    1. Total number of suspicious activities
    2. Timeline of key events
    3. Most common types of suspicious behavior
    4. Recommendations for improving security
    """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            report = response.choices[0].message.content

            with open(self.data_dir / "daily_report.txt", "w", encoding="utf-8") as f:
                f.write(report)

            print("[REPORT] Daily report generated successfully.")
            return report

        except Exception as e:
            print(f"[ERROR] Daily report generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Error generating report. Please check logs."

    def query_events(self, query):
        """Query logged events using ChatGPT API."""
        try:
            # Load metadata text file
            with open(self.data_dir / "metadata.txt", "r", encoding="utf-8") as f:
                metadata = f.read()

            # Truncate if metadata is too long (safe buffer for GPT-3.5 token limit)
            if len(metadata) > 24000:
                metadata = metadata[-24000:]  # Keep recent logs only

            # Improved prompt with smart instructions
            prompt = f"""
    You are an intelligent CCTV analyst reviewing camera logs.

    Below is a collection of CCTV metadata logs from various cameras.

    Each entry includes:
    - A timestamp
    - The camera name
    - A description (caption)
    - A flag for suspicious activity
    - A reason if suspicious activity occurred

    ------------------------
    USER QUERY:
    "{query}"
    ------------------------
    
    TASK:
    1. Search through the logs and extract only the entries that directly match the user’s query.
    2. Look for connections in time, people, behavior, or location.
    3. Treat terms like “suspicious activity” and “suspicious behavior” as equivalent.
    4. Interpret “today” as matching the date found in the logs if unspecified.
    5. Only respond with relevant matches — no summaries or hallucinations.
    6. Asking about suspicious activities, means asking about all events that contains Suspicious: True, so only return the corresponding captions as bullets points and between parenthesis indicate the timestamp and camera).
    7. If yes, no question: answer first by yes or no.
    
    LOGS:
    {metadata}
    """

            # Make the call to OpenAI's chat API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful security analyst."},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[ERROR] Chat query failed: {e}")
            return "Sorry, I couldn't process your question right now due to system limitations."

    #
#     def generate_report(self):
#         """Generate and save a summary report via ChatGPT API based on logged metadata."""
#         try:
#             with open(self.data_dir / "metadata.txt", "r") as f:
#                 metadata = f.read()
#             prompt = f"""Generate a summary report of the CCTV monitoring data:
#
# {metadata}
#
# Please provide:
# 1. Total number of suspicious activities
# 2. Timeline of events
# 3. Most common types of suspicious activities
# 4. Recommendations for security improvements
# """
#             response = self.client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": prompt}]
#             )
#             report = response.choices[0].message.content
#             with open(self.data_dir / "daily_report.txt", "w") as f:
#                 f.write(report)
#             return report
#         except Exception as e:
#             print(f"[ERROR] Report generation failed: {str(e)}")
#             return "Error generating report. Please check logs."

    def query_events(self, query):
        """Query logged events using ChatGPT API."""
        with open(self.data_dir / "metadata.txt", "r") as f:
            metadata = f.read()
        prompt = f"""Based on the following CCTV metadata, answer this query: {query}

{metadata}

Please provide specific timestamps and locations for any matching events.
"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def set_storage_path(self, new_path):
        """Change the base storage path at runtime."""
        self.base_storage_path = Path(new_path)
        # If relative path is provided, make it absolute
        if not self.base_storage_path.is_absolute():
            cwd = Path.cwd()
            self.base_storage_path = cwd / self.base_storage_path
        self._create_storage_directories()

    def get_current_storage_path(self):
        """Return the current storage path as a string."""
        return str(self.footage_dir.resolve())


if __name__ == "__main__":
    # Optionally specify a custom storage path
    custom_path = r"C:\Users\Lenovo\OneDrive\Desktop\cctv"
    monitor = CCTVMonitor(storage_path=custom_path)
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping monitoring...")
    finally:
        monitor.save_data()
        report = monitor.generate_report()
        print("[INFO] Daily report generated.")
        print(f"[COMPLETE] Final storage location: {monitor.get_current_storage_path()}")
        print(f"[REPORT] {report}")
        print("[INFO] CCTV monitoring session ended.")