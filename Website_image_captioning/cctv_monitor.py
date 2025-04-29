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
        model_path = os.getenv("MODEL_PATH", "./NourFakih-Vit-GPT2-UCA-UCF-07")
        
        # Check if model files exist and have all necessary components
        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"]
        missing_files = []
        
        if not os.path.exists(model_path):
            print(f"[WARNING] Model directory not found at {model_path}")
            missing_files = required_files
        else:
            # Check for specific required files
            for file in required_files:
                if not os.path.exists(os.path.join(model_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"[WARNING] Missing required model files in {model_path}: {', '.join(missing_files)}")
        
        # Download model if any required files are missing
        if missing_files:
            print(f"[INFO] Downloading model from Hugging Face...")
            try:
                from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
                
                # Define the Hugging Face model ID
                hf_model_id = "NourFakih/Vit-GPT2-UCA-UCF-07"
                
                # Create the directory if it doesn't exist
                os.makedirs(model_path, exist_ok=True)
                
                # Download model components directly
                model = VisionEncoderDecoderModel.from_pretrained(hf_model_id)
                processor = ViTImageProcessor.from_pretrained(hf_model_id)
                tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
                
                # Save components locally
                model.save_pretrained(model_path)
                processor.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                
                print(f"[SUCCESS] Model downloaded and saved to {model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to download model: {str(e)}")
                print("[FALLBACK] Using Hugging Face model directly...")
                model_path = "NourFakih/Vit-GPT2-UCA-UCF-07"
        else:
            print(f"[INFO] Using existing model at {model_path}")
        
        self.inference_engine = ImageInferenceEngine(model_path=model_path)
        
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

    def process_uploaded_image(self, image_path, camera_name="Uploaded Image"):
        """Process an uploaded image file.
        
        Args:
            image_path: Path to the uploaded image
            camera_name: Name/source of the image
            
        Returns:
            Caption and analysis results
        """
        try:
            timestamp = datetime.now()
            
            # Load the image using PIL
            pil_image = Image.open(image_path)
            
            # Generate caption
            caption = self.inference_engine.caption_image(pil_image)
            print(f"[CAPTION] Generated for {image_path}: {caption}")
            
            # Skip analysis for empty or error captions
            if caption.startswith("Error") or not caption:
                analysis = {'is_suspicious': False, 'reason': 'No valid caption', 'confidence': 0.0}
            else:
                # Analyze the caption with OpenAI
                analysis = self.analyze_suspicious_activity(caption)
            
            # Save to dataframe
            new_row = pd.DataFrame([{
                'timestamp': timestamp,
                'camera_name': camera_name,
                'caption': caption,
                'is_suspicious': analysis.get('is_suspicious', False),
                'reason': analysis.get('reason', 'Normal activity'),
                'confidence': analysis.get('confidence', 0.0),
                'image_path': str(image_path)
            }])
            
            # Add to dataframe and save
            self.captions_df = pd.concat([self.captions_df, new_row], ignore_index=True)
            self.save_data()
            
            return caption, analysis
            
        except Exception as e:
            print(f"[ERROR] Failed to process uploaded image: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Error processing image", {'is_suspicious': False, 'reason': str(e), 'confidence': 0.0}

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

                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                ],
                temperature=0,
                top_p=1,
                seed=1234  # Set a fixed seed for deterministic results
        )
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"[ERROR] Failed to parse analysis response: {e}")
            return {'is_suspicious': False, 'reason': 'Parsing error', 'confidence': 0.0}

    def save_data(self):
        """Persist the collected captions and analysis data."""
        try:
            # Make sure the data directory exists
            if not self.data_dir.exists():
                self.data_dir.mkdir(parents=True, exist_ok=True)

            # Save CSV data
            captions_csv = self.data_dir / "captions.csv"
            self.captions_df.to_csv(captions_csv, index=False)

            if captions_csv.exists() and captions_csv.stat().st_size > 0:
                print(f"[DATA] Saved {len(self.captions_df)} records to {captions_csv}")
            else:
                print(f"[ERROR] Failed to write data to {captions_csv}")

            # Save metadata.txt for chat/report use
            metadata_file = self.data_dir / "metadata.txt"
            with open(metadata_file, "w", encoding="utf-8") as f:
                for _, row in self.captions_df.iterrows():
                    f.write(f"Time: {row['timestamp']}\n")
                    f.write(f"Camera: {row['camera_name']}\n")
                    f.write(f"Caption: {row['caption']}\n")
                    f.write(f"Suspicious: {row['is_suspicious']}\n")
                    f.write(f"Reason: {row['reason']}\n")
                    f.write(f"Image: {row['image_path']}\n")
                    f.write("-" * 50 + "\n")
                f.flush()
                os.fsync(f.fileno())

            # ✅ NEW: Save alerts.json in static/ for global buzz polling
            static_dir = Path("static")  # <- This ensures Flask can serve it
            static_dir.mkdir(exist_ok=True)
            alerts_json_path = static_dir / "alerts.json"
            with open(alerts_json_path, "w", encoding="utf-8") as f:
                json.dump(self.captions_df.to_dict(orient="records"), f, indent=2, default=str)
            print(f"[DATA] alerts.json updated at: {alerts_json_path.resolve()}")

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

                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                ],
                temperature=0,
                top_p=1,
                seed=1234  # Set a fixed seed for deterministic results
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
            with open(self.data_dir / "metadata.txt", "r", encoding="utf-8") as f:
                metadata = f.read()

            if len(metadata) > 24000:
                metadata = metadata[-24000:]

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
    6. Asking about suspicious activities means showing all events with `Suspicious: True`. Return captions as bullet points with timestamps and camera.
    7. If it's a yes/no question, respond with yes/no and evidence.

    LOGS:
    {metadata}
    """

            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
                top_p=1,
                seed=1234
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[ERROR] Chat query failed: {e}")
            return "Sorry, I couldn't process your question right now due to system limitations."


if __name__ == "__main__":
    # Example of processing an uploaded image
    monitor = CCTVMonitor()
    
    # Example: Process a test image if available
    test_image = "static/test_image.jpg"
    if os.path.exists(test_image):
        caption, analysis = monitor.process_uploaded_image(test_image, "Test Image")
        print(f"Caption: {caption}")
        print(f"Analysis: {analysis}")
    
    # Generate report
    report = monitor.generate_report()
    print(f"Report: {report[:200]}...")
