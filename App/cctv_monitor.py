import cv2
import torch
import av
import numpy as np
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
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
from inference import VideoInferenceEngine
import threading
from PIL import Image

class CCTVMonitor:
    def __init__(self, storage_path=None):
        # Environment configuration for Hugging Face (if needed)
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

        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize video inference engine
        local_model_path = r"C:/Users/Lenovo/Downloads/local_timesformer_gpt2"
        self.inference_engine = VideoInferenceEngine(model_path=local_model_path)
        self.clip_len = self.inference_engine.clip_len
        

        # Initialize camera capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"[INIT] Using FPS: {self.fps}")

        # Frame buffers and counters
        self.frames_per_video = 100  # Save video every 100 frames
        self.frame_interval = 30     # Process captioning every 30 frames (can be adjusted)
        self.video_buffer = []       # Buffer for full video clip (RGB frames)
        self.frame_buffer = []       # Buffer for caption analysis (RGB frames)
        self.frame_counter = 0       # Counter for frame buffering

        # Data tracking (log captions/analysis)
        self.captions_df = pd.DataFrame(columns=[
            'timestamp', 'camera_name', 'caption', 
            'is_suspicious', 'reason', 'confidence'
        ])
        
        # Load existing data if available
        self._load_existing_data()
        
        # Pending video analysis
        self.pending_videos = {}

    def _create_storage_directories(self):
        """Create required storage directories."""
        try:
            # Create main data directory
            self.data_dir = self.base_storage_path / "data"
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create footage directory
            self.footage_dir = self.data_dir / "footage"
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

    def _get_video_path(self, timestamp):
        """Generate video path based on timestamp."""
        # Ensure the footage directory exists
        if not self.footage_dir.exists():
            self.footage_dir.mkdir(parents=True, exist_ok=True)
            
        filename = f"video_{timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
        return self.footage_dir / filename

    def save_video(self, video_frames, start_time):
        """Save a video clip from buffered frames using a separate thread."""
        if not video_frames:
            return None

        video_path = self._get_video_path(start_time)
        print(f"[SAVING] Attempting to save video to: {video_path}")

        # Get frame dimensions from first frame (assume uniform size)
        height, width, _ = video_frames[0].shape

        # Use an event to signal when writing is complete
        write_complete = threading.Event()
        write_success = [False]  # Use a list to allow modification in the inner function

        def write_video():
            try:
                # Ensure the directory exists before writing
                video_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Use AVI format instead of MP4 since it's more reliable for writing
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                output_path = str(video_path).replace('.mp4', '.avi')
                writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
                
                if not writer.isOpened():
                    print(f"[ERROR] Failed to initialize writer for {output_path}")
                    write_complete.set()
                    return
                    
                # OpenCV frames are already in BGR format, so no need to convert
                for frame in video_frames:
                    writer.write(frame)  # Don't convert BGR to BGR
                
                # Make sure writer is released
                writer.release()
                
                # Verify file was written
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"[SUCCESS] Saved {len(video_frames)} frames to {output_path}")
                    # Update the path to the AVI file instead of MP4
                    write_success[0] = output_path  # Store the actual path used
                else:
                    print(f"[ERROR] File not created or empty: {output_path}")
            except Exception as e:
                print(f"[ERROR] Exception during video writing: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                write_complete.set()

        # Start writing in a separate thread
        write_thread = Thread(target=write_video)
        write_thread.daemon = True  # Make the thread daemon so it doesn't block program exit
        write_thread.start()
        
        # Return a tuple of the path and the completion event
        return video_path, write_complete, write_success

    def generate_caption(self, frames):
        """Generate caption for a list of frames using the vision model.
        
        This method uses the inference engine's caption_frames method for simpler processing.
        """
        try:
            print(f"[DEBUG] Generating caption for {len(frames)} frames")
            
            # Use the more reliable direct captioning method from the inference engine
            caption = self.inference_engine.caption_frames(frames)
            return caption
            
        except Exception as e:
            print(f"[ERROR] Failed to generate caption: {e}")
            import traceback
            traceback.print_exc()
            return "No caption available"

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

    def on_caption_ready(self, video_id, caption):
        """Callback for when a video caption is ready."""
        if video_id not in self.pending_videos:
            print(f"[WARNING] Received caption for unknown video ID: {video_id}")
            return
            
        video_info = self.pending_videos.pop(video_id)
        print(f"[CAPTION] Generated caption for {video_id}: {caption}")
        
        # Skip GPT analysis for empty or error captions
        if caption.startswith("Error") or not caption:
            analysis = {'is_suspicious': False, 'reason': 'No valid caption', 'confidence': 0.0}
        else:
            analysis = self.analyze_suspicious_activity(caption)
        
        # Add to dataframe
        timestamp = video_info['timestamp']
        new_row = pd.DataFrame([{
            'timestamp': timestamp,
            'camera_name': video_info['camera_name'],
            'caption': caption,
            'is_suspicious': analysis.get('is_suspicious', False),
            'reason': analysis.get('reason', 'Normal activity'),
            'confidence': analysis.get('confidence', 0.0)
        }])
        
        # Add the row and immediately save to disk
        self.captions_df = pd.concat([self.captions_df, new_row], ignore_index=True)
        self.save_data()
        
        is_suspicious = analysis.get('is_suspicious', False)
        print(f"[ANALYSIS] {'Suspicious' if is_suspicious else 'Normal'} activity detected: {caption}")
        
        # If suspicious, save frames as backup
        if is_suspicious and 'frames' in video_info:
            self.save_footage(video_info['frames'], timestamp)
        # 🔊 Server-side Buzz Sound
        try:
            from playsound import playsound
            playsound("static/buzz.mp3", block=False)
        except Exception as e:
            print("[BUZZ ERROR]", e)

    def process_video_clip(self, video_path_info):
        """Queue a video for asynchronous processing with the inference engine."""
        try:
            if not video_path_info:
                return None
                
            video_path, write_complete, write_success = video_path_info
            
            # Wait for the video file to be fully written
            if not write_complete.is_set():
                print(f"[PROCESSING] Waiting for video file to be written: {video_path}")
                write_complete.wait(timeout=30.0)  # Wait up to 30 seconds
            
            # Check if writing was successful
            if not write_success[0]:
                print(f"[ERROR] Video writing failed for {video_path}")
                return None
                
            # Get the actual path used (AVI instead of MP4)
            actual_path = write_success[0]
            if not isinstance(actual_path, str):
                print(f"[ERROR] Invalid path type: {type(actual_path)}")
                return None
                
            # Additional check to make sure the file exists and has size
            if not os.path.exists(actual_path):
                print(f"[ERROR] Video file not found: {actual_path}")
                return None
                
            file_size = os.path.getsize(actual_path)
            if file_size == 0:
                print(f"[ERROR] Video file is empty: {actual_path}")
                return None
                
            print(f"[PROCESSING] Processing video with size {file_size} bytes: {actual_path}")
                
            timestamp = datetime.now()
            video_id = f"video_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Store information about this video
            self.pending_videos[video_id] = {
                'timestamp': timestamp,
                'camera_name': 'Camera 1',
                'path': actual_path
            }
            
            # Queue the video for processing
            self.inference_engine.queue_video(
                actual_path, 
                video_id=video_id, 
                callback=self.on_caption_ready
            )
            
            return video_id
        except Exception as e:
            print(f"[ERROR] Video processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def process_frames_directly(self, frames):
        """Process frames directly without saving to video first (for live processing)."""
        try:
            timestamp = datetime.now()
            
            # Generate caption directly (blocking call)
            caption = self.generate_caption(frames)
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
                'confidence': analysis.get('confidence', 0.0)
            }])
            
            # Add the row and immediately save to disk
            self.captions_df = pd.concat([self.captions_df, new_row], ignore_index=True)
            self.save_data()
            
            # If suspicious, save the frames
            if analysis.get('is_suspicious', False):
                self.save_footage(frames, timestamp)
                
            return caption, analysis
        except Exception as e:
            print(f"[ERROR] Frame processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Error processing frames", {'is_suspicious': False}

    def save_footage(self, frames, timestamp):
        """Optionally save individual frames when suspicious activity is detected."""
        for i, frame in enumerate(frames):
            filename = self.footage_dir / f"frame_{timestamp.strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def run(self):
        """Main loop for capturing footage and processing it sequentially."""
        frame_count = 0  # Initialize counter for frames
        max_frames = 100  # Capture footage for 600 frames

        try:
            print(f"[INFO] Starting capture for {max_frames} frames...")
            while frame_count < max_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Append the raw frame for video saving (BGR format from OpenCV)
                self.video_buffer.append(frame.copy())  # Use copy to avoid reference issues
                frame_count += 1

                # Display the current frame (optional for demo purposes)
                cv2.imshow('CCTV Monitor - Capturing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Capture interrupted by user.")
                    break

            print(f"[INFO] Capture complete. Total frames captured: {frame_count}")

            # EXPLICITLY RELEASE THE CAMERA RIGHT AFTER CAPTURE
            self.cap.release()
            print("[INFO] Camera released after capture.")

            # Close all OpenCV windows before proceeding with video processing
            try:
                cv2.waitKey(100)  # Give a short delay to ensure windows update
                cv2.destroyAllWindows()
                cv2.waitKey(100)  # Give time for windows to actually close
                print("[INFO] Display windows closed.")
            except Exception as e:
                print(f"[WARNING] Error closing windows: {e}")

            # Save the captured footage as a video and get the video path
            current_timestamp = datetime.now()
            saved_video_info = self.save_video(self.video_buffer, current_timestamp)
            self.video_buffer = []  # Clear the buffer after saving

            if saved_video_info:
                video_path, write_complete, write_success = saved_video_info
                
                print("[INFO] Starting processing of the captured video...")
                # Process ONLY the video we just captured, not all videos in the folder
                
                # Wait for the write to complete
                if not write_complete.is_set():
                    print("[INFO] Waiting for video writing to complete...")
                    write_complete.wait(timeout=30.0)  # Wait up to 30 seconds
                
                if write_success[0]:  # Check if video was successfully written
                    actual_path = write_success[0]
                    print(f"[INFO] Processing newly captured video: {actual_path}")
                    self.process_video(actual_path)
                else:
                    print("[ERROR] Failed to save or process the captured video.")
            else:
                print("[ERROR] No video was saved.")

            print("[INFO] Processing complete. Press Ctrl+C to exit.")
            # Just wait for keyboard interrupt
            while True:
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("[INFO] Processing interrupted by user.")
        except Exception as e:
            print(f"[ERROR] Exception during capture/processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Double check that camera is released (in case of exception before explicit release)
            print("[INFO] Cleaning up resources...")
            try:
                if self.cap is not None and self.cap.isOpened():
                    self.cap.release()
                    print("[INFO] Camera released during cleanup.")
            except Exception as e:
                print(f"[ERROR] Failed to release camera: {e}")
                
            try:
                # Force close all OpenCV windows
                cv2.waitKey(100)  # Give a short delay
                cv2.destroyAllWindows()
                cv2.waitKey(100)  # Allow time for windows to close
                print("[INFO] All display windows destroyed.")
            except Exception as e:
                print(f"[ERROR] Failed to destroy windows: {e}")
                
            # Clear any remaining buffers
            self.video_buffer = []
            self.frame_buffer = []
            print("[INFO] All buffers cleared.")
            
            # One last attempt to ensure camera is released
            try:
                self.cap = None  # Remove reference to the camera object
                print("[INFO] Camera reference removed.")
            except:
                pass
                
            print("[INFO] Cleanup complete.")

    def process_video(self, video_path):
        """Process a saved video file to generate captions and analyze it."""
        try:
            # Verify the file exists and has size before attempting to open
            if not os.path.exists(video_path):
                print(f"[ERROR] Video file not found: {video_path}")
                return False
                
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                print(f"[ERROR] Video file is empty: {video_path}")
                return False
                
            print(f"[INFO] Processing video file: {video_path} ({file_size} bytes)")
            
            # Open the video using PyAV
            container = av.open(video_path)

            # Extract evenly spaced frames from video
            seg_len = container.streams.video[0].frames
            clip_len = self.inference_engine.clip_len  # Use inference_engine for clip_len
            indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))

            print(f"[INFO] Extracting {clip_len} frames from {seg_len} total frames")
            
            frames = []
            container.seek(0)
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    frames.append(frame.to_ndarray(format="rgb24"))

            print(f"[INFO] Successfully extracted {len(frames)} frames")
            
            # Generate caption using inference_engine
            caption = self.inference_engine.caption_frames(frames)
            print(f"[INFO] Generated caption: {caption}")

            # Analyze the caption for suspicious activity
            analysis = self.analyze_suspicious_activity(caption)
            print(f"[INFO] Analysis: {analysis}")
            
            # Add to dataframe
            timestamp = datetime.now()
            new_row = pd.DataFrame([{
                'timestamp': timestamp,
                'camera_name': 'Camera 1',
                'caption': caption,
                'is_suspicious': analysis.get('is_suspicious', False),
                'reason': analysis.get('reason', 'Normal activity'),
                'confidence': analysis.get('confidence', 0.0)
            }])
            
            # Add the row and immediately save to disk
            self.captions_df = pd.concat([self.captions_df, new_row], ignore_index=True)
            self.save_data()
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to process video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return False

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
                    f.write("-" * 50 + "\n")
                f.flush()
                os.fsync(f.fileno())  # Ensure it's written to disk
                
        except Exception as e:
            print(f"[ERROR] Failed to save data: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_report(self):
        """Generate and save a summary report via ChatGPT API based on logged metadata."""
        try:
            with open(self.data_dir / "metadata.txt", "r") as f:
                metadata = f.read()
            prompt = f"""Generate a summary report of the CCTV monitoring data:
            
{metadata}

Please provide:
1. Total number of suspicious activities
2. Timeline of events
3. Most common types of suspicious activities
4. Recommendations for security improvements
"""
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            report = response.choices[0].message.content
            with open(self.data_dir / "daily_report.txt", "w") as f:
                f.write(report)
            return report
        except Exception as e:
            print(f"[ERROR] Report generation failed: {str(e)}")
            return "Error generating report. Please check logs."

    def query_events(self, query):
        """Query logged events using ChatGPT API."""
        with open(self.data_dir / "metadata.txt", "r") as f:
            metadata = f.read()

        prompt = f"""
    You are an intelligent security assistant reviewing CCTV surveillance logs.

    The user has asked: "{query}"

    The logs below are in this format:
    [YYYY-MM-DD HH:MM:SS] CameraName
    Caption: description of what the camera saw
    [optional]: 🚨 Suspicious Behavior Detected!
    ------------------------------------------------------------

    Please analyze and respond by:
    1. Only referring to logs that directly match the query.
    2. Filtering by time, date, behavior, or camera when applicable.
    3. Ignoring unrelated logs.
    4. If no relevant events are found, respond clearly: "No relevant activity found for the specified criteria."

    Logs:
    {metadata}
    """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful security analyst."},
                {"role": "user", "content": prompt}
            ]
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