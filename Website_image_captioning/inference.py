import av
import numpy as np
import torch
from transformers import pipeline, VisionEncoderDecoderModel, GPT2Tokenizer, ViTImageProcessor
import os
from pathlib import Path
import queue
import threading
import time
from datetime import datetime
import cv2
from PIL import Image
import pandas as pd


class ImageInferenceEngine:
    def __init__(self, model_path=None):
        """Initialize the image inference engine with the ViT-GPT2 model.

        Args:
            model_path: Path to the local model. If None, will use the Hugging Face model.
        """
        # Configure environment
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '10000'  # 15 minutes timeout

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFERENCE] Using device: {self.device}")

        try:
            # Load model components
            print(f"[INFERENCE] Loading image captioning models...")
            # Use ENV variable or default to GitHub-cloned model inside Docker
            model_path = model_path or os.getenv("MODEL_PATH", "/app/NourFakih-Vit-GPT2-UCA-UCF-07")

            if os.path.exists(os.path.join(model_path, "config.json")):
                print(f"[INFERENCE] Using local model path: {model_path}")
                self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
                self.feature_extractor = ViTImageProcessor.from_pretrained(model_path)
            else:
                print(f"[INFERENCE] Downloading model from Hugging Face: NourFakih/Vit-GPT2-UCA-UCF-07")
                self.model = VisionEncoderDecoderModel.from_pretrained("NourFakih/Vit-GPT2-UCA-UCF-07")
                self.tokenizer = GPT2Tokenizer.from_pretrained("NourFakih/Vit-GPT2-UCA-UCF-07")
                self.feature_extractor = ViTImageProcessor.from_pretrained("NourFakih/Vit-GPT2-UCA-UCF-07")

            # Create the pipeline
            self.image_captioner = pipeline(
                task="image-to-text",
                model=self.model,
                tokenizer=self.tokenizer,
                feature_extractor=self.feature_extractor,
                device=self.device
            )

            print(f"[INFERENCE] Image captioning model loaded successfully")

        except Exception as e:
            print(f"[INFERENCE ERROR] Failed to load pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        # Threading setup
        self.queue = queue.Queue(maxsize=10)  # Image processing queue
        self.results = {}  # Store results keyed by image_id
        self.running = False
        self.worker_thread = None

    def start_worker(self):
        """Start the background worker thread for processing images."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        print("[INFERENCE] Worker thread started")


    def stop_worker(self):
        """Stop the background worker thread."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            print("[INFERENCE] Worker thread stopped")

    def _process_queue(self):
        """Background thread function to process images in the queue."""
        while self.running:
            try:
                # Get next item with timeout to allow checking running flag
                item = self.queue.get(timeout=1.0)
                if item is None:
                    self.queue.task_done()
                    continue

                image_path, image_id, callback = item
                print(f"[INFERENCE] Processing image from queue: {image_id} - {image_path}")

                # Process the image and get the caption
                caption = self.process_image(image_path)

                # Store result and call callback if provided
                self.results[image_id] = caption

                # Use a try-except block to ensure callback errors don't crash the worker
                if callback:
                    try:
                        callback(image_id, caption)
                    except Exception as e:
                        print(f"[INFERENCE ERROR] Callback error for {image_id}: {str(e)}")
                        import traceback
                        traceback.print_exc()

                # Log the result
                print(f"[INFERENCE] Completed image {image_id} - Caption: {caption[:50]}...")

                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[INFERENCE ERROR] Queue processing error: {str(e)}")
                import traceback
                traceback.print_exc()
                self.queue.task_done()

    def queue_image(self, image_path, image_id=None, callback=None):
        """Add an image to the processing queue.

        Args:
            image_path: Path to the image file
            image_id: Optional ID for the image (defaults to timestamp)
            callback: Optional function to call when processing is complete
                      with signature callback(image_id, caption)

        Returns:
            image_id: The ID assigned to this image
        """
        if not self.running:
            self.start_worker()

        if image_id is None:
            image_id = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.queue.put((image_path, image_id, callback))
        return image_id

    def get_result(self, image_id, wait=False, timeout=None):
        """Get the processing result for an image.

        Args:
            image_id: The ID of the image
            wait: If True, wait for the result if not available
            timeout: Maximum time to wait in seconds (if wait=True)

        Returns:
            Caption string or None if not available
        """
        if image_id in self.results:
            return self.results[image_id]

        if not wait:
            return None

        # Wait for the result with a timeout
        start_time = time.time()
        while timeout is None or (time.time() - start_time) < timeout:
            if image_id in self.results:
                return self.results[image_id]
            time.sleep(0.1)

        return None

    def process_image(self, image_path):
        """Process an image file and generate a caption.

        Args:
            image_path: Path to the image file

        Returns:
            Caption string
        """
        try:
            # Validate file exists and has size
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            file_size = os.path.getsize(image_path)
            if file_size == 0:
                raise ValueError(f"Image file is empty (0 bytes): {image_path}")

            print(f"[INFERENCE] Processing image: {image_path} ({file_size} bytes)")

            # Generate caption using the pipeline
            result = self.image_captioner(
                image,
                generate_kwargs={
                    "max_length": 60,
                    "min_length": 5,
                    "num_beams": 4,
                    "do_sample": False
                }
            )

            if result and isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    return result[0]['generated_text']
                else:
                    return str(result[0])
            else:
                return "No caption generated"

        except Exception as e:
            print(f"[INFERENCE ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            return "Error generating caption"

    def caption_image(self, image):
        """Generate caption for an image directly without saving to file.

        Args:
            image: PIL Image or numpy array

        Returns:
            Caption string
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Generate caption using the pipeline
            result = self.image_captioner(
                image,
                generate_kwargs={
                    "max_length": 60,
                    "min_length": 5,
                    "num_beams": 4,
                    "do_sample": False
                }
            )

            if result and isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    return result[0]['generated_text']
                else:
                    return str(result[0])
            else:
                return "No caption generated"

        except Exception as e:
            print(f"[INFERENCE ERROR] Direct captioning failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Error generating caption"
#
#
# class VideoInferenceEngine:
#     def __init__(self, model_path=None):
#         """Initialize the video inference engine with the SpaceTimeGPT model.
#
#         Args:
#             model_path: Path to the local model. If None, will use the Hugging Face model.
#         """
#         # Configure environment
#         os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
#         os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '10000'  # 15 minutes timeout
#
#         # Set device
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"[INFERENCE] Using device: {self.device}")
#
#         try:
#             # Initialize the pipeline - much simpler than loading model components separately
#             print(f"[INFERENCE] Loading video captioning pipeline...")
#
#             if model_path and os.path.exists(model_path):
#                 print(f"[INFERENCE] Using local model path: {model_path}")
#                 self.pipeline = pipeline("video-to-text", model=model_path, device=self.device)
#             else:
#                 print(f"[INFERENCE] Loading model from Hugging Face: Neleac/SpaceTimeGPT")
#                 self.pipeline = pipeline("video-to-text", model="Neleac/SpaceTimeGPT", device=self.device)
#
#             # Set the clip length based on model requirements
#             self.clip_len = 8  # Default for SpaceTimeGPT
#             print(f"[INFERENCE] Using clip length: {self.clip_len}")
#         except Exception as e:
#             print(f"[INFERENCE ERROR] Failed to load pipeline: {str(e)}")
#             print("[INFERENCE] Falling back to custom implementation...")
#             # Fallback to manual implementation if pipeline fails
#             self._init_legacy_model(model_path)
#
#         # Threading setup
#         self.queue = queue.Queue(maxsize=10)  # Video processing queue
#         self.results = {}  # Store results keyed by video_id
#         self.running = False
#         self.worker_thread = None
#
#     def _init_legacy_model(self, model_path):
#         """Legacy initialization method if pipeline fails."""
#         from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
#
#         # Load models
#         print("[INFERENCE] Loading model components individually...")
#         self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
#         self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
#
#         # Load the model - either from local path or HuggingFace
#         if model_path and os.path.exists(model_path):
#             print(f"[INFERENCE] Loading model from local path: {model_path}")
#             self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
#         else:
#             print("[INFERENCE] Loading model from HuggingFace (this may take time)")
#             self.model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(
#                 self.device)
#
#         self.clip_len = self.model.config.encoder.num_frames
#         print(f"[INFERENCE] Using legacy model with clip length: {self.clip_len}")
#         self.using_pipeline = False
#
#     def _extract_frames(self, video_path):
#         """Extract frames from a video file using multiple methods.
#
#         Args:
#             video_path: Path to the video file
#
#         Returns:
#             List of frames or None if extraction failed
#         """
#         # Try PyAV first
#         try:
#             container = av.open(str(video_path))
#             if not container.streams.video:
#                 raise ValueError("No video streams found")
#
#             seg_len = container.streams.video[0].frames
#             if seg_len <= 0:
#                 raise ValueError(f"Invalid frame count: {seg_len}")
#
#             # Extract evenly spaced frames
#             indices = set(np.linspace(0, seg_len - 1, num=self.clip_len, endpoint=False).astype(np.int64))
#             frames = []
#
#             container.seek(0)
#             for i, frame in enumerate(container.decode(video=0)):
#                 if i in indices:
#                     frames.append(frame.to_ndarray(format="rgb24"))
#
#             if len(frames) >= self.clip_len:
#                 return frames
#             else:
#                 print(f"[INFERENCE WARNING] Only extracted {len(frames)} frames with PyAV, needed {self.clip_len}")
#         except Exception as e:
#             print(f"[INFERENCE WARNING] PyAV extraction failed: {str(e)}")
#
#         # Fall back to OpenCV
#         try:
#             print(f"[INFERENCE] Trying OpenCV extraction for {video_path}")
#             cap = cv2.VideoCapture(str(video_path))
#             if not cap.isOpened():
#                 raise ValueError("Could not open video with OpenCV")
#
#             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             if frame_count <= 0:
#                 # Try counting frames manually
#                 frame_count = 0
#                 while True:
#                     ret, _ = cap.read()
#                     if not ret:
#                         break
#                     frame_count += 1
#
#                 # Reset to beginning
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
#             if frame_count <= 0:
#                 raise ValueError("No frames in video")
#
#             # Extract evenly spaced frames
#             indices = set(np.linspace(0, frame_count - 1, num=self.clip_len, endpoint=False).astype(np.int64))
#             frames = []
#
#             for i in range(frame_count):
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#
#                 if i in indices:
#                     # Convert BGR to RGB
#                     frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#             cap.release()
#
#             # Ensure we have enough frames
#             if len(frames) < self.clip_len and frames:
#                 # Duplicate last frame if needed
#                 while len(frames) < self.clip_len:
#                     frames.append(frames[-1])
#
#             if frames:
#                 return frames
#         except Exception as e:
#             print(f"[INFERENCE WARNING] OpenCV extraction failed: {str(e)}")
#
#         return None
#
#     def _generate_caption(self, frames):
#         """Generate a caption from frames.
#
#         Args:
#             frames: List of RGB frames
#
#         Returns:
#             Caption string
#         """
#         try:
#             # Check if we're using the pipeline
#             if hasattr(self, 'pipeline'):
#                 # Resize frames to expected dimensions
#                 resized_frames = []
#                 for frame in frames:
#                     # Convert to PIL Image for consistent processing
#                     pil_img = Image.fromarray(frame)
#                     resized = pil_img.resize((224, 224))
#                     resized_frames.append(resized)
#
#                 # Use the pipeline to generate caption
#                 result = self.pipeline(resized_frames)
#
#                 # Extract caption from pipeline result
#                 if isinstance(result, list) and result:
#                     if isinstance(result[0], dict) and 'generated_text' in result[0]:
#                         return result[0]['generated_text']
#                     elif hasattr(result[0], 'generated_text'):
#                         return result[0].generated_text
#                     else:
#                         return str(result[0])
#                 else:
#                     return str(result)
#             else:
#                 # Legacy approach using manual model components
#                 with torch.no_grad():
#                     # Resize frames to expected dimensions
#                     resized_frames = [Image.fromarray(frame).resize((224, 224)) for frame in frames]
#
#                     gen_kwargs = {
#                         "min_length": 10,
#                         "max_length": 20,
#                         "num_beams": 8,
#                     }
#
#                     # Process frames with image processor
#                     pixel_values = self.image_processor(resized_frames, return_tensors="pt").pixel_values.to(
#                         self.device)
#
#                     # Generate tokens
#                     tokens = self.model.generate(pixel_values, **gen_kwargs)
#
#                     # Decode tokens to text
#                     caption = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
#                     return caption
#
#         except Exception as e:
#             print(f"[INFERENCE ERROR] Caption generation failed: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return "Error generating caption"
#
#     def caption_frames(self, frames):
#         """Generate caption for frames directly (without saving to video).
#
#         Args:
#             frames: List of RGB frames
#
#         Returns:
#             Caption string
#         """
#         try:
#             # Ensure we have exactly clip_len frames
#             if len(frames) < self.clip_len:
#                 print(f"[WARNING] Got {len(frames)} frames, needed {self.clip_len}")
#                 # Duplicate last frame if needed
#                 while len(frames) < self.clip_len and frames:
#                     frames.append(frames[-1])
#             elif len(frames) > self.clip_len:
#                 # Extract evenly spaced frames
#                 indices = set(np.linspace(0, len(frames) - 1, num=self.clip_len, endpoint=False).astype(np.int64))
#                 frames = [frames[i] for i in indices]
#
#             return self._generate_caption(frames)
#
#         except Exception as e:
#             print(f"[INFERENCE ERROR] Direct captioning failed: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return "Error generating caption"
#

# Example usage
if __name__ == "__main__":
    # Initialize the engine - provide a local model path if available
    local_model_path = r"C:/Users/Lenovo/Downloads/local_timesformer_gpt2"
    engine = VideoInferenceEngine(model_path=local_model_path)


    # Example callback function
    def on_caption_ready(video_id, caption):
        print(f"Caption for {video_id}: {caption}")


    # Process a video file
    video_path = r"C:/Users/Lenovo/Downloads/Abuse009_x264_1.mp4"

    # Queue the video for processing
    video_id = engine.queue_video(video_path, callback=on_caption_ready)
    print(f"Queued video with ID: {video_id}")

    # Wait for some time to let the worker process the video
    time.sleep(10)

    # Get the result directly
    result = engine.get_result(video_id)
    if result:
        print(f"Result: {result}")
    else:
        print("Result not available yet")

    # Clean shutdown
    engine.stop_worker()
