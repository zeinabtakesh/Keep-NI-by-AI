# Video Captioning Pipeline (SpaceTimeGPT)

Visualized sample ground-truth vs. predicted captions.
---
## 1. Dataset Overview
- **Dataset**: UCF-UCA (UCF Crime Abnormality)  
- **Kaggle Link**: https://www.kaggle.com/datasets/vigneshwar472/ucaucf-crime-annotation-dataset
- **Description**:  
  - Surveillance videos containing both **abnormal** (e.g. fights, thefts) and **normal** activities.  
  - Human-written captions aligned to temporal clips indicating the behavior in each segment.

![Video dataset overview](docs/ucf_uca_video_overview.png)

## 2. Data Preparation: https://www.kaggle.com/datasets/nourfakih/splitted-ucf-120videospercategory
1. **Sampling**  
   - **Abnormal categories**: up to **120 videos** per category (downsampled for resource constraints)  
   - **Normal class**: **120 videos**  
2. **Segmentation**  
   - Split each video into clips based on provided timestamps  
   - Ensure each clip corresponds to a single caption  
3. **Preprocessing**  
   - Resize frames to model’s input resolution  
   - Normalize pixel values  
   - Cache video tensors for faster loading
 

![Clip segmentation example](docs/ucf_uca_clip_segmentation.png)

## 3. Dataset Split
| Split       | # Clips |
| ----------- | ------- |
| Training    | 8,500   |
| Validation  | 1000    |
| Testing     | 500     |

## 4. Model Choice & Rationale
- **CoCap**  
  - Pros: High accuracy on video tasks  
  - Cons: No public pretrained checkpoint + very high GPU requirements  
- **TimeSformer** (as standalone)  
  - Pros: Strong video encoder  
  - Cons: Lacks integrated language decoder  
- **SpaceTimeGPT** (TimeSformer + GPT-2)  
  - Pros: End-to-end video→caption architecture  
  - Cons: Demands substantial memory/compute for large datasets
  - Our Finetuned model HuggingFace Repository: NourFakih/TimeSformer-GPT2-UCF-7000

## 5. Training Attempts & Challenges
- **Initial Training** (`Train-TimeSformerGPT2-UCF.ipynb`)  
  - Used full training set → **GPU OOM errors**  
- **Portion-wise Fine-tuning**  
  - Trained on subsets → coherent captions on common actions.
- **Key Challenges**  
  - **Resource Constraints**: Insufficient GPU RAM for >100k clips  
  - **Data Volume**: Need large-scale video+caption pairs  
  - **Overfitting**: Model learned “normal” patterns better than rare “anomalous” events
 

## 📂 File Structure & Descriptions

- **`video_captioning/Split_UCF_Videos.ipynb`**  
  • **Purpose:** Read the raw UCF-UCA JSON annotations and video files, then split each video into clips based on the provided timestamps.  
  • **Actions:**  
    1. Loaded `UCF-UCA` metadata and video paths  
    2. Generated individual clip files (e.g. MP4s) for each `[start, end]` segment  
    3. Saved clip-to-caption mappings for downstream processing  

- **`video_captioning/Get-50-videospercategory.ipynb`**  
  • **Purpose:** Balance the dataset by sampling a uniform number of videos per abnormality category.  
  • **Actions:**  
    1. Counted available videos in each class  
    2. Randomly selected up to 150 per abnormal category, and 150 normals  
    3. Logged the final train/val/test split lists  

- **`video_captioning/Train-TimeSformerGPT2-UCF.ipynb`**  
  • **Purpose:** Initial end-to-end training attempt of SpaceTimeGPT on the full set of clips.  
  • **Actions:**  
    1. Preprocessed clips into tensors (resizing, normalization)  
    2. Configured SpaceTimeGPT model and training loop  
    3. Launched training   
    4. Logged memory usage and early diagnostics

 - **`video_captioning/Test-video.ipynb`**  
  • **Purpose:** Test the trained SpaceTimeGPT checkpoint on held-out clips and compute evaluation metrics..  
  • **Actions:**  
    1. Loaded the best model checkpoint.  
    2. Ran inference on the test split of 500 clips. 

## 6. Decision & Future Work
- **Current Focus**:  
  - Return to the **ViT-GPT2 image captioning** pipeline for our demo (lighter, stable).  
- **Planned Next Steps**:  
  1. Acquire resources to train SpaceTimeGPT on ≥100k clips.  
  2. Explore lighter backbones (e.g. Video Swin Transformer).  
  3. Incorporate anomaly-specific classification head.  
  4. Build a streaming pipeline for real-time inference on CCTV feeds.
