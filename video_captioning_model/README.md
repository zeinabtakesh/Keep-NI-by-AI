# Video Captioning Pipeline (SpaceTimeGPT)

## 1. Dataset Overview
- **Dataset**: UCF-UCA (UCF Crime Abnormality)  
- **Kaggle Link**: https://www.kaggle.com/datasets/username/ucf-uca  
- **Description**:  
  - Surveillance videos containing both **abnormal** (e.g. fights, thefts) and **normal** activities.  
  - Human-written captions aligned to temporal clips indicating the behavior in each segment.

![Video dataset overview](docs/ucf_uca_video_overview.png)

## 2. Data Preparation
1. **Sampling**  
   - **Abnormal categories**: up to **120 videos** per category (max available, downsampled for resource constraints)  
   - **Normal class**: **400 videos**  
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
| Validation  | 500     |
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

## 5. Training Attempts & Challenges
- **Initial Training** (`proccess+train_ucf.ipynb`)  
  - Used full training set → **GPU OOM errors**  
- **Gradient Checkpointing** (`Process+resume_training.ipynb`)  
  - Reduced memory footprint, but **slower** epochs  
- **Portion-wise Fine-tuning**  
  - Trained on subsets → coherent captions on common actions, **missed anomalies**  
- **Key Challenges**  
  - **Resource Constraints**: Insufficient GPU RAM for >100k clips  
  - **Data Volume**: Need large-scale video+caption pairs  
  - **Overfitting**: Model learned “normal” patterns better than rare “anomalous” events

## 6. Decision & Future Work
- **Current Focus**:  
  - Return to the **ViT-GPT2 image captioning** pipeline for our demo (lighter, stable).  
- **Planned Next Steps**:  
  1. Acquire resources to train SpaceTimeGPT on ≥100k clips.  
  2. Explore lighter backbones (e.g. Video Swin Transformer).  
  3. Incorporate anomaly-specific classification head.  
  4. Build a streaming pipeline for real-time inference on CCTV feeds.
