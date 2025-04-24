# Image Captioning Pipeline (ViT-GPT2)

## 1. Dataset Overview
- **Dataset**: UCF-UCA (UCF Crime Abnormality)  
- **Kaggle Link**:https://www.kaggle.com/datasets/vigneshwar472/ucaucf-crime-annotation-dataset
- **Description**:  
  - A collection of surveillance videos labeled with human‐written captions for segments containing abnormal or suspicious activity.  
  - Each video comes with timestamps indicating when each caption applies.

![Dataset overview screenshot](docs/ucf_uca_dataset_overview.png)

## 2. Data Subsampling
- **Abnormal categories**: Sampled **50–150 videos** per category (maximum available).  
- **Normal class**: Sampled **200 videos** from the pool of 800.  
- **Reason**:  
  - Limited storage and compute (Kaggle environment).  
  - Maintain balanced representation across categories.
  - **Kaggle Link**:[https://www.kaggle.com/datasets/nourfakih/ucf-crime-extracted-frames ](https://www.kaggle.com/datasets/nourfakih/ucf-crime-extracted-frames) 
## 3. Frame Extraction
- **Tool**: PySceneDetect for scene-change detection.  
- **Approach**:  
  1. Detect key scenes in each sampled video segment.  
  2. Extract **1–2 representative frames** per segment.  
  3. Associate each frame with its human‐written caption.  

![Frame extraction example](docs/ucf_frame_extraction_example.png)

## 4. Model Choice & Rationale
- **Model**: ViT-GPT2 (Vision Transformer + GPT-2)  
- **Why ViT-GPT2?**  
  - Moderate GPU/memory requirements.  
  - Strong off-the-shelf image captioning performance.  
  - Easily fine-tuned on custom frame–caption pairs.
  - Finetuned Model HiggingFace Repository: NourFakih/Vit-GPT2-UCA-UCF-06

## 5. Dataset Split
| Split       | # Images |
| ----------- | -------- |
| Training    | 25,000   |
| Validation  | 1,000    |
| Testing     | 1,000    |

## 6. Training Configuration
- **Optimizer**: AdamW  
- **Learning Rate**: 5e-5  
- **Batch Size**: 4  
- **Epochs**: 3 

## 7. Evaluation Metrics
| Metric    | Score |
| --------- | ----- |
| ROUGE-L   |34.6 |


## 8. Observations & Next Steps
- **Strengths**  
  - Good static‐scene caption quality.  
  - Lightweight and fast end-to-end training.

- **Limitations**  
  - Occasional hallucinations on complex scenes.  
  - Lacks temporal context for multi-step actions.

- **Next Step**  
  - Transition to a video captioning approach (temporal modeling) for richer context understanding.
