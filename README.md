# Keep NI (An Eye) by AI  ![logo](https://github.com/user-attachments/assets/c631e93d-145b-4da0-ac84-7a54e9ac2f1e) 
## CCTV-AI Monitoring

**Objective:** Build an AI agent to monitor CCTV feeds, detect malicious or anomalous behavior, and issue natural-language alerts.

---
# CCTV AI Monitoring – An Intro to Machine Learning Project

## 🔍 Project Motivation

![Screenshot 2025-04-24 123849](https://github.com/user-attachments/assets/757d1ead-45e9-47c0-b6ee-723b9df3fdbb)


With the increasing demand for automated surveillance in both public and private environments, our project aims to build an intelligent **CCTV AI Monitoring System**. The system is designed to detect **malicious and suspicious behavior** in surveillance footage, providing real-time alerts and captioned descriptions of the scene. By combining computer vision and natural language processing, we propose a dual-approach system leveraging both **image** and **video captioning models**.

Our motivation came from real-world security limitations—human fatigue, late response time, and the volume of camera feeds to monitor simultaneously. Our AI agent acts as a **first line of analysis**, highlighting potential threats automatically.

## 🚀 Motivation & Vision
- **Problem:** Traditional CCTV requires human operators who can miss critical events due to fatigue or volume.  
- **Approach:** Combine image- and video-captioning models with a lightweight demo app that flags suspicious activity and leverages ChatGPT for human-readable alerts.  
- **Benefits:**  
  - Faster, automated incident detection  
  - Reduced operator workload  
  - Adaptable to new environments via fine-tuning  

## 📁 Repository Structure

This repository is divided into three main parts:

### 1. `image-captioning-model/`
- Focuses on using the **ViT-GPT2** transformer model for image captioning.
- Includes data preprocessing, frame extraction, model training, and testing notebooks.
- Readme explains why this approach was initiated and later paused.

### 2. `video-captioning-model/`
- Implements video captioning using models like **SpaceTimeGPT** (TimeSformer + GPT-2).
- Contains video splitting by timestamps, data processing, and training notebooks.
- Readme details dataset structure, training challenges, and reasons for continuing this approach.

### 3. `cctv-app/`
- A functional **demo** that uses the trained captioning model for real-time video inference.
- Built with Python and includes interaction with the **ChatGPT API** for intelligent feedback.
- Readme documents the setup, metadata logging, limitations, and planned interface features.

## 💡 Our Approach

We explored two complementary deep learning pipelines:

1. **Image Captioning** using ViT-GPT2:
   - Lightweight and easy to fine-tune.
   - Efficient but prone to hallucination without full visual context.

2. **Video Captioning** using TimeSformer and SpaceTimeGPT:
   - Better context modeling through temporal understanding.
   - More resource-intensive but offers higher accuracy in complex behaviors.

After evaluating both, we adopted the **image captioning model** for the demo due to hardware limitations, while preserving video captioning for future work.

## 📦 Output Models & Datasets (needs to be editted)

| Type               | Link                                                                 |
|--------------------|----------------------------------------------------------------------|
| Kaggle Dataset     | [UCF-UCA Sample + Processed]([https://www.kaggle.com/](https://www.kaggle.com/datasets/nourfakih/splitted-ucf-120videospercategory))  |
| HuggingFace Models | [ViT-GPT2 Captioning]([https://huggingface.co/](https://huggingface.co/NourFakih/Vit-GPT2-UCA-UCF-06))    |


## 🔗 Why Kaggle & Hugging Face

- **Kaggle for Data & Compute**  
  - The UCF-UCA dataset is large (hundreds of GB), and we needed a hosted workspace to store, preprocess, and train on it without local infrastructure limits.  
  - Kaggle provides **up to 12 hours of continuous GPU/TPU** per session, which was essential for our multi-hour training runs and video processing pipelines.  
  - Alternative (AUB HPC) could not reliably download or mount the full dataset, making local or VM-based workflows infeasible.

- **Hugging Face for Models & Artifacts**  
  - We selected and fine-tuned **lightweight transformer checkpoints** (ViT-GPT2 and SpaceTimeGPT) directly from Hugging Face’s model hub.  
  - After training, we host our best-performing model artifacts on Hugging Face for easy sharing, versioning, and integration into our `cctv_app` demo.  


## 🚧 Future Work

- Build a scalable infrastructure to run the video captioning model on full datasets.
- Integrate a GUI-based monitoring dashboard.
- Implement suspicious behavior classification and anomaly detection.
- Enable continuous learning and feedback loops for model improvement.

*Important Note*:
While the 3 of us worked on the models, we have only used 1 account on Kaggle(Nour's account), because it was the only account having free GPU. Fatima and Zainab's accounts needed phone verification to access GPU, but Kaggle is not offering it currently in Lebanon.
---

