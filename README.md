# Keep NI (An Eye) by AI
## CCTV-AI Monitoring

**Objective:** Build an AI agent to monitor CCTV feeds, detect malicious or anomalous behavior, and issue natural-language alerts.

---
# CCTV AI Monitoring – An Intro to Machine Learning Project

## 🔍 Project Motivation

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

## 📦 Output Models & Datasets

| Type               | Link                                                                 |
|--------------------|----------------------------------------------------------------------|
| Kaggle Dataset     | [UCF-UCA Sample + Processed](https://www.kaggle.com/) *(add link)*   |
| HuggingFace Models | [ViT-GPT2 Captioning](https://huggingface.co/) *(add link)*          |

## 🚧 Future Work

- Build a scalable infrastructure to run the video captioning model on full datasets.
- Integrate a GUI-based monitoring dashboard.
- Implement suspicious behavior classification and anomaly detection.
- Enable continuous learning and feedback loops for model improvement.

---

