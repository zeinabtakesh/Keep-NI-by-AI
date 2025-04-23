# Keep NI (An Eye) by AI
## CCTV-AI Monitoring

**Objective:** Build an AI agent to monitor CCTV feeds, detect malicious or anomalous behavior, and issue natural-language alerts.

---

## 🚀 Motivation & Vision
- **Problem:** Traditional CCTV requires human operators who can miss critical events due to fatigue or volume.  
- **Approach:** Combine image- and video-captioning models with a lightweight demo app that flags suspicious activity and leverages ChatGPT for human-readable alerts.  
- **Benefits:**  
  - Faster, automated incident detection  
  - Reduced operator workload  
  - Adaptable to new environments via fine-tuning  

---

## 📂 Repository Structure
CCTV-AI-Monitoring/
├── README.md
├── image_captioning/
│   ├── Training_On_UCF-UCA.ipynb
│   ├── Video_to_image_UCF.ipynb
│   ├── Test-ucf-image-models.ipynb
│   └── README.md
├── video_captioning/
│   ├── Split_UCF_Videos.ipynb
│   ├── Get-50-videospercategory.ipynb
│   ├── proccess+train_ucf.ipynb
│   ├── Process+resume_training.ipynb
│   └── README.md
└── cctv_app/
    ├── env/
    ├── cctv_monitor.py
    ├── inference.py
    ├── main.py
    ├── requirements.txt
    └── README.md
