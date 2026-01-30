# 🩻 X-Ray Anomaly Detection System: End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat&logo=docker)](https://www.docker.com/)
[![Google Cloud Run](https://img.shields.io/badge/Google_Cloud-Run-4285F4?style=flat&logo=google-cloud)](https://cloud.google.com/run)
[![Model](https://img.shields.io/badge/Model-YOLO26-yellow)](https://github.com/ultralytics/ultralytics)

A comprehensive anomaly detection system for chest X-rays using Deep Learning. This project implements a complete **MLOps** workflow, ranging from model inference to serverless cloud deployment, featuring visual explainability (Explainable AI) for medical diagnostics.

## 🚀 **Live Demo**
Try the system deployed on Google Cloud Run (Serverless):
### 👉 **[INSERT_YOUR_APP_LINK_HERE]**

*(Note: Since the service uses a "Scale-to-Zero" configuration for cost efficiency, the first request might take 15-30 seconds to wake up the container. Subsequent requests will be instant).*

---

## 🧠 **System Architecture**

The project follows a containerized microservices architecture, designed for scalability and cost-efficiency.

### **Key Components:**
1.  **Core AI Model:**
    * **Architecture:** YOLO26 (Classification Head).
    * **Input:** Chest X-Ray images (Grayscale/RGB).
    * **Output:** Binary Classification (Normal vs. Anomaly) + Confidence Score.
    * **XAI (Explainability):** Implementation of **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate heatmaps highlighting pathological regions in the lungs.

2.  **Backend & API:**
    * Built with **FastAPI** for high-performance asynchronous processing.
    * Image processing using **OpenCV** and **PyTorch**.
    * Automatic documentation via Swagger UI.

3.  **MLOps & Observability:**
    * **MLflow (Local):** Experiment tracking and inference metric logging (latency, confidence distribution, drift detection).
    * **Docker:** Reproducible environment with *Multi-stage builds* to optimize image size and security.

4.  **Infrastructure (Cloud):**
    * **Google Cloud Run:** Fully managed Serverless deployment.
    * **Artifact Registry:** Secure Docker container management.
    * **CI/CD:** Atomic deployments via Google Cloud Build.

---

## 🛠️ **Local Installation & Usage**

To run the project locally and access the MLOps dashboard:

### Prerequisites
* Docker & Docker Compose
* Git

### Steps
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Esteebaan23/XRay_YOLO26.git](https://github.com/Esteebaan23/XRay_YOLO26.git)
    cd XRay_YOLO26
    ```

2.  **Build and Start Services:**
    ```bash
    docker-compose up --build
    ```

3.  **Access Interfaces:**
    * **Web App & API:** [http://localhost:8080](http://localhost:8080)
    * **MLflow Dashboard (Metrics):** [http://localhost:5000](http://localhost:5000)

---

## 📊 **MLOps: Metrics Monitoring**

The system not only predicts but also logs the health of the model in production (simulated locally). Key metrics monitored include:

* **Inference Latency:** Time (ms) from image upload to diagnosis delivery.
* **Confidence Distribution:** Detection of *Model Drift* (monitoring if the model becomes less certain over time).
* **Prediction Balance:** Ratio between "Normal" and "Anomaly" detections.

---

## 🖼️ **Screenshots**

### 1. Diagnostic Interface (With Grad-CAM)
![Web Interface](https://via.placeholder.com/800x400?text=Insert+Interface+Screenshot+Here)
*The system displays the diagnosis, confidence score, and a heatmap overlay on the affected area.*

### 2. MLflow Dashboard
![MLflow Dashboard](https://via.placeholder.com/800x400?text=Insert+MLflow+Screenshot+Here)
*Real-time logging of experiments and performance metrics.*

---

## 📂 **Project Structure**
XRay_YOLO26/ ├── app/ │ ├── main.py # FastAPI Entry Point and Frontend │ ├── service.py # Inference Logic and Grad-CAM │ ├── config.py # Global Configurations │ └── schemas.py # Pydantic Data Models ├── models/ │ └── best_model.pt # YOLO26 Trained Weights ├── mlruns/ # MLflow Logs (Gitignored) ├── Dockerfile # Multi-stage Docker Definition ├── docker-compose.yml # Service Orchestration (API + MLflow) └── requirements.txt # Python Dependencies
---

## 📂 **Project Structure**
## 👨‍💻 **Author**
**Esteban Lucero**
* [GitHub](https://github.com/Esteebaan23)
* [LinkedIn](INSERT_YOUR_LINKEDIN_URL)

---
*This project was developed for academic and demonstration purposes in MLOps and AI Engineering.*
