# Bankcrisis OpenEnv Environment

A custom OpenEnv-compatible environment that simulates a bank crisis scenario. This project exposes the environment through a FastAPI server and is fully containerized using Docker for deployment.

---

## 🚀 Features

* OpenEnv-compatible environment
* FastAPI-based HTTP server
* Dockerized for easy deployment
* Hugging Face Spaces integration
* Supports multi-mode validation via OpenEnv

---

## 📦 Project Structure

```
bankcrisis/
├── bankcrisis/
│   ├── models.py
│   ├── client.py
│   └── server/
│       ├── app.py
│       └── bankcrisis_environment.py
│
├── server/                # Wrapper for OpenEnv compatibility
│   └── app.py
│
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## ⚙️ Installation (Local)

```bash
git clone https://github.com/Syam-18/bankcrisis.git
cd bankcrisis

pip install -e .
```

---

## ▶️ Run Locally

```bash
uvicorn bankcrisis.server.app:app --host 0.0.0.0 --port 8000
```

Then open:

```
http://localhost:8000/docs
```

---

## 🐳 Run with Docker

```bash
docker build -t bankcrisis-env .
docker run -p 8000:8000 bankcrisis-env
```

---

## 🌐 Deployment

Deployed on Hugging Face Spaces:

👉 https://huggingface.co/spaces/Sk8erBoi369/bankcrisis

---

## ✅ Validation

This project passes OpenEnv validation:

* Docker build ✅
* Live endpoint check ✅
* Multi-mode deployment support ✅

---

## 📌 Notes

* Includes a compatibility wrapper (`server/app.py`) required by OpenEnv
* Uses absolute imports to ensure compatibility across environments

---

## 👤 Author

Syam

---

## 📄 License

This project follows the BSD-style license as per OpenEnv guidelines.
