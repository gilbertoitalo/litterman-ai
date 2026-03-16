# 🧠 Litterman.ai — Black-Litterman Voice Co-Pilot

**Gemini Live Agent Challenge 2026 — Live Agents Category**

> A real-time voice-driven portfolio optimization agent for asset managers.

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live%20on%20Cloud%20Run-4a9eff?style=flat-square)](https://litterman-dashboard-1084835415345.us-central1.run.app)
[![Challenge](https://img.shields.io/badge/Gemini%20Live%20Agent%20Challenge-2026-c9a84c?style=flat-square)](https://devpost.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🎯 What It Does

Litterman.ai is a voice agent that listens to financial market news, runs the **Black-Litterman portfolio optimization model**, and responds verbally with a rebalancing recommendation — all in real time.

A portfolio manager speaks naturally:
> *"The Fed raised rates 50 basis points today, citing persistent inflation."*

The agent:
1. 🎙️ Detects this as a market event
2. 🔍 Classifies and extracts quantitative views using Gemini 2.5 Flash + Google Search Grounding
3. ⚙️ Runs the Black-Litterman optimizer (numpy/scipy)
4. 🔊 Responds aloud with the new allocations and Sharpe ratio
5. 📊 Updates the live dashboard on Cloud Run in real time via Firestore

The manager can **interrupt mid-response** with follow-up questions — the agent stops, listens, and answers.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL (manager's machine)                 │
│                                                             │
│  Microphone → PyAudio → Gemini Live API (Native Audio)     │
│                              ↓                              │
│              Input transcription (real-time)               │
│                              ↓                              │
│         Gemini 2.5 Flash: is this market news?             │
│                              ↓ YES                          │
│         Gemini 2.5 Flash + Google Search Grounding         │
│              extract Black-Litterman views (JSON)          │
│                              ↓                              │
│         Black-Litterman Engine (numpy/scipy/SLSQP)         │
│                              ↓                              │
│         Gemini Live: speak result aloud                    │
│                              ↓                              │
│              Firestore ←── push state                      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Google Cloud      │
                    │                     │
                    │   Firestore         │
                    │   (shared state)    │
                    │        ↓            │
                    │   Cloud Run         │
                    │   (dashboard)       │
                    │        ↓            │
                    │   Public URL        │
                    └─────────────────────┘
```

### ☁️ Google Cloud Services Used

| Service | Role |
|---|---|
| Gemini 2.5 Flash Native Audio (Live API) | Voice input/output, barge-in |
| Gemini 2.5 Flash + Google Search Grounding | News classification + view extraction |
| Cloud Firestore | Real-time shared state between agent and dashboard |
| Cloud Run | Hosts the Flask dashboard publicly |
| Cloud Build | Automated container image build |
| Google Container Registry | Stores the Docker image |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Voice I/O | Gemini 2.5 Flash Native Audio · PyAudio |
| AI SDK | Google GenAI SDK (google-genai) |
| Portfolio model | Black-Litterman (numpy · scipy · SLSQP) |
| State | Google Cloud Firestore (default database) |
| Dashboard | Flask + Plotly.js (dark/light theme) |
| Backend hosting | Google Cloud Run · us-central1 |
| Container build | Google Cloud Build |

---

## 📁 Project Structure

```
litterman-ai/
├── core/
│   ├── voice_agent.py      # Gemini Live session, audio I/O, orchestration
│   ├── gemini_agent.py     # View extraction, BL pipeline, voice formatting
│   ├── bl_engine.py        # Black-Litterman model (pure numpy/scipy)
│   └── shared_state.py     # Firestore read/write bridge
├── dashboard.html          # Flask dashboard (dark/light, Plotly.js)
├── server.py               # Flask: GET /, POST /analyse, GET /health
├── main.py                 # Entry point for local voice agent
├── Dockerfile              # Cloud Run container definition
├── deploy.sh               # Automated Cloud Run deploy script
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🌐 Live Dashboard

The dashboard is publicly accessible at:

**[https://litterman-dashboard-1084835415345.us-central1.run.app](https://litterman-dashboard-1084835415345.us-central1.run.app)**

It updates in real time as the voice agent processes market news locally and writes results to Firestore.

---

## 🧪 Reproducible Testing

Follow the steps below to run the voice agent locally and reproduce the demo.

## 🚀 Running Locally (Voice Agent)

### Prerequisites

- Python 3.11+
- A Google Cloud project with Firestore enabled (database must be named `(default)`)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com)
- `gcloud` CLI installed and authenticated (`gcloud auth application-default login`)
- A working microphone

### 1. Clone the repository

```bash
git clone https://github.com/gilbertoitalo/litterman-ai
cd litterman-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyAudio requires system audio libraries.
> - Windows: `pip install pipwin && pipwin install pyaudio`
> - Ubuntu: `sudo apt-get install portaudio19-dev && pip install pyaudio`
> - Mac: `brew install portaudio && pip install pyaudio`

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
```

> Authentication uses Application Default Credentials (ADC) via `gcloud auth application-default login` — no service account JSON required.

### 5. Run the voice agent

```bash
python main.py
```

The agent will start listening. Speak any financial market news and it will respond with a portfolio rebalancing recommendation.

**Example inputs:**
- *"The Federal Reserve raised interest rates by 50 basis points today."*
- *"US non-farm payrolls came in at 350,000, well above the 200,000 consensus."*
- *"China announced new stimulus measures worth 1 trillion yuan."*

**To interrupt mid-response:** speak naturally — the agent stops and answers your follow-up.

### 6. Open the dashboard

Visit the [live Cloud Run URL](https://litterman-dashboard-1084835415345.us-central1.run.app) to see the dashboard update in real time, or run locally:

```bash
python server.py
```

Then open `http://localhost:8080`.

---

## ☁️ Cloud Deployment (Dashboard)

The dashboard is deployed to Google Cloud Run using the included automated deploy script.

**Proof of deployment:** [deploy.sh](deploy.sh) | [Live dashboard](https://litterman-dashboard-1084835415345.us-central1.run.app)

### Prerequisites

- `gcloud` CLI installed and authenticated
- Docker not required — uses Cloud Build

### Deploy

```bash
# Mac/Linux
chmod +x deploy.sh
./deploy.sh
```

```powershell
# Windows (PowerShell)
$PROJECT_ID = "your-project-id"
$REGION = "us-central1"
$SERVICE_NAME = "litterman-dashboard"
$IMAGE = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

gcloud builds submit --tag $IMAGE --project $PROJECT_ID

gcloud run deploy $SERVICE_NAME `
    --image $IMAGE `
    --platform managed `
    --region $REGION `
    --allow-unauthenticated `
    --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GEMINI_API_KEY=your_key_here" `
    --memory 512Mi `
    --port 8080 `
    --project $PROJECT_ID
```

---

## 📐 How Black-Litterman Works

The Black-Litterman model combines a market equilibrium prior with manager views extracted from news to produce posterior expected returns.

```
Market weights → Implied equilibrium returns (prior Π)
                            +
News → Gemini + Grounding → Views (P matrix · Q vector)
                            ↓
                 Posterior expected returns μ_BL
                            ↓
              SLSQP optimization (turnover ≤ 20pp)
                            ↓
                 Optimal portfolio weights
```

**Assets:** `Stocks_USA` · `Stocks_EM` · `Bonds_USA`
**Constraints:** turnover ≤ 20pp per asset · concentration ≤ 75%

---

## 🏆 Hackathon Submission

- **Challenge:** Gemini Live Agent Challenge 2026
- **Category:** Live Agents
- **Mandatory tech:** Gemini Live API · Google GenAI SDK · Cloud Run · Firestore · Google Search Grounding
- **Submission deadline:** March 16, 2026

*This project was built for the Gemini Live Agent Challenge. #GeminiLiveAgentChallenge*
