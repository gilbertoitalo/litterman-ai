# Litterman.ai — Black-Litterman Voice Co-Pilot

> **Gemini Live Agent Challenge — Live Agents Category**  
> A real-time voice-driven portfolio optimization agent for asset managers.

---

## What It Does

Litterman.ai is a voice agent that listens to financial market news, runs the **Black-Litterman portfolio optimization model**, and responds verbally with a rebalancing recommendation — all in real time.

A portfolio manager speaks naturally: *"The Fed raised rates 50 basis points today, citing persistent inflation."* The agent detects this as a market event, extracts quantitative views using Gemini, runs the Black-Litterman model, and responds aloud: *"Following the Fed decision, the model recommends moving Bonds USA down to 17.9 percent. Stocks USA moves up to 52.1 percent. The portfolio Sharpe ratio is negative 0.10, reflecting current headwinds."*

Simultaneously, a live dashboard hosted on Google Cloud Run updates in real time showing the new portfolio weights, extracted views, Sharpe ratio, and event history.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL (manager's machine)                 │
│                                                             │
│  Microphone → PyAudio → Gemini Live API (Native Audio)     │
│                              ↓                              │
│              Input transcription (real-time)               │
│                              ↓                              │
│         Gemini classifier: is this market news?            │
│                              ↓ YES                          │
│         Gemini 2.5 Flash: extract BL views (JSON)          │
│                              ↓                              │
│         Black-Litterman Engine (numpy/scipy)               │
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

**Google Cloud services used:**
- Gemini 2.5 Flash Native Audio (Live API) — voice input/output
- Gemini 2.5 Flash — view extraction and news classification
- Cloud Firestore — real-time shared state between agent and dashboard
- Cloud Run — hosts the Streamlit dashboard publicly
- Cloud Build — automated container image build
- Google Container Registry — stores the Docker image

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Voice I/O | Gemini 2.5 Flash Native Audio, PyAudio |
| AI SDK | Google GenAI SDK (`google-genai`) |
| Portfolio model | Black-Litterman (numpy, scipy) |
| State | Google Cloud Firestore |
| Dashboard | Streamlit + Plotly |
| Backend hosting | Google Cloud Run |
| Container build | Google Cloud Build |

---

## Project Structure

```
litterman-ai/
├── core/
│   ├── __init__.py
│   ├── voice_agent.py      # Gemini Live session, audio I/O, orchestration
│   ├── gemini_agent.py     # View extraction, BL pipeline, voice formatting
│   ├── bl_engine.py        # Black-Litterman model (pure numpy/scipy)
│   └── shared_state.py     # Firestore read/write bridge
├── .streamlit/
│   └── config.toml         # Streamlit server config
├── dashboard.py            # Streamlit dashboard (Cloud Run)
├── main.py                 # Entry point for local voice agent
├── Dockerfile              # Cloud Run container definition
├── deploy.sh               # Automated Cloud Run deploy script
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Live Dashboard

The dashboard is publicly accessible at:

```
https://litterman-dashboard-1084835415345.us-central1.run.app
```

It updates in real time as the voice agent processes market news locally and writes results to Firestore.

---

## Running Locally (Voice Agent)

### Prerequisites

- Python 3.11+
- A Google Cloud project with Firestore enabled
- A Gemini API key from [Google AI Studio](https://aistudio.google.com)
- A service account JSON with `Cloud Datastore User` role
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
> On Windows: `pip install pipwin && pipwin install pyaudio`  
> On Ubuntu: `sudo apt-get install portaudio19-dev && pip install pyaudio`  
> On Mac: `brew install portaudio && pip install pyaudio`

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
```

### 5. Run the voice agent

```bash
python main.py
```

The agent will greet you and start listening. Speak any financial market news and it will respond with a portfolio rebalancing recommendation.

**Example inputs:**
- *"The Federal Reserve raised interest rates by 50 basis points today."*
- *"US non-farm payrolls came in at 350,000, well above the 200,000 consensus."*
- *"China announced new stimulus measures worth 1 trillion yuan."*

### 6. Open the dashboard

Open `http://localhost:8501` in your browser (if running dashboard locally), or visit the Cloud Run URL above to see the live dashboard update in real time.

To run the dashboard locally:

```bash
streamlit run dashboard.py
```

---

## Cloud Deployment (Dashboard)

The dashboard is deployed to Google Cloud Run using the included automated deploy script.

### Prerequisites

- `gcloud` CLI installed and authenticated
- Docker not required — uses Cloud Build

### Deploy

```bash
# On Mac/Linux
chmod +x deploy.sh
./deploy.sh

# On Windows (PowerShell)
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

## How Black-Litterman Works

The Black-Litterman model combines a **market equilibrium prior** (what the market implies) with **manager views** (what Gemini extracts from the news) to produce a posterior expected return for each asset.

```
Market weights → Implied returns (prior)
                        +
News → Gemini → Views (P matrix, Q vector)
                        ↓
              Posterior expected returns
                        ↓
              Utility maximization (SLSQP)
                        ↓
              Optimal portfolio weights
```

The optimizer maximizes `μ'w - (δ/2) * w'Σw` subject to weights summing to 1, with a maximum turnover of 20 percentage points per asset from market weights. This ensures results are perturbations of market weights, not extreme concentrations.

---

## Hackathon Submission

- **Challenge:** Gemini Live Agent Challenge
- **Category:** Live Agents
- **Mandatory tech:** Gemini Live API, Google GenAI SDK, Google Cloud Run, Firestore
- **Submission deadline:** March 16, 2026

*This project was built for the Gemini Live Agent Challenge. #GeminiLiveAgentChallenge*