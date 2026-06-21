import os
import asyncio
import traceback
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.utils.real_data_trainer import train_and_save_all
from app.utils.tabular_engine import TabularDiagnosticEngine
from app.utils.vision_engine import ECGVisionEngine
from app.utils.llm_engine import ClinicalReportingEngine

# Global Engine Instances
tabular_engine = None
vision_engine = None
llm_engine = None

app = FastAPI(
    title="Multimodal Heart Disease Diagnostic System",
    description="Production-grade clinical decision support integrating tabular vitals, ECG vision, and generative reporting.",
    version="1.0.0"
)

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global tabular_engine, vision_engine, llm_engine
    
    # Establish local directories
    assets_dir = os.path.abspath("./assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    xgb_file = os.path.join(assets_dir, "xgboost_model.json")
    cnn_file = os.path.join(assets_dir, "resnet_ecg.weights.h5")
    
    # Check asset availability; if not present, train and write synthetic clinical models
    if not os.path.exists(xgb_file) or not os.path.exists(cnn_file):
        await asyncio.to_thread(train_and_save_all, assets_dir)
        
    try:
        # Load and instantiate ML/AI engines
        tabular_engine = TabularDiagnosticEngine(model_dir=assets_dir)
        vision_engine = ECGVisionEngine(model_path=cnn_file)
        llm_engine = ClinicalReportingEngine()
        print("All diagnostic engines successfully initialized.")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize diagnostic engines: {str(e)}")
        traceback.print_exc()

@app.post("/api/v1/diagnose")
async def diagnose(
    age: float = Form(..., description="Patient age in years"),
    trestbps: float = Form(..., description="Resting blood pressure in mmHg"),
    chol: float = Form(..., description="Serum cholesterol in mg/dl"),
    thalach: float = Form(..., description="Maximum heart rate achieved in bpm"),
    ecg_image: UploadFile = File(..., description="Uploaded ECG report graph image")
) -> JSONResponse:
    """
    Accepts patient vitals and an ECG scan image.
    Standardizes numerical parameters and evaluates them using XGBoost + SHAP.
    Preprocesses the ECG scan image using OpenCV and evaluates it via ResNet/CNN.
    Asynchronously coordinates predictions and compiles an AI narrative clinical report.
    """
    if not ecg_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")
        
    try:
        # Read uploaded image bytes
        image_bytes = await ecg_image.read()
        
        # Concurrently execute Tabular predict and Vision predict to maximize performance
        tabular_task = asyncio.to_thread(
            tabular_engine.preprocess_and_predict, 
            age, trestbps, chol, thalach
        )
        vision_task = asyncio.to_thread(
            vision_engine.predict_ecg, 
            image_bytes
        )
        
        tabular_res, vision_res = await asyncio.gather(tabular_task, vision_task)
        
        # Compile a combined risk score
        overall_risk_score = (tabular_res["risk_probability"] + vision_res["ecg_risk_probability"]) / 2.0
        
        # Generate LLM narrative report
        clinical_report = await asyncio.to_thread(
            llm_engine.generate_report, 
            tabular_res, vision_res
        )
        
        return JSONResponse(content={
            "success": True,
            "overall_risk_score": overall_risk_score,
            "tabular_analysis": tabular_res,
            "vision_analysis": vision_res,
            "clinical_report": clinical_report
        })
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "detail": "An internal diagnostic pipeline exception occurred."
            }
        )

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """
    Serves a modern, premium HTML/CSS user interface for clinical practitioners.
    Includes input fields, image dropzone, loading states, and structured output renderers.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multimodal Heart Disease Diagnostic System</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <!-- Markdown Parser -->
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            :root {
                --bg-dark: #070a13;
                --bg-card: rgba(15, 23, 42, 0.6);
                --bg-input: #101726;
                --primary: #ff4757;
                --primary-gradient: linear-gradient(135deg, #ff4757, #ff6b81);
                --accent-blue: #00d2d3;
                --accent-green: #1dd1a1;
                --border-color: rgba(255, 255, 255, 0.08);
                --text-main: #f1f2f6;
                --text-muted: #a4b0be;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                background-color: var(--bg-dark);
                background-image: radial-gradient(circle at 10% 20%, rgba(255, 71, 87, 0.05) 0%, transparent 40%),
                                  radial-gradient(circle at 90% 80%, rgba(0, 210, 211, 0.04) 0%, transparent 40%);
                color: var(--text-main);
                font-family: 'Plus Jakarta Sans', sans-serif;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                overflow-x: hidden;
            }

            header {
                backdrop-filter: blur(12px);
                background-color: rgba(7, 10, 19, 0.8);
                border-bottom: 1px solid var(--border-color);
                padding: 1.25rem 2rem;
                position: sticky;
                top: 0;
                z-index: 100;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .logo-container {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }

            .logo-icon {
                color: var(--primary);
                font-size: 1.8rem;
                animation: pulse 2s infinite alternate;
            }

            .logo-text {
                font-weight: 800;
                font-size: 1.4rem;
                letter-spacing: -0.5px;
                background: linear-gradient(90deg, #fff, var(--text-muted));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .system-badge {
                background: rgba(255, 71, 87, 0.1);
                border: 1px solid rgba(255, 71, 87, 0.2);
                color: var(--primary);
                font-size: 0.75rem;
                font-weight: 700;
                padding: 0.4rem 0.8rem;
                border-radius: 20px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            main {
                flex: 1;
                max-width: 1400px;
                width: 100%;
                margin: 0 auto;
                padding: 2.5rem 2rem;
                display: grid;
                grid-template-columns: 1fr 1.2fr;
                gap: 2.5rem;
            }

            @media (max-width: 1024px) {
                main {
                    grid-template-columns: 1fr;
                }
            }

            .section-title {
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                letter-spacing: -0.3px;
            }

            .card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 2rem;
                backdrop-filter: blur(8px);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                transition: transform 0.3s ease, border-color 0.3s ease;
            }

            .card:hover {
                border-color: rgba(255, 71, 87, 0.2);
            }

            form {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }

            .input-group {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.25rem;
            }

            .field {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }

            label {
                font-size: 0.85rem;
                font-weight: 600;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            input[type="number"] {
                background-color: var(--bg-input);
                border: 1px solid var(--border-color);
                border-radius: 10px;
                color: var(--text-main);
                font-family: inherit;
                font-size: 1rem;
                padding: 0.85rem 1rem;
                transition: border-color 0.2s, box-shadow 0.2s;
                outline: none;
            }

            input[type="number"]:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(255, 71, 87, 0.15);
            }

            .upload-zone {
                border: 2px dashed rgba(255, 255, 255, 0.15);
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                cursor: pointer;
                background-color: rgba(16, 23, 38, 0.4);
                transition: border-color 0.3s, background-color 0.3s;
                position: relative;
            }

            .upload-zone:hover {
                border-color: var(--primary);
                background-color: rgba(255, 71, 87, 0.02);
            }

            .upload-zone.dragover {
                border-color: var(--accent-blue);
                background-color: rgba(0, 210, 211, 0.05);
            }

            .upload-icon {
                font-size: 2.5rem;
                color: var(--text-muted);
                margin-bottom: 0.75rem;
                transition: color 0.3s;
            }

            .upload-zone:hover .upload-icon {
                color: var(--primary);
            }

            .upload-text {
                font-size: 0.95rem;
                font-weight: 500;
                color: var(--text-main);
            }

            .upload-subtext {
                font-size: 0.8rem;
                color: var(--text-muted);
                margin-top: 0.25rem;
            }

            input[type="file"] {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0;
                cursor: pointer;
            }

            .file-preview {
                margin-top: 1rem;
                font-size: 0.85rem;
                color: var(--accent-blue);
                display: none;
                align-items: center;
                gap: 0.5rem;
                justify-content: center;
                background: rgba(0, 210, 211, 0.08);
                padding: 0.5rem;
                border-radius: 6px;
                border: 1px solid rgba(0, 210, 211, 0.15);
            }

            .submit-btn {
                background: var(--primary-gradient);
                border: none;
                border-radius: 12px;
                color: white;
                font-family: inherit;
                font-size: 1.1rem;
                font-weight: 700;
                padding: 1rem;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 0.75rem;
            }

            .submit-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(255, 71, 87, 0.3);
            }

            .submit-btn:active {
                transform: translateY(0);
            }

            .submit-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }

            /* Loader Styling */
            .spinner {
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 1s ease-in-out infinite;
                display: none;
            }

            /* Right Pane (Results) */
            .results-wrapper {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
                opacity: 0.7;
                transition: opacity 0.3s ease;
            }

            .results-wrapper.active {
                opacity: 1;
            }

            .empty-state {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                padding: 6rem 2rem;
                border: 1px dashed var(--border-color);
                border-radius: 16px;
                background: rgba(15, 23, 42, 0.2);
            }

            .empty-icon {
                font-size: 4rem;
                color: rgba(255, 255, 255, 0.05);
                margin-bottom: 1.5rem;
            }

            .empty-text {
                color: var(--text-muted);
                font-size: 1.1rem;
                font-weight: 500;
            }

            /* Visual Risk Gauges */
            .metrics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 1.25rem;
                margin-bottom: 1.5rem;
            }

            .metric-card {
                background: rgba(16, 23, 38, 0.6);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 1.25rem;
                text-align: center;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.5rem;
            }

            .metric-val {
                font-size: 1.8rem;
                font-weight: 800;
                font-family: 'JetBrains Mono', monospace;
            }

            .metric-label {
                font-size: 0.75rem;
                font-weight: 700;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .risk-critical { color: #ff4757; }
            .risk-moderate { color: #ffa502; }
            .risk-safe { color: var(--accent-green); }

            /* SHAP Feature Weights */
            .shap-section {
                background: rgba(16, 23, 38, 0.4);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
            }

            .shap-title {
                font-size: 0.95rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 1rem;
                color: var(--accent-blue);
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .shap-list {
                display: flex;
                flex-direction: column;
                gap: 0.85rem;
            }

            .shap-row {
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
            }

            .shap-meta {
                display: flex;
                justify-content: space-between;
                font-size: 0.85rem;
                font-weight: 600;
            }

            .shap-bar-bg {
                background: rgba(255, 255, 255, 0.05);
                height: 8px;
                border-radius: 4px;
                overflow: hidden;
            }

            .shap-bar-fill {
                height: 100%;
                border-radius: 4px;
                transition: width 1s ease-out;
            }

            /* Report Panel */
            .report-card {
                background: rgba(16, 23, 38, 0.8);
                border: 1px solid rgba(0, 210, 211, 0.15);
                border-radius: 16px;
                padding: 2rem;
                box-shadow: inset 0 0 20px rgba(0, 210, 211, 0.05);
            }

            .report-header {
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 1rem;
                margin-bottom: 1.5rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .report-title {
                font-size: 1.1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: var(--text-main);
            }

            .report-content {
                font-size: 0.95rem;
                line-height: 1.6;
                color: var(--text-main);
            }

            .report-content h1, .report-content h2, .report-content h3 {
                margin: 1.5rem 0 0.75rem 0;
                color: var(--accent-blue);
            }

            .report-content h1:first-child {
                margin-top: 0;
            }

            .report-content p {
                margin-bottom: 1rem;
            }

            .report-content ul, .report-content ol {
                margin-left: 1.5rem;
                margin-bottom: 1rem;
            }

            .report-content li {
                margin-bottom: 0.35rem;
            }

            .report-content blockquote {
                background: rgba(255, 71, 87, 0.05);
                border-left: 4px solid var(--primary);
                padding: 1rem;
                border-radius: 0 8px 8px 0;
                margin: 1.5rem 0;
                font-style: italic;
            }

            .report-content hr {
                border: 0;
                height: 1px;
                background: var(--border-color);
                margin: 2rem 0;
            }

            /* Animations */
            @keyframes pulse {
                0% { transform: scale(1); opacity: 0.8; }
                100% { transform: scale(1.08); opacity: 1; }
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .animated-fade {
                animation: fadeIn 0.4s ease forwards;
            }
        </style>
    </head>
    <body>
        <header>
            <div class="logo-container">
                <span class="logo-icon">❤️</span>
                <span class="logo-text">AegisMultimodal</span>
            </div>
            <div class="system-badge">Precision Diagnostic Hub</div>
        </header>

        <main>
            <!-- Left Column: Form Intake -->
            <section class="card">
                <h2 class="section-title"><span>📥</span> Patient Intake & Diagnostics</h2>
                <form id="diagnose-form" enctype="multipart/form-data">
                    <div class="input-group">
                        <div class="field">
                            <label for="age">Patient Age (years)</label>
                            <input type="number" id="age" name="age" required min="1" max="120" value="58">
                        </div>
                        <div class="field">
                            <label for="trestbps">Blood Pressure (mmHg)</label>
                            <input type="number" id="trestbps" name="trestbps" required min="50" max="250" value="135">
                        </div>
                    </div>

                    <div class="input-group">
                        <div class="field">
                            <label for="chol">Serum Cholesterol (mg/dl)</label>
                            <input type="number" id="chol" name="chol" required min="100" max="600" value="254">
                        </div>
                        <div class="field">
                            <label for="thalach">Max Heart Rate (bpm)</label>
                            <input type="number" id="thalach" name="thalach" required min="50" max="250" value="142">
                        </div>
                    </div>

                    <div class="field">
                        <label>Electrocardiogram (ECG) Report Image</label>
                        <div class="upload-zone" id="drop-zone">
                            <div class="upload-icon">📄</div>
                            <div class="upload-text">Drag & drop clinical ECG report image here</div>
                            <div class="upload-subtext">Supports PNG, JPG, JPEG (automatically filters grids)</div>
                            <input type="file" id="ecg_image" name="ecg_image" accept="image/*" required>
                        </div>
                        <div class="file-preview" id="file-indicator">
                            <span>📄</span> <span id="file-name">filename.jpg</span>
                        </div>
                    </div>

                    <button type="submit" class="submit-btn" id="submit-btn">
                        <div class="spinner" id="btn-spinner"></div>
                        <span id="btn-text">Execute Multimodal Assessment</span>
                    </button>
                </form>
            </section>

            <!-- Right Column: Interactive Diagnostic Report -->
            <section>
                <div class="results-wrapper" id="results-wrapper">
                    <!-- Default Screen -->
                    <div class="empty-state" id="empty-state">
                        <div class="empty-icon">📊</div>
                        <p class="empty-text">Awaiting diagnostic execution. Enter patient vitals and submit an ECG report to generate results.</p>
                    </div>

                    <!-- Output Elements -->
                    <div id="diagnostics-content" style="display: none;">
                        <!-- Overall score dashboard -->
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <span class="metric-label">Composite Risk</span>
                                <div class="metric-val" id="val-composite">0%</div>
                            </div>
                            <div class="metric-card">
                                <span class="metric-label">Vitals Risk</span>
                                <div class="metric-val" id="val-tabular">0%</div>
                            </div>
                            <div class="metric-card">
                                <span class="metric-label">ECG Signal Risk</span>
                                <div class="metric-val" id="val-vision">0%</div>
                            </div>
                        </div>

                        <!-- SHAP explaining vitals -->
                        <div class="shap-section">
                            <div class="shap-title">
                                <span>🔑 SHAP Risk Contributions</span>
                                <span style="font-size: 0.75rem; text-transform: none; color: var(--text-muted);">Tabular Feature Importance</span>
                            </div>
                            <div class="shap-list" id="shap-list">
                                <!-- Generated Dynamically -->
                            </div>
                        </div>

                        <!-- AI Clinical Assessment Summary -->
                        <div class="report-card animated-fade">
                            <div class="report-header">
                                <span class="report-title">📝 AI Narrative Assessment</span>
                                <span style="font-size: 0.75rem; color: var(--accent-green); font-weight: 700;">Verified Narrative</span>
                            </div>
                            <div class="report-content" id="report-rendered">
                                <!-- Rendered Markdown -->
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </main>

        <script>
            // Handle Drag & Drop indicators
            const fileInput = document.getElementById('ecg_image');
            const dropZone = document.getElementById('drop-zone');
            const fileIndicator = document.getElementById('file-indicator');
            const fileNameSpan = document.getElementById('file-name');

            fileInput.addEventListener('change', (e) => {
                if (fileInput.files.length > 0) {
                    fileNameSpan.textContent = fileInput.files[0].name;
                    fileIndicator.style.display = 'flex';
                }
            });

            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dropZone.classList.add('dragover');
                }, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    dropZone.classList.remove('dragover');
                }, false);
            });

            // Handle submission
            const form = document.getElementById('diagnose-form');
            const submitBtn = document.getElementById('submit-btn');
            const btnText = document.getElementById('btn-text');
            const btnSpinner = document.getElementById('btn-spinner');
            const resultsWrapper = document.getElementById('results-wrapper');
            const emptyState = document.getElementById('empty-state');
            const diagnosticsContent = document.getElementById('diagnostics-content');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();

                // Show spinner
                submitBtn.disabled = true;
                btnText.textContent = "Analyzing Patient Datastreams...";
                btnSpinner.style.display = 'block';
                
                const formData = new FormData(form);

                try {
                    const response = await fetch('/api/v1/diagnose', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    
                    if (response.ok && data.success) {
                        renderResults(data);
                    } else {
                        alert("Diagnostic pipeline error: " + (data.error || "Unknown server error"));
                    }
                } catch (err) {
                    console.error(err);
                    alert("Network connection error to diagnostic backend.");
                } finally {
                    submitBtn.disabled = false;
                    btnText.textContent = "Execute Multimodal Assessment";
                    btnSpinner.style.display = 'none';
                }
            });

            function renderResults(data) {
                // Activate Wrapper panel
                resultsWrapper.classList.add('active');
                emptyState.style.display = 'none';
                diagnosticsContent.style.display = 'block';

                // Fill Gauges
                updateGauge('val-composite', data.overall_risk_score);
                updateGauge('val-tabular', data.tabular_analysis.risk_probability);
                updateGauge('val-vision', data.vision_analysis.ecg_risk_probability);

                // Render SHAP features
                const shapList = document.getElementById('shap-list');
                shapList.innerHTML = '';

                const shapVals = data.tabular_analysis.shap_values;
                
                // Map logical labels
                const labels = {
                    "Age": "Patient Age",
                    "RestingBP": "Resting Blood Pressure",
                    "Cholesterol": "Serum Cholesterol",
                    "MaxHR": "Maximum Heart Rate"
                };

                // Find max SHAP value for scaling bars
                let maxShap = 0.01;
                for (const key in shapVals) {
                    const absVal = Math.abs(shapVals[key]);
                    if (absVal > maxShap) maxShap = absVal;
                }

                for (const key in shapVals) {
                    const val = shapVals[key];
                    const label = labels[key] || key;
                    const percentage = Math.min(100, Math.max(5, (Math.abs(val) / maxShap) * 100));
                    
                    const isAggravating = val > 0.05;
                    const color = isAggravating ? 'var(--primary)' : 'var(--accent-green)';
                    const sign = val >= 0 ? '+' : '';

                    const row = document.createElement('div');
                    row.className = 'shap-row';
                    row.innerHTML = `
                        <div class="shap-meta">
                            <span>${label}</span>
                            <span style="color: ${color};">${sign}${val.toFixed(3)}</span>
                        </div>
                        <div class="shap-bar-bg">
                            <div class="shap-bar-fill" style="width: ${percentage}%; background-color: ${color};"></div>
                        </div>
                    `;
                    shapList.appendChild(row);
                }

                // Render Generative Markdown Report
                const rawReport = data.clinical_report;
                document.getElementById('report-rendered').innerHTML = marked.parse(rawReport);
            }

            function updateGauge(elementId, value) {
                const element = document.getElementById(elementId);
                const percent = (value * 100).toFixed(1);
                element.textContent = `${percent}%`;
                
                // Apply color class
                element.className = 'metric-val';
                if (value > 0.6) {
                    element.classList.add('risk-critical');
                } else if (value > 0.35) {
                    element.classList.add('risk-moderate');
                } else {
                    element.classList.add('risk-safe');
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content
