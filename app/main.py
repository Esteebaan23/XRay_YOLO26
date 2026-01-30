import psutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from app.service import XRayService
from app.schemas import XRayOutput

app = FastAPI(
    title="X-Ray Anomaly Detection API",
    version="1.0.0"
)

service = None

# --- HTML DEL FRONTEND (Incrustado para facilitar el deployment) ---
html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X-Ray AI Diagnostic</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-900 text-white min-h-screen font-sans">

    <nav class="bg-slate-800 border-b border-slate-700 p-4">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-2xl font-bold text-blue-400"><i class="fa-solid fa-user-doctor mr-2"></i>X-Ray AI Diagnostic</h1>
            <span class="text-xs bg-blue-900 text-blue-300 py-1 px-3 rounded-full">Model: YOLO26-Cls (CPU Optimized)</span>
        </div>
    </nav>

    <div class="container mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        <div class="lg:col-span-1 space-y-6">
            <div class="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700">
                <h2 class="text-xl font-semibold mb-4 text-gray-200">1. Upload X-ray </h2>
                
                <div class="flex items-center justify-center w-full">
                    <label for="dropzone-file" class="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer bg-slate-700 hover:bg-slate-600 transition">
                        <div class="flex flex-col items-center justify-center pt-5 pb-6">
                            <i class="fa-solid fa-cloud-arrow-up text-4xl text-gray-400 mb-3"></i>
                            <p class="mb-2 text-sm text-gray-400"><span class="font-semibold">Click to upload</span> or drag</p>
                            <p class="text-xs text-gray-500">PNG, JPG (MAX. 800x800px)</p>
                        </div>
                        <input id="dropzone-file" type="file" class="hidden" accept="image/*" onchange="previewImage(event)" />
                    </label>
                </div> 

                <div class="mt-4 flex items-center">
                    <input id="gradcam-check" type="checkbox" value="" class="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-600 ring-offset-gray-800">
                    <label for="gradcam-check" class="ml-2 text-sm font-medium text-gray-300">Generate Explainability (Grad-CAM)</label>
                </div>
                <p class="text-xs text-gray-500 mt-1 ml-6">* Verify which areas the model focuses on.</p>

                <button onclick="analyzeXray()" class="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition flex justify-center items-center">
                    <span id="btn-text">Analize Image</span>
                    <i id="btn-spinner" class="fa-solid fa-circle-notch fa-spin ml-2 hidden"></i>
                </button>
            </div>
        </div>

        <div class="lg:col-span-2 space-y-6">
            
            <div id="welcome-state" class="bg-slate-800 p-10 rounded-xl shadow-lg border border-slate-700 text-center flex flex-col items-center justify-center h-full min-h-[400px]">
                <i class="fa-solid fa-microscope text-6xl text-slate-600 mb-4"></i>
                <h3 class="text-xl text-gray-400">Waiting for image for analysis...</h3>
            </div>

            <div id="result-state" class="hidden space-y-6">
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div id="card-pred" class="bg-slate-800 p-4 rounded-lg border-l-4 border-gray-500 shadow-md">
                        <p class="text-gray-400 text-xs uppercase font-bold">Diagnosis</p>
                        <p id="res-label" class="text-2xl font-bold text-white mt-1">--</p>
                    </div>
                    <div class="bg-slate-800 p-4 rounded-lg border border-slate-700 shadow-md">
                        <p class="text-gray-400 text-xs uppercase font-bold">Confidence AI</p>
                        <div class="flex items-end">
                            <p id="res-conf" class="text-2xl font-bold text-blue-400 mt-1">--</p>
                            <span class="text-sm text-gray-500 mb-1 ml-1">%</span>
                        </div>
                    </div>
                    <div class="bg-slate-800 p-4 rounded-lg border border-slate-700 shadow-md">
                        <p class="text-gray-400 text-xs uppercase font-bold">Total Latency</p>
                        <p id="res-time" class="text-2xl font-bold text-green-400 mt-1">-- ms</p>
                    </div>
                </div>

                <div class="bg-slate-800 p-4 rounded-xl border border-slate-700">
                    <h3 class="text-lg font-semibold mb-4 text-gray-200">Medical Visualization</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="relative group">
                            <span class="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">Original (Processed)</span>
                            <img id="img-original" class="w-full h-auto rounded-lg border border-slate-600" src="" alt="Original">
                        </div>
                        <div id="gradcam-container" class="relative group hidden">
                            <span class="absolute top-2 left-2 bg-red-600/90 text-white text-xs px-2 py-1 rounded">Grad-CAM (AI Focus)</span>
                            <img id="img-gradcam" class="w-full h-auto rounded-lg border border-slate-600" src="" alt="GradCAM">
                        </div>
                        <div id="gradcam-placeholder" class="flex items-center justify-center bg-slate-900 rounded-lg border border-slate-700 h-64 text-gray-500 text-sm p-4 text-center">
                            Grad-CAM disabled.<br>Check the box to see where the AI is focusing.
                        </div>
                    </div>
                </div>

//                <details class="bg-slate-900 p-4 rounded-lg border border-slate-800">
          //          <summary class="cursor-pointer text-xs text-gray-500 font-mono">Ver JSON Respuesta (Debug)</summary>
          //          <pre id="json-debug" class="text-xs text-green-500 mt-2 overflow-x-auto"></pre>
         //       </details>

            </div>
        </div>
    </div>

    <script>
        // Previsualización local
        function previewImage(event) {
            const file = event.target.files[0];
            if(file){
                // Solo feedback visual de que se seleccionó algo
                const btn = document.getElementById('btn-text');
                btn.innerText = "Analize: " + file.name.substring(0, 15) + "...";
            }
        }

        async function analyzeXray() {
            const fileInput = document.getElementById('dropzone-file');
            const useGradcam = document.getElementById('gradcam-check').checked;
            
            if (fileInput.files.length === 0) {
                alert("Por favor selecciona una imagen primero.");
                return;
            }

            // UI Loading State
            document.getElementById('btn-spinner').classList.remove('hidden');
            document.getElementById('btn-text').innerText = "Processing...";
            
            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            try {
                // Petición a la API
                const url = `/predict?include_gradcam=${useGradcam}`;
                const response = await fetch(url, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error("Server Error");

                const data = await response.json();
                renderResults(data);

            } catch (error) {
                alert("Error while analyzing: " + error.message);
                console.error(error);
            } finally {
                // Reset UI
                document.getElementById('btn-spinner').classList.add('hidden');
                document.getElementById('btn-text').innerText = "Analyze Image";
            }
        }

        function renderResults(data) {
            document.getElementById('welcome-state').classList.add('hidden');
            document.getElementById('result-state').classList.remove('hidden');

            // 1. Predicción y Color
            const label = data.prediction.label;
            const predEl = document.getElementById('res-label');
            const cardPred = document.getElementById('card-pred');
            
            predEl.innerText = label.toUpperCase();
            
            // Lógica de colores (Asumiendo 'anomaly' vs 'normal')
            if(label.toLowerCase().includes("anomaly")) {
                cardPred.className = "bg-slate-800 p-4 rounded-lg border-l-4 border-red-500 shadow-md bg-red-900/10";
                predEl.className = "text-2xl font-bold text-red-400 mt-1";
            } else {
                cardPred.className = "bg-slate-800 p-4 rounded-lg border-l-4 border-green-500 shadow-md bg-green-900/10";
                predEl.className = "text-2xl font-bold text-green-400 mt-1";
            }

            // 2. Métricas
            document.getElementById('res-conf').innerText = (data.prediction.confidence * 100).toFixed(1);
            document.getElementById('res-time').innerText = data.performance.total_latency_ms + " ms";

            // 3. Imágenes Base64
            // Nota: El backend debe devolver 'original_processed_base64' (si lo configuramos) 
            // Si tu backend NO devuelve la original procesada, usa la local.
            // Para este ejemplo, usaremos la respuesta del backend si existe.
            
            if (data.original_processed_base64) {
                 document.getElementById('img-original').src = "data:image/jpeg;base64," + data.original_processed_base64;
            } else {
                // Fallback: mostrar la que subió el usuario localmente
                 const file = document.getElementById('dropzone-file').files[0];
                 document.getElementById('img-original').src = URL.createObjectURL(file);
            }

            // 4. GradCAM
            const gradContainer = document.getElementById('gradcam-container');
            const gradPlaceholder = document.getElementById('gradcam-placeholder');
            
            if (data.explainability && data.explainability.overlay_base64) {
                gradContainer.classList.remove('hidden');
                gradPlaceholder.classList.add('hidden');
                document.getElementById('img-gradcam').src = "data:image/jpeg;base64," + data.explainability.overlay_base64;
            } else {
                gradContainer.classList.add('hidden');
                gradPlaceholder.classList.remove('hidden');
            }

            // 5. JSON Raw
            document.getElementById('json-debug').innerText = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
"""

@app.on_event("startup")
def startup_event():
    global service
    print("🚀 Levantando servicio y cargando modelo IA...")
    service = XRayService()
    print("✅ Servicio listo para recibir peticiones.")

# --- RUTA DE INICIO (HOME) ---
@app.get("/", response_class=HTMLResponse)
async def home():
    """Sirve la interfaz gráfica en la raíz"""
    return html_content

@app.post("/predict", response_model=XRayOutput)
async def predict_xray(
    file: UploadFile = File(..., description="Archivo de imagen"),
    include_gradcam: bool = Query(False)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")
    
    try:
        content = await file.read()
        result = await run_in_threadpool(
            service.analyze, 
            image_bytes=content, 
            filename=file.filename, 
            include_gradcam=include_gradcam
        )
        return result
    except Exception as e:
        print(f"❌ Internal Error: {e}")
        raise HTTPException(status_code=500, detail="Internal error processing the image.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "cpu_usage_percent": psutil.cpu_percent(),
        "ram_usage_percent": psutil.virtual_memory().percent
    }