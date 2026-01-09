from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

from src.inference import predict_disease

app = FastAPI(title="Animal Disease Predictor")

# Allow cross-origin requests from local dev servers (e.g., Live Server on :5500)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class SampleInput(BaseModel):
    Animal: str
    Age: float
    Gender: str
    Breed: str
    WBC: Optional[float] = None
    RBC: Optional[float] = None
    Hemoglobin: Optional[float] = None
    Platelets: Optional[float] = None
    Glucose: Optional[float] = None
    ALT: Optional[float] = None
    AST: Optional[float] = None
    Urea: Optional[float] = None
    Creatinine: Optional[float] = None
    Symptom_Fever: Optional[int] = Field(default=0, ge=0, le=1)
    Symptom_Lethargy: Optional[int] = Field(default=0, ge=0, le=1)
    Symptom_Vomiting: Optional[int] = Field(default=0, ge=0, le=1)
    Symptom_Diarrhea: Optional[int] = Field(default=0, ge=0, le=1)
    Symptom_WeightLoss: Optional[int] = Field(default=0, ge=0, le=1)
    Symptom_SkinLesion: Optional[int] = Field(default=0, ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(sample: SampleInput):
    try:
        inp = sample.dict()
        result = predict_disease(inp)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail="Index not found")
