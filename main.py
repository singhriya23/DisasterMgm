from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional
from datetime import datetime

# Import your existing workflow
from langraph import app as workflow_app  # Replace with your actual module name

app = FastAPI(
    title="Disaster Analysis API",
    description="API for analyzing disaster data and generating reports",
    version="1.0.0"
)

class AnalysisRequest(BaseModel):
    prompt: str
    save_report: Optional[bool] = True
    output_dir: Optional[str] = "reports"

class AnalysisResponse(BaseModel):
    status: str
    message: str
    dashboard_path: Optional[str] = None
    report_path: Optional[str] = None
    timestamp: str

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_disaster(request: AnalysisRequest):
    """
    Process a natural language prompt about disaster analysis.
    Returns paths to generated visualizations and reports.
    """
    try:
        # Create output directory if it doesn't exist
        if request.save_report and not os.path.exists(request.output_dir):
            os.makedirs(request.output_dir)
        
        # Run the workflow
        result = workflow_app.invoke({"prompt": request.prompt})
        
        # Prepare response
        response_data = {
            "status": "success" if "validation_error" not in result else "error",
            "message": result["output"],
            "dashboard_path": result.get("dashboard_path"),
            "report_path": result.get("report_path"),
            "timestamp": datetime.now().isoformat()
        }
        
        return response_data
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/download-report")
async def download_report(path: str):
    """
    Download a generated report file.
    """
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )
    
    return FileResponse(
        path,
        media_type='application/pdf',
        filename=os.path.basename(path)
    )

@app.get("/download-dashboard")
async def download_dashboard(path: str):
    """
    Download a generated dashboard file.
    """
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )
    
    return FileResponse(
        path,
        media_type='application/html',
        filename=os.path.basename(path)
    )

@app.get("/health")
async def health_check():
    """
    Service health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

