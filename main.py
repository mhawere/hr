from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
from routers import biometric
from routers import leave
from routers import auth, dashboard, staff, departments, attendance, settings, users, performance
from models.database import create_tables
from routers import reports
from routers import ai


# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_tables()
    yield
    # Shutdown (if needed)

# Create FastAPI app with lifespan
app = FastAPI(
    title="HR Management System", 
    description="Comprehensive HR Application with Department Management and Attendance System",
    lifespan=lifespan
)

# Setup paths
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Exception handlers for authentication redirects
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with custom redirect logic"""
    
    # Check if it's a redirect exception (status 302 with redirect detail)
    if exc.status_code == 302 and exc.detail and exc.detail.startswith("redirect:"):
        # Extract the redirect URL
        redirect_url = exc.detail.replace("redirect:", "")
        response = RedirectResponse(url=redirect_url, status_code=302)
        # Clear the invalid token cookie
        response.delete_cookie("access_token")
        return response
    
    # For API requests or other errors, return JSON
    if exc.status_code == 401:
        # Check if it's likely an API request
        accept_header = request.headers.get("accept", "")
        if "application/json" in accept_header or request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )
        else:
            # Fallback redirect for web requests
            response = RedirectResponse(url="/auth/login?message=Session expired, please login again", status_code=302)
            response.delete_cookie("access_token")
            return response
    
    # Default JSON response for other HTTP exceptions
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(StarletteHTTPException)
async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle Starlette HTTP exceptions"""
    
    # Same logic as HTTP exception handler
    if exc.status_code == 302 and hasattr(exc, 'detail') and exc.detail and str(exc.detail).startswith("redirect:"):
        redirect_url = str(exc.detail).replace("redirect:", "")
        response = RedirectResponse(url=redirect_url, status_code=302)
        response.delete_cookie("access_token")
        return response
    
    if exc.status_code == 401:
        accept_header = request.headers.get("accept", "")
        if "application/json" in accept_header or request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": str(exc.detail)}
            )
        else:
            response = RedirectResponse(url="/auth/login?message=Session expired, please login again", status_code=302)
            response.delete_cookie("access_token")
            return response
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)}
    )

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
app.include_router(staff.router, prefix="/staff", tags=["Staff Management"])
app.include_router(departments.router, prefix="/departments", tags=["Department Management"])
app.include_router(attendance.router, prefix="/attendance", tags=["Attendance Management"])
app.include_router(settings.router, prefix="/settings", tags=["Settings"])
app.include_router(users.router, prefix="/users", tags=["Users Management"])
app.include_router(performance.router, prefix="/performance", tags=["Performance Management"])  # Fixed: Added prefix
app.include_router(biometric.router)
app.include_router(leave.router, prefix="/leave", tags=["leave"])
app.include_router(reports.router)
app.include_router(ai.ai, prefix="/ai", tags=["AI"])





@app.get("/")
async def root():
    return RedirectResponse(url="/auth/login")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8123, reload=True)