from fastapi import APIRouter, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt

from models.database import get_db
from models.employee import User
from utils.auth import create_access_token, get_current_user

router = APIRouter()
templates = Jinja2Templates(directory="templates")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, message: str = None):
    """Display login page with optional message"""
    context = {
        "request": request,
        "message": message,
        "page_title": "Login"
    }
    return templates.TemplateResponse("login.html", context)

@router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()
    
    if not user or not pwd_context.verify(password, user.hashed_password):
        return templates.TemplateResponse(
            "login.html", 
            {
                "request": request, 
                "error": "Invalid username or password",
                "username": username,
                "page_title": "Login"
            }
        )
    
    access_token = create_access_token(data={"sub": user.username})
    response = RedirectResponse(url="/dashboard/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key="access_token", 
        value=f"Bearer {access_token}", 
        httponly=True,
        max_age=1800,  # 30 minutes
        samesite="lax"
    )
    return response

@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/auth/login?message=You have been logged out successfully", status_code=302)
    response.delete_cookie("access_token")
    return response

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Check if user exists
    if db.query(User).filter(User.username == username).first():
        return templates.TemplateResponse(
            "register.html", 
            {"request": request, "error": "Username already exists"}
        )
    
    # Create new user
    hashed_password = pwd_context.hash(password)
    new_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    
    return RedirectResponse(url="/auth/login?message=Registration successful! Please login.", status_code=status.HTTP_302_FOUND)