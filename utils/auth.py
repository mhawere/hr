from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from models.database import get_db
from models.employee import User

SECRET_KEY = "your-secret-key-here"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def _is_api_request(request: Request) -> bool:
    """Check if the request is an API request"""
    # Check for API indicators
    accept_header = request.headers.get("accept", "")
    content_type = request.headers.get("content-type", "")
    
    # If it's explicitly asking for JSON
    if "application/json" in accept_header or "application/json" in content_type:
        return True
    
    # Check URL patterns for API endpoints
    path = request.url.path
    if path.startswith("/api/") or path.startswith("/docs") or path.startswith("/openapi"):
        return True
    
    # Check for AJAX requests
    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return True
    
    return False

def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Get current user with automatic redirect for web requests"""
    
    token = request.cookies.get("access_token")
    
    # If no token, handle based on request type
    if not token:
        if _is_api_request(request):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            # For web requests, redirect to login
            raise HTTPException(
                status_code=status.HTTP_302_FOUND,
                detail="redirect:/auth/login?message=Please login to continue",
                headers={"Location": "/auth/login?message=Please login to continue"}
            )
    
    try:
        # Clean the token
        clean_token = token.replace("Bearer ", "")
        payload = jwt.decode(clean_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise JWTError("Invalid token payload")
    except JWTError:
        # Token is invalid, handle based on request type
        if _is_api_request(request):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            # For web requests, redirect to login with expired session message
            raise HTTPException(
                status_code=status.HTTP_302_FOUND,
                detail="redirect:/auth/login?message=Session expired, please login again",
                headers={"Location": "/auth/login?message=Session expired, please login again"}
            )
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        if _is_api_request(request):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_302_FOUND,
                detail="redirect:/auth/login?message=User not found, please login again",
                headers={"Location": "/auth/login?message=User not found, please login again"}
            )
    
    return user

# Optional: Create a separate function for API-only routes
def get_current_user_api(request: Request, db: Session = Depends(get_db)):
    """Get current user for API endpoints (always raises HTTPException)"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = request.cookies.get("access_token")
    if not token:
        raise credentials_exception
    
    try:
        clean_token = token.replace("Bearer ", "")
        payload = jwt.decode(clean_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user