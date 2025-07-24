"""
Users Management Router
System users administration and management for the HR system.
"""

import logging
from typing import Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from passlib.context import CryptContext

from fastapi import (
    APIRouter, Request, Depends, HTTPException, Form, Query
)
from fastapi import status as http_status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from models.database import get_db
from models.employee import User
from utils.auth import get_current_user, create_access_token

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.get("/", response_class=HTMLResponse)
async def users_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=5, le=100)
):
    """Users management dashboard page"""
    try:
        # Build query
        query = db.query(User)
        
        # Apply search filter
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (User.username.ilike(search_term)) |
                (User.email.ilike(search_term))
            )
        
        # Get total count
        total_users = query.count()
        
        # Apply pagination
        offset = (page - 1) * per_page
        users = query.offset(offset).limit(per_page).all()
        
        # Calculate pagination info
        total_pages = (total_users + per_page - 1) // per_page
        has_next = page < total_pages
        has_prev = page > 1
        
        context = {
            "request": request,
            "user": current_user,
            "page_title": "Users Management",
            "users": users,
            "search": search or "",
            "page": page,
            "per_page": per_page,
            "total_users": total_users,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev,
            "start_index": offset + 1,
            "end_index": min(offset + per_page, total_users)
        }
        return templates.TemplateResponse("users.html", context)
        
    except Exception as e:
        logger.error(f"Error loading users page: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading users"
        )

@router.post("/add")
async def add_user(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    is_active: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a new user"""
    try:
        # Check if username already exists
        if db.query(User).filter(User.username == username).first():
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Username already exists"}
            )
        
        # Check if email already exists
        if db.query(User).filter(User.email == email).first():
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Email already exists"}
            )
        
        # Hash password
        hashed_password = pwd_context.hash(password)
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            is_active=is_active,
            created_at=datetime.utcnow()
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"User {username} created successfully by {current_user.username}")
        
        return JSONResponse(
            status_code=201,
            content={"success": True, "message": "User created successfully"}
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating user: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Error creating user"}
        )

@router.get("/{user_id}")
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user details"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "User not found"}
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat() if user.created_at else None
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Error retrieving user"}
        )

@router.post("/{user_id}/edit")
async def edit_user(
    user_id: int,
    username: str = Form(...),
    email: str = Form(...),
    is_active: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Edit an existing user"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "User not found"}
            )
        
        # Check if username is taken by another user
        existing_user = db.query(User).filter(
            User.username == username,
            User.id != user_id
        ).first()
        if existing_user:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Username already exists"}
            )
        
        # Check if email is taken by another user
        existing_email = db.query(User).filter(
            User.email == email,
            User.id != user_id
        ).first()
        if existing_email:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Email already exists"}
            )
        
        # Update user
        user.username = username
        user.email = email
        user.is_active = is_active
        
        db.commit()
        
        logger.info(f"User {user.username} updated successfully by {current_user.username}")
        
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "User updated successfully"}
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user {user_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Error updating user"}
        )

@router.post("/{user_id}/change-password")
async def change_password(
    user_id: int,
    new_password: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "User not found"}
            )
        
        # Hash new password
        user.hashed_password = pwd_context.hash(new_password)
        db.commit()
        
        logger.info(f"Password changed for user {user.username} by {current_user.username}")
        
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "Password changed successfully"}
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error changing password for user {user_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Error changing password"}
        )

@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a user"""
    try:
        # Prevent self-deletion
        if user_id == current_user.id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Cannot delete your own account"}
            )
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "User not found"}
            )
        
        username = user.username
        db.delete(user)
        db.commit()
        
        logger.info(f"User {username} deleted successfully by {current_user.username}")
        
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "User deleted successfully"}
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting user {user_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Error deleting user"}
        )

@router.post("/{user_id}/toggle-status")
async def toggle_user_status(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle user active status"""
    try:
        # Prevent self-deactivation
        if user_id == current_user.id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Cannot deactivate your own account"}
            )
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "User not found"}
            )
        
        user.is_active = not user.is_active
        db.commit()
        
        status = "activated" if user.is_active else "deactivated"
        logger.info(f"User {user.username} {status} by {current_user.username}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True, 
                "message": f"User {status} successfully",
                "is_active": user.is_active
            }
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error toggling user status {user_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Error updating user status"}
        )

# API Endpoints
@router.get("/api/stats")
async def get_users_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get users statistics"""
    try:
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        inactive_users = total_users - active_users
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": inactive_users,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting users stats: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching users statistics"
        )

logger.info("Users management router initialized successfully")