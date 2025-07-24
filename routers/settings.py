"""
Settings Management Router
Centralized settings and configuration management for the HR system.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from fastapi import (
    APIRouter, Request, Depends, HTTPException
)
from fastapi import status as http_status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from models.database import get_db
from models.employee import User, Department
from models.custom_fields import CustomField
from utils.auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Settings dashboard page"""
    try:
        # Get stats for the settings page
        departments_count = db.query(Department).count()
        
        # Try to get custom fields count if the model exists
        try:
            custom_fields_count = db.query(CustomField).count()
        except Exception as e:
            logger.warning(f"Could not get custom fields count: {str(e)}")
            custom_fields_count = 0
        
        context = {
            "request": request,
            "user": current_user,
            "page_title": "Settings",
            "departments_count": departments_count,
            "custom_fields_count": custom_fields_count,
        }
        return templates.TemplateResponse("components/settings.html", context)
        
    except Exception as e:
        logger.error(f"Error loading settings page: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading settings"
        )

@router.get("/general", response_class=HTMLResponse)
async def general_settings(
    request: Request, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """General system settings (coming soon)"""
    try:
        context = {
            "request": request,
            "user": current_user,
            "page_title": "General Settings",
            "message": "General settings will be available in a future update.",
            "departments_count": 0,
            "custom_fields_count": 0,
        }
        return templates.TemplateResponse("components/settings.html", context)
    except Exception as e:
        logger.error(f"Error loading general settings: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading general settings"
        )

@router.get("/employee-status", response_class=HTMLResponse)
async def employee_status_settings(
    request: Request, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Employee status configuration (coming soon)"""
    try:
        context = {
            "request": request,
            "user": current_user,
            "page_title": "Employee Status Settings",
            "message": "Employee status configuration will be available in a future update.",
            "departments_count": 0,
            "custom_fields_count": 0,
        }
        return templates.TemplateResponse("components/settings.html", context)
    except Exception as e:
        logger.error(f"Error loading employee status settings: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading employee status settings"
        )

@router.get("/work-schedules", response_class=HTMLResponse)
async def work_schedules_settings(
    request: Request, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Work schedules configuration (coming soon)"""
    try:
        context = {
            "request": request,
            "user": current_user,
            "page_title": "Work Schedules",
            "message": "Work schedules configuration will be available in a future update.",
            "departments_count": 0,
            "custom_fields_count": 0,
        }
        return templates.TemplateResponse("components/settings.html", context)
    except Exception as e:
        logger.error(f"Error loading work schedules settings: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading work schedules settings"
        )

@router.get("/attendance-policies", response_class=HTMLResponse)
async def attendance_policies_settings(
    request: Request, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Attendance policies configuration (coming soon)"""
    try:
        context = {
            "request": request,
            "user": current_user,
            "page_title": "Attendance Policies",
            "message": "Attendance policies configuration will be available in a future update.",
            "departments_count": 0,
            "custom_fields_count": 0,
        }
        return templates.TemplateResponse("components/settings.html", context)
    except Exception as e:
        logger.error(f"Error loading attendance policies settings: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading attendance policies settings"
        )

# NEW: Add only the navigation route to public holidays
@router.get("/public-holidays", response_class=HTMLResponse)
async def public_holidays_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Redirect to leave module's public holidays management"""
    return RedirectResponse(url="/leave/holidays/manage", status_code=302)

@router.get("/backup", response_class=HTMLResponse)
async def backup_settings(
    request: Request, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Backup and export settings (coming soon)"""
    try:
        context = {
            "request": request,
            "user": current_user,
            "page_title": "Backup & Export",
            "message": "Backup and export options will be available in a future update.",
            "departments_count": 0,
            "custom_fields_count": 0,
        }
        return templates.TemplateResponse("components/settings.html", context)
    except Exception as e:
        logger.error(f"Error loading backup settings: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading backup settings"
        )

# API Endpoints
@router.get("/api/stats")
async def get_settings_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get settings statistics"""
    try:
        departments_count = db.query(Department).count()
        active_departments = db.query(Department).filter(Department.is_active == True).count()
        
        # Try to get custom fields stats
        try:
            custom_fields_count = db.query(CustomField).count()
            active_custom_fields = db.query(CustomField).filter(CustomField.is_active == True).count()
        except Exception:
            custom_fields_count = 0
            active_custom_fields = 0
        
        return {
            "departments_count": departments_count,
            "active_departments": active_departments,
            "custom_fields_count": custom_fields_count,
            "active_custom_fields": active_custom_fields,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting settings stats: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching settings statistics"
        )

# Health check
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy", 
            "service": "settings_management",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

logger.info("Settings management router initialized successfully")