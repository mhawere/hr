"""
Performance Management Router
Handles employee performance tracking, achievements, warnings, and badge system.
"""

import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc, asc

from fastapi import (
    APIRouter, Request, Form, Depends, HTTPException, 
    Query, BackgroundTasks
)
from fastapi import status as http_status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator

from models.database import get_db
from models.employee import User, Employee, Department
from models.performance import (
    PerformanceRecord, Badge, EmployeeBadge, 
    PerformanceType, BadgeCategory, BadgeLevel
)
from utils.auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Pydantic Models
class PerformanceRecordCreate(BaseModel):
    employee_id: int
    record_type: PerformanceType
    title: str
    description: str
    points: Optional[int] = 0
    effective_date: Optional[date] = None
    
    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Title must be at least 2 characters long')
        return v.strip()

class BadgeCreate(BaseModel):
    name: str
    description: str
    category: BadgeCategory
    level: BadgeLevel
    criteria: str
    points_required: Optional[int] = 0
    icon: Optional[str] = "fas fa-award"

# Helper Functions
def get_employee_by_id(db: Session, employee_id: int) -> Employee:
    """Get employee by ID with error handling"""
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Employee with ID {employee_id} not found"
            )
        return employee
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error while fetching employee {employee_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error accessing employee data"
        )

def calculate_employee_points(db: Session, employee_id: int) -> int:
    """Calculate total performance points for an employee"""
    try:
        total_points = db.query(func.sum(PerformanceRecord.points)).filter(
            and_(
                PerformanceRecord.employee_id == employee_id,
                PerformanceRecord.is_active == True
            )
        ).scalar()
        return total_points or 0
    except Exception as e:
        logger.error(f"Error calculating points for employee {employee_id}: {str(e)}")
        return 0

def check_badge_eligibility(db: Session, employee_id: int, badge_id: int) -> bool:
    """Check if employee is eligible for a specific badge"""
    try:
        badge = db.query(Badge).filter(Badge.id == badge_id).first()
        if not badge:
            return False
        
        # Check if already earned
        existing = db.query(EmployeeBadge).filter(
            and_(
                EmployeeBadge.employee_id == employee_id,
                EmployeeBadge.badge_id == badge_id
            )
        ).first()
        
        if existing:
            return False
        
        # Check points requirement
        employee_points = calculate_employee_points(db, employee_id)
        return employee_points >= badge.points_required
    except Exception as e:
        logger.error(f"Error checking badge eligibility: {str(e)}")
        return False

def auto_award_badges(db: Session, employee_id: int):
    """Automatically award eligible badges to employee"""
    try:
        badges = db.query(Badge).filter(Badge.is_active == True).all()
        
        for badge in badges:
            if check_badge_eligibility(db, employee_id, badge.id):
                # Award the badge
                employee_badge = EmployeeBadge(
                    employee_id=employee_id,
                    badge_id=badge.id,
                    notes="Automatically awarded based on criteria"
                )
                db.add(employee_badge)
                logger.info(f"Auto-awarded badge '{badge.name}' to employee {employee_id}")
        
        db.commit()
    except Exception as e:
        logger.error(f"Error auto-awarding badges: {str(e)}")
        db.rollback()

def initialize_default_badges(db: Session):
    """Initialize default badge system"""
    try:
        # Check if badges already exist
        badge_count = db.query(Badge).count()
        if badge_count > 0:
            return
            
        default_badges = [
            # Attendance Badges
            Badge(
                name="Perfect Month",
                description="Perfect attendance for one month",
                category=BadgeCategory.ATTENDANCE,
                level=BadgeLevel.BRONZE,
                criteria="100% attendance for one month",
                points_required=50,
                icon="fas fa-calendar-check"
            ),
            Badge(
                name="Punctuality Champion",
                description="No late arrivals for 30 days",
                category=BadgeCategory.ATTENDANCE,
                level=BadgeLevel.SILVER,
                criteria="No late arrivals for 30 consecutive days",
                points_required=75,
                icon="fas fa-clock"
            ),
            Badge(
                name="Attendance Superstar",
                description="Perfect attendance for 3 months",
                category=BadgeCategory.ATTENDANCE,
                level=BadgeLevel.GOLD,
                criteria="100% attendance for 3 consecutive months",
                points_required=150,
                icon="fas fa-star"
            ),
            
            # Tenure Badges
            Badge(
                name="One Year Strong",
                description="Completed one year with the company",
                category=BadgeCategory.TENURE,
                level=BadgeLevel.BRONZE,
                criteria="365 days of employment",
                points_required=100,
                icon="fas fa-trophy"
            ),
            Badge(
                name="Two Year Veteran",
                description="Completed two years with the company",
                category=BadgeCategory.TENURE,
                level=BadgeLevel.SILVER,
                criteria="730 days of employment",
                points_required=200,
                icon="fas fa-medal"
            ),
            Badge(
                name="Five Year Legend",
                description="Completed five years with the company",
                category=BadgeCategory.TENURE,
                level=BadgeLevel.GOLD,
                criteria="1825 days of employment",
                points_required=500,
                icon="fas fa-crown"
            ),
            
            # Performance Badges
            Badge(
                name="Rising Star",
                description="Outstanding performance in first 6 months",
                category=BadgeCategory.PERFORMANCE,
                level=BadgeLevel.BRONZE,
                criteria="Excellent performance in first 6 months",
                points_required=100,
                icon="fas fa-rocket"
            ),
            Badge(
                name="Excellence Award",
                description="Consistently excellent performance",
                category=BadgeCategory.PERFORMANCE,
                level=BadgeLevel.GOLD,
                criteria="Multiple achievements and commendations",
                points_required=300,
                icon="fas fa-award"
            ),
            
            # Training Badges
            Badge(
                name="Learning Enthusiast",
                description="Completed multiple training programs",
                category=BadgeCategory.TRAINING,
                level=BadgeLevel.BRONZE,
                criteria="Complete 3 training programs",
                points_required=75,
                icon="fas fa-graduation-cap"
            ),
            
            # Teamwork Badges
            Badge(
                name="Team Player",
                description="Exceptional collaboration and teamwork",
                category=BadgeCategory.TEAMWORK,
                level=BadgeLevel.SILVER,
                criteria="Recognition for outstanding teamwork",
                points_required=125,
                icon="fas fa-users"
            ),
            
            # Innovation Badges
            Badge(
                name="Innovator",
                description="Contributed innovative ideas or solutions",
                category=BadgeCategory.INNOVATION,
                level=BadgeLevel.GOLD,
                criteria="Implemented innovative solutions",
                points_required=250,
                icon="fas fa-lightbulb"
            )
        ]
        
        for badge in default_badges:
            db.add(badge)
        
        db.commit()
        logger.info("Default badges initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing default badges: {str(e)}")
        db.rollback()

# Routes
@router.get("/", response_class=HTMLResponse)
async def performance_dashboard(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Performance management dashboard"""
    try:
        # Initialize badges if they don't exist
        initialize_default_badges(db)
        
        # Get recent performance records
        recent_records = db.query(PerformanceRecord).join(Employee).order_by(
            desc(PerformanceRecord.created_at)
        ).limit(10).all()
        
        # Get performance statistics
        total_records = db.query(PerformanceRecord).count()
        achievements_count = db.query(PerformanceRecord).filter(
            PerformanceRecord.record_type == PerformanceType.ACHIEVEMENT
        ).count()
        warnings_count = db.query(PerformanceRecord).filter(
            PerformanceRecord.record_type == PerformanceType.WARNING
        ).count()
        
        # Get badge statistics
        total_badges = db.query(Badge).filter(Badge.is_active == True).count()
        awarded_badges = db.query(EmployeeBadge).count()
        
        # Get top performers (by points)
        top_performers = db.query(
            Employee.id,
            Employee.first_name,
            Employee.last_name,
            func.sum(PerformanceRecord.points).label('total_points')
        ).join(PerformanceRecord).group_by(
            Employee.id, Employee.first_name, Employee.last_name
        ).order_by(desc('total_points')).limit(5).all()
        
        context = {
            "request": request,
            "user": current_user,
            "recent_records": recent_records,
            "stats": {
                "total_records": total_records,
                "achievements_count": achievements_count,
                "warnings_count": warnings_count,
                "total_badges": total_badges,
                "awarded_badges": awarded_badges
            },
            "top_performers": top_performers,
            "date": date,  # Add this line
            "page_title": "Performance Management"
        }
        
        return templates.TemplateResponse("staff/performance.html", context)
        
    except Exception as e:
        logger.error(f"Error loading performance dashboard: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading performance dashboard"
        )

@router.get("/employee/{employee_id}", response_class=HTMLResponse)
async def employee_performance(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Individual employee performance page"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Get performance records
        performance_records = db.query(PerformanceRecord).filter(
            PerformanceRecord.employee_id == employee_id
        ).order_by(desc(PerformanceRecord.effective_date)).all()
        
        # Get earned badges
        earned_badges = db.query(EmployeeBadge).join(Badge).filter(
            EmployeeBadge.employee_id == employee_id
        ).order_by(desc(EmployeeBadge.earned_date)).all()
        
        # Get available badges (not yet earned)
        earned_badge_ids = [eb.badge_id for eb in earned_badges]
        available_badges = db.query(Badge).filter(
            and_(
                Badge.is_active == True,
                ~Badge.id.in_(earned_badge_ids) if earned_badge_ids else True
            )
        ).all()
        
        # Calculate points and statistics
        total_points = calculate_employee_points(db, employee_id)
        
        # Performance record statistics
        record_stats = {}
        for record_type in PerformanceType:
            count = sum(1 for r in performance_records if r.record_type == record_type)
            record_stats[record_type.value] = count
        
        # Create record type options for form
        record_types = [{"value": rt.value, "label": rt.value.replace('_', ' ').title()} for rt in PerformanceType]
        
        context = {
            "request": request,
            "user": current_user,
            "employee": employee,
            "performance_records": performance_records,
            "earned_badges": earned_badges,
            "available_badges": available_badges,
            "total_points": total_points,
            "record_stats": record_stats,
            "record_types": record_types,
            "date": date,  # Add this line - import date object for template
            "page_title": f"Performance - {employee.first_name} {employee.last_name}"
        }
        
        return templates.TemplateResponse("staff/employee_performance.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading employee performance: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading employee performance"
        )

@router.post("/record/add")
async def add_performance_record(
    employee_id: int = Form(...),
    record_type: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    points: int = Form(0),
    effective_date: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a new performance record"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Validate record type
        try:
            performance_type = PerformanceType(record_type)
        except ValueError:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Invalid record type"
            )
        
        # Parse effective date
        if effective_date:
            try:
                effective_date = datetime.strptime(effective_date, '%Y-%m-%d').date()
            except ValueError:
                effective_date = date.today()
        else:
            effective_date = date.today()
        
        # Create performance record
        record = PerformanceRecord(
            employee_id=employee_id,
            record_type=performance_type,
            title=title.strip(),
            description=description.strip(),
            points=points,
            effective_date=effective_date,
            created_by=current_user.id
        )
        
        db.add(record)
        db.commit()
        
        # Check for badge eligibility after adding record
        auto_award_badges(db, employee_id)
        
        logger.info(f"Performance record added for employee {employee_id} by user {current_user.username}")
        
        return RedirectResponse(
            url=f"/performance/employee/{employee_id}?success=Performance record added successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding performance record: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error adding performance record"
        )

@router.post("/badge/award")
async def award_badge(
    employee_id: int = Form(...),
    badge_id: int = Form(...),
    notes: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manually award a badge to an employee"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Check if badge exists
        badge = db.query(Badge).filter(Badge.id == badge_id).first()
        if not badge:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="Badge not found"
            )
        
        # Check if already awarded
        existing = db.query(EmployeeBadge).filter(
            and_(
                EmployeeBadge.employee_id == employee_id,
                EmployeeBadge.badge_id == badge_id
            )
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Badge already awarded to this employee"
            )
        
        # Award the badge
        employee_badge = EmployeeBadge(
            employee_id=employee_id,
            badge_id=badge_id,
            awarded_by=current_user.id,
            notes=notes.strip()
        )
        
        db.add(employee_badge)
        db.commit()
        
        logger.info(f"Badge '{badge.name}' awarded to employee {employee_id} by user {current_user.username}")
        
        return RedirectResponse(
            url=f"/performance/employee/{employee_id}?success=Badge awarded successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error awarding badge: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error awarding badge"
        )

@router.get("/badges", response_class=HTMLResponse)
async def manage_badges(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manage badge system"""
    try:
        badges = db.query(Badge).order_by(Badge.category, Badge.level).all()
        
        # Get badge statistics
        badge_stats = {}
        for badge in badges:
            awarded_count = db.query(EmployeeBadge).filter(EmployeeBadge.badge_id == badge.id).count()
            badge_stats[badge.id] = awarded_count
        
        context = {
            "request": request,
            "user": current_user,
            "badges": badges,
            "badge_stats": badge_stats,
            "badge_categories": [{"value": bc.value, "label": bc.value.replace('_', ' ').title()} for bc in BadgeCategory],
            "badge_levels": [{"value": bl.value, "label": bl.value.title()} for bl in BadgeLevel],
            "page_title": "Manage Badges"
        }
        
        return templates.TemplateResponse("staff/manage_badges.html", context)
        
    except Exception as e:
        logger.error(f"Error loading badge management: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading badge management"
        )

@router.post("/badges/add")
async def add_badge(
    name: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    level: str = Form(...),
    criteria: str = Form(...),
    points_required: int = Form(0),
    icon: str = Form("fas fa-award"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a new badge"""
    try:
        # Validate enums
        try:
            badge_category = BadgeCategory(category)
            badge_level = BadgeLevel(level)
        except ValueError:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Invalid category or level"
            )
        
        # Check if badge name already exists
        existing = db.query(Badge).filter(Badge.name == name.strip()).first()
        if existing:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Badge name already exists"
            )
        
        badge = Badge(
            name=name.strip(),
            description=description.strip(),
            category=badge_category,
            level=badge_level,
            criteria=criteria.strip(),
            points_required=points_required,
            icon=icon.strip()
        )
        
        db.add(badge)
        db.commit()
        
        logger.info(f"Badge '{name}' created by user {current_user.username}")
        
        return RedirectResponse(
            url="/performance/badges?success=Badge created successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating badge: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating badge"
        )

@router.delete("/record/{record_id}")
async def delete_performance_record(
    record_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a performance record"""
    try:
        record = db.query(PerformanceRecord).filter(PerformanceRecord.id == record_id).first()
        if not record:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="Performance record not found"
            )
        
        employee_id = record.employee_id
        db.delete(record)
        db.commit()
        
        logger.info(f"Performance record {record_id} deleted by user {current_user.username}")
        
        return {"success": True, "message": "Performance record deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting performance record: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting performance record"
        )

@router.get("/api/employee/{employee_id}/stats")
async def get_employee_performance_stats(
    employee_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get performance statistics for an employee"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        total_points = calculate_employee_points(db, employee_id)
        badges_earned = db.query(EmployeeBadge).filter(EmployeeBadge.employee_id == employee_id).count()
        
        # Performance trends (last 6 months)
        six_months_ago = date.today() - timedelta(days=180)
        monthly_stats = db.query(
            func.extract('month', PerformanceRecord.effective_date).label('month'),
            func.extract('year', PerformanceRecord.effective_date).label('year'),
            func.count(PerformanceRecord.id).label('count')
        ).filter(
            and_(
                PerformanceRecord.employee_id == employee_id,
                PerformanceRecord.effective_date >= six_months_ago
            )
        ).group_by('month', 'year').all()
        
        return {
            "employee_id": employee_id,
            "total_points": total_points,
            "badges_earned": badges_earned,
            "monthly_stats": [
                {
                    "month": stat.month,
                    "year": stat.year,
                    "count": stat.count
                }
                for stat in monthly_stats
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting employee stats: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching employee statistics"
        )

# Health check
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy", 
            "service": "performance_management",
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

logger.info("Performance management router initialized successfully")