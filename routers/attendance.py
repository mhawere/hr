"""
Attendance Management Router
Comprehensive attendance system with shift management, time tracking, 
and employee scheduling capabilities.
"""

import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, time, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from models.shift import Shift
from fastapi import (
    APIRouter, Request, Form, Depends, HTTPException, Query, BackgroundTasks
)
from fastapi import status as http_status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator, model_validator
from sqlalchemy import Column, Integer, String, Text, Boolean, Time, DateTime, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from models.database import get_db, Base
from models.employee import User, Employee
from utils.auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Shift Models
# class Shift(Base):
#     __tablename__ = "shifts"
    
#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(100), nullable=False)
#     description = Column(Text, nullable=True)
#     shift_type = Column(String(20), nullable=False)  # 'standard' or 'dynamic'
    
    # Standard shift fields
    # weekday_start = Column(Time, nullable=True)
    # weekday_end = Column(Time, nullable=True)
    
    # Individual weekend day fields (used by both standard and dynamic shifts)
    # saturday_start = Column(Time, nullable=True)
    # saturday_end = Column(Time, nullable=True)
    # sunday_start = Column(Time, nullable=True)
    # sunday_end = Column(Time, nullable=True)
    
    # # Dynamic shift fields (weekday individual days)
    # monday_start = Column(Time, nullable=True)
    # monday_end = Column(Time, nullable=True)
    # tuesday_start = Column(Time, nullable=True)
    # tuesday_end = Column(Time, nullable=True)
    # wednesday_start = Column(Time, nullable=True)
    # wednesday_end = Column(Time, nullable=True)
    # thursday_start = Column(Time, nullable=True)
    # thursday_end = Column(Time, nullable=True)
    # friday_start = Column(Time, nullable=True)
    # friday_end = Column(Time, nullable=True)
    
    # is_active = Column(Boolean, default=True, nullable=False)
    # assigned_employees_count = Column(Integer, default=0, nullable=False)
    
    # created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    # employees = relationship("Employee", foreign_keys="Employee.shift_id", back_populates="shift")

# Pydantic models for validation
class StandardShiftCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    weekday_start: time
    weekday_end: time
    saturday_start: Optional[time] = None
    saturday_end: Optional[time] = None
    sunday_start: Optional[time] = None
    sunday_end: Optional[time] = None
    is_active: bool = True
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Shift name must be at least 2 characters long')
        if len(v.strip()) > 100:
            raise ValueError('Shift name must be less than 100 characters')
        return v.strip()
    
    @model_validator(mode='after')
    def validate_times(self):
        # Validate weekday times
        if self.weekday_end <= self.weekday_start:
            raise ValueError('Weekday end time must be after start time')
        
        # Validate Saturday times
        if self.saturday_start and self.saturday_end:
            if self.saturday_end <= self.saturday_start:
                raise ValueError('Saturday end time must be after start time')
        elif self.saturday_start and not self.saturday_end:
            raise ValueError('Saturday end time must be provided when start time is specified')
        elif not self.saturday_start and self.saturday_end:
            raise ValueError('Saturday start time must be provided when end time is specified')
        
        # Validate Sunday times
        if self.sunday_start and self.sunday_end:
            if self.sunday_end <= self.sunday_start:
                raise ValueError('Sunday end time must be after start time')
        elif self.sunday_start and not self.sunday_end:
            raise ValueError('Sunday end time must be provided when start time is specified')
        elif not self.sunday_start and self.sunday_end:
            raise ValueError('Sunday start time must be provided when end time is specified')
            
        return self

class DynamicShiftCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    monday_start: Optional[time] = None
    monday_end: Optional[time] = None
    tuesday_start: Optional[time] = None
    tuesday_end: Optional[time] = None
    wednesday_start: Optional[time] = None
    wednesday_end: Optional[time] = None
    thursday_start: Optional[time] = None
    thursday_end: Optional[time] = None
    friday_start: Optional[time] = None
    friday_end: Optional[time] = None
    saturday_start: Optional[time] = None
    saturday_end: Optional[time] = None
    sunday_start: Optional[time] = None
    sunday_end: Optional[time] = None
    is_active: bool = True
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Shift name must be at least 2 characters long')
        if len(v.strip()) > 100:
            raise ValueError('Shift name must be less than 100 characters')
        return v.strip()
    
    @model_validator(mode='after')
    def validate_day_times_and_at_least_one_day(self):
        """Validate individual day times and ensure at least one day is configured"""
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        has_configured_day = False
        
        for day in days:
            start_time = getattr(self, f'{day}_start')
            end_time = getattr(self, f'{day}_end')
            
            # If both start and end are provided, validate them
            if start_time and end_time:
                if end_time <= start_time:
                    raise ValueError(f'{day.capitalize()} end time must be after start time')
                has_configured_day = True
            # If only one is provided, that's an error
            elif start_time and not end_time:
                raise ValueError(f'{day.capitalize()} end time must be provided when start time is specified')
            elif not start_time and end_time:
                raise ValueError(f'{day.capitalize()} start time must be provided when end time is specified')
        
        # Ensure at least one day is configured
        if not has_configured_day:
            raise ValueError('At least one day must have both start and end times configured')
            
        return self
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Shift name must be at least 2 characters long')
        if len(v.strip()) > 100:
            raise ValueError('Shift name must be less than 100 characters')
        return v.strip()
    
    @model_validator(mode='after')
    def validate_day_times_and_at_least_one_day(self):
        """Validate individual day times and ensure at least one day is configured"""
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        has_configured_day = False
        
        for day in days:
            start_time = getattr(self, f'{day}_start')
            end_time = getattr(self, f'{day}_end')
            
            # If both start and end are provided, validate them
            if start_time and end_time:
                if end_time <= start_time:
                    raise ValueError(f'{day.capitalize()} end time must be after start time')
                has_configured_day = True
            # If only one is provided, that's an error
            elif start_time and not end_time:
                raise ValueError(f'{day.capitalize()} end time must be provided when start time is specified')
            elif not start_time and end_time:
                raise ValueError(f'{day.capitalize()} start time must be provided when end time is specified')
        
        # Ensure at least one day is configured
        if not has_configured_day:
            raise ValueError('At least one day must have both start and end times configured')
            
        return self

# =============================================================================
# PYDANTIC MODELS AND RESPONSE SCHEMAS
# =============================================================================

class ShiftResponse(BaseModel):
    """
    Comprehensive shift response model with computed fields for frontend consumption
    """
    # Core fields
    id: int
    name: str
    description: Optional[str] = None
    shift_type: str
    is_active: bool
    assigned_employees_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    # Standard shift weekday fields
    weekday_start: Optional[time] = None
    weekday_end: Optional[time] = None
    
    # Weekend day fields (used by both standard and dynamic shifts)
    saturday_start: Optional[time] = None
    saturday_end: Optional[time] = None
    sunday_start: Optional[time] = None
    sunday_end: Optional[time] = None
    
    # Dynamic shift individual weekday fields
    monday_start: Optional[time] = None
    monday_end: Optional[time] = None
    tuesday_start: Optional[time] = None
    tuesday_end: Optional[time] = None
    wednesday_start: Optional[time] = None
    wednesday_end: Optional[time] = None
    thursday_start: Optional[time] = None
    thursday_end: Optional[time] = None
    friday_start: Optional[time] = None
    friday_end: Optional[time] = None
    
    # Computed fields for enhanced frontend experience
    weekend_policy: Optional[str] = None
    total_weekly_hours: Optional[float] = None
    working_days_count: Optional[int] = None
    working_days: Optional[List[str]] = None
    is_overnight_shift: Optional[bool] = None
    shift_duration_display: Optional[str] = None
    
    class Config:
        from_attributes = True
        json_encoders = {
            time: lambda v: v.strftime('%H:%M') if v else None,
            datetime: lambda v: v.isoformat() if v else None
        }
    
    @classmethod
    def from_shift(cls, shift: Shift) -> 'ShiftResponse':
        """
        Create ShiftResponse with all computed fields populated
        """
        try:
            # Extract base fields
            base_data = {
                'id': shift.id,
                'name': shift.name,
                'description': shift.description,
                'shift_type': shift.shift_type,
                'is_active': shift.is_active,
                'assigned_employees_count': shift.assigned_employees_count or 0,
                'created_at': shift.created_at,
                'updated_at': shift.updated_at,
                
                # Time fields
                'weekday_start': shift.weekday_start,
                'weekday_end': shift.weekday_end,
                'saturday_start': shift.saturday_start,
                'saturday_end': shift.saturday_end,
                'sunday_start': shift.sunday_start,
                'sunday_end': shift.sunday_end,
                'monday_start': shift.monday_start,
                'monday_end': shift.monday_end,
                'tuesday_start': shift.tuesday_start,
                'tuesday_end': shift.tuesday_end,
                'wednesday_start': shift.wednesday_start,
                'wednesday_end': shift.wednesday_end,
                'thursday_start': shift.thursday_start,
                'thursday_end': shift.thursday_end,
                'friday_start': shift.friday_start,
                'friday_end': shift.friday_end,
            }
            
            # Calculate computed fields
            computed_fields = calculate_shift_metrics(shift)
            base_data.update(computed_fields)
            
            return cls(**base_data)
            
        except Exception as e:
            logger.error(f"Error creating ShiftResponse from shift {shift.id}: {str(e)}")
            # Return basic response without computed fields on error
            return cls(
                id=shift.id,
                name=shift.name or f"Shift {shift.id}",
                description=shift.description,
                shift_type=shift.shift_type,
                is_active=shift.is_active,
                assigned_employees_count=shift.assigned_employees_count or 0,
                created_at=shift.created_at,
                updated_at=shift.updated_at
            )

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_shift_metrics(shift: Shift) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a shift including weekend policy,
    total hours, working days, and other computed fields
    """
    try:
        metrics = {
            'weekend_policy': determine_weekend_policy(shift),
            'total_weekly_hours': 0.0,
            'working_days_count': 0,
            'working_days': [],
            'is_overnight_shift': False,
            'shift_duration_display': ''
        }
        
        if shift.shift_type == 'standard':
            metrics.update(calculate_standard_shift_metrics(shift))
        elif shift.shift_type == 'dynamic':
            metrics.update(calculate_dynamic_shift_metrics(shift))
        
        # Format duration display
        if metrics['total_weekly_hours'] > 0:
            hours = int(metrics['total_weekly_hours'])
            minutes = int((metrics['total_weekly_hours'] - hours) * 60)
            if minutes > 0:
                metrics['shift_duration_display'] = f"{hours}h {minutes}m per week"
            else:
                metrics['shift_duration_display'] = f"{hours}h per week"
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating shift metrics: {str(e)}")
        return {
            'weekend_policy': 'none',
            'total_weekly_hours': 0.0,
            'working_days_count': 0,
            'working_days': [],
            'is_overnight_shift': False,
            'shift_duration_display': ''
        }

def determine_weekend_policy(shift: Shift) -> str:
    """Determine weekend policy based on configured times"""
    has_saturday = shift.saturday_start and shift.saturday_end
    has_sunday = shift.sunday_start and shift.sunday_end
    
    if has_saturday and has_sunday:
        return 'both'
    elif has_saturday and not has_sunday:
        return 'saturday_only'
    elif not has_saturday and has_sunday:
        return 'sunday_only'
    else:
        return 'none'

def calculate_standard_shift_metrics(shift: Shift) -> Dict[str, Any]:
    """Calculate metrics for standard shifts"""
    total_hours = 0.0
    working_days = []
    is_overnight = False
    
    # Weekday hours (Monday-Friday)
    if shift.weekday_start and shift.weekday_end:
        weekday_hours = calculate_time_difference(shift.weekday_start, shift.weekday_end)
        total_hours += weekday_hours * 5  # 5 weekdays
        working_days.extend(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
        
        # Check if it's an overnight shift
        if shift.weekday_end < shift.weekday_start:
            is_overnight = True
    
    # Saturday hours
    if shift.saturday_start and shift.saturday_end:
        saturday_hours = calculate_time_difference(shift.saturday_start, shift.saturday_end)
        total_hours += saturday_hours
        working_days.append('Saturday')
        
        if shift.saturday_end < shift.saturday_start:
            is_overnight = True
    
    # Sunday hours
    if shift.sunday_start and shift.sunday_end:
        sunday_hours = calculate_time_difference(shift.sunday_start, shift.sunday_end)
        total_hours += sunday_hours
        working_days.append('Sunday')
        
        if shift.sunday_end < shift.sunday_start:
            is_overnight = True
    
    return {
        'total_weekly_hours': round(total_hours, 2),
        'working_days_count': len(working_days),
        'working_days': working_days,
        'is_overnight_shift': is_overnight
    }

def calculate_dynamic_shift_metrics(shift: Shift) -> Dict[str, Any]:
    """Calculate metrics for dynamic shifts"""
    total_hours = 0.0
    working_days = []
    is_overnight = False
    
    # Define all days with their corresponding fields
    days_config = [
        ('Monday', shift.monday_start, shift.monday_end),
        ('Tuesday', shift.tuesday_start, shift.tuesday_end),
        ('Wednesday', shift.wednesday_start, shift.wednesday_end),
        ('Thursday', shift.thursday_start, shift.thursday_end),
        ('Friday', shift.friday_start, shift.friday_end),
        ('Saturday', shift.saturday_start, shift.saturday_end),
        ('Sunday', shift.sunday_start, shift.sunday_end)
    ]
    
    for day_name, start_time, end_time in days_config:
        if start_time and end_time:
            day_hours = calculate_time_difference(start_time, end_time)
            total_hours += day_hours
            working_days.append(day_name)
            
            # Check for overnight shifts
            if end_time < start_time:
                is_overnight = True
    
    return {
        'total_weekly_hours': round(total_hours, 2),
        'working_days_count': len(working_days),
        'working_days': working_days,
        'is_overnight_shift': is_overnight
    }

def calculate_time_difference(start_time: time, end_time: time) -> float:
    """
    Calculate the difference between two time objects in hours
    Handles overnight shifts correctly
    """
    if not start_time or not end_time:
        return 0.0
    
    try:
        # Convert to datetime for calculation
        base_date = datetime.today()
        start_dt = datetime.combine(base_date, start_time)
        end_dt = datetime.combine(base_date, end_time)
        
        # Handle overnight shifts (end time is before start time)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        
        # Calculate hours
        time_diff = end_dt - start_dt
        hours = time_diff.total_seconds() / 3600
        
        return round(hours, 2)
        
    except Exception as e:
        logger.error(f"Error calculating time difference: {str(e)}")
        return 0.0

# =============================================================================
# DATABASE HELPER FUNCTIONS
# =============================================================================

def get_shift_by_id(db: Session, shift_id: int) -> Shift:
    """
    Get shift by ID with comprehensive error handling and logging
    """
    try:
        if not isinstance(shift_id, int) or shift_id <= 0:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Invalid shift ID provided"
            )
        
        shift = db.query(Shift).filter(Shift.id == shift_id).first()
        
        if not shift:
            logger.warning(f"Shift with ID {shift_id} not found")
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Shift with ID {shift_id} not found"
            )
        
        logger.debug(f"Successfully retrieved shift {shift_id}: {shift.name}")
        return shift
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching shift {shift_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while accessing shift data"
        )
    except Exception as e:
        logger.error(f"Unexpected error while fetching shift {shift_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error accessing shift data"
        )

def check_shift_name_exists(db: Session, name: str, exclude_id: Optional[int] = None) -> bool:
    """
    Check if shift name already exists with improved validation and error handling
    """
    try:
        if not name or not name.strip():
            return False
        
        # Normalize the name for comparison
        normalized_name = name.strip().lower()
        
        query = db.query(Shift).filter(func.lower(Shift.name) == normalized_name)
        
        if exclude_id and isinstance(exclude_id, int) and exclude_id > 0:
            query = query.filter(Shift.id != exclude_id)
        
        exists = query.first() is not None
        
        if exists:
            logger.info(f"Shift name '{name}' already exists (excluding ID: {exclude_id})")
        
        return exists
        
    except SQLAlchemyError as e:
        logger.error(f"Database error while checking shift name existence: {str(e)}")
        # Return False on database error to allow operation to continue
        return False
    except Exception as e:
        logger.error(f"Unexpected error while checking shift name existence: {str(e)}")
        return False

def get_shifts_with_filters(
    db: Session, 
    shift_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    name_search: Optional[str] = None,
    offset: int = 0,
    limit: int = 100
) -> List[Shift]:
    """
    Get shifts with optional filters and pagination
    """
    try:
        query = db.query(Shift)
        
        # Apply filters
        if shift_type and shift_type in ['standard', 'dynamic']:
            query = query.filter(Shift.shift_type == shift_type)
        
        if is_active is not None:
            query = query.filter(Shift.is_active == is_active)
        
        if name_search and name_search.strip():
            search_term = f"%{name_search.strip()}%"
            query = query.filter(
                or_(
                    Shift.name.ilike(search_term),
                    Shift.description.ilike(search_term)
                )
            )
        
        # Apply pagination and ordering
        shifts = query.order_by(
            Shift.is_active.desc(),  # Active shifts first
            Shift.created_at.desc()  # Then by creation date
        ).offset(offset).limit(limit).all()
        
        return shifts
        
    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching shifts: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching shifts"
        )
    except Exception as e:
        logger.error(f"Unexpected error while fetching shifts: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching shifts"
        )

def parse_time_string(time_str: str) -> Optional[time]:
    """
    Parse time string to time object with comprehensive format support and validation
    """
    if not time_str or not time_str.strip():
        return None
    
    # Clean and normalize the input
    time_str = time_str.strip().upper()
    
    # Define supported time patterns with their descriptions
    time_patterns = [
        ('%H:%M:%S', 'HH:MM:SS (24-hour with seconds)'),
        ('%H:%M', 'HH:MM (24-hour)'),
        ('%I:%M:%S %p', 'HH:MM:SS AM/PM (12-hour with seconds)'),
        ('%I:%M %p', 'HH:MM AM/PM (12-hour)'),
        ('%H%M', 'HHMM (24-hour no colon)'),
        ('%I%M %p', 'HHMM AM/PM (12-hour no colon)')
    ]
    
    for pattern, description in time_patterns:
        try:
            parsed_time = datetime.strptime(time_str, pattern).time()
            # Ensure consistent HH:MM format (strip seconds and microseconds)
            normalized_time = time(parsed_time.hour, parsed_time.minute)
            
            logger.debug(f"Successfully parsed time '{time_str}' as {normalized_time} using pattern {description}")
            return normalized_time
            
        except ValueError:
            continue
    
    # If no pattern matched, provide helpful error message
    supported_formats = [desc.split('(')[1].rstrip(')') for _, desc in time_patterns]
    error_msg = (
        f"Invalid time format: '{time_str}'. "
        f"Supported formats: {', '.join(supported_formats)}"
    )
    
    logger.warning(f"Failed to parse time string: {error_msg}")
    raise ValueError(error_msg)

def validate_shift_times(shift_data: Dict[str, Any]) -> List[str]:
    """
    Validate shift times and return list of validation errors
    """
    errors = []
    
    try:
        if shift_data.get('shift_type') == 'standard':
            errors.extend(validate_standard_shift_times(shift_data))
        elif shift_data.get('shift_type') == 'dynamic':
            errors.extend(validate_dynamic_shift_times(shift_data))
        
        return errors
        
    except Exception as e:
        logger.error(f"Error during shift time validation: {str(e)}")
        return [f"Validation error: {str(e)}"]

def validate_standard_shift_times(shift_data: Dict[str, Any]) -> List[str]:
    """Validate standard shift time configuration"""
    errors = []
    
    # Validate weekday times (required for standard shifts)
    weekday_start = shift_data.get('weekday_start')
    weekday_end = shift_data.get('weekday_end')
    
    if not weekday_start or not weekday_end:
        errors.append("Weekday start and end times are required for standard shifts")
    elif weekday_start >= weekday_end:
        errors.append("Weekday end time must be after start time")
    
    # Validate weekend times based on policy
    weekend_policy = shift_data.get('weekend_policy', 'none')
    
    if weekend_policy in ['both', 'saturday_only']:
        sat_start = shift_data.get('saturday_start')
        sat_end = shift_data.get('saturday_end')
        if not sat_start or not sat_end:
            errors.append("Saturday times are required for this weekend policy")
        elif sat_start >= sat_end:
            errors.append("Saturday end time must be after start time")
    
    if weekend_policy in ['both', 'sunday_only']:
        sun_start = shift_data.get('sunday_start')
        sun_end = shift_data.get('sunday_end')
        if not sun_start or not sun_end:
            errors.append("Sunday times are required for this weekend policy")
        elif sun_start >= sun_end:
            errors.append("Sunday end time must be after start time")
    
    return errors

def validate_dynamic_shift_times(shift_data: Dict[str, Any]) -> List[str]:
    """Validate dynamic shift time configuration"""
    errors = []
    
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    working_days = 0
    
    for day in days:
        start_key = f'{day}_start'
        end_key = f'{day}_end'
        start_time = shift_data.get(start_key)
        end_time = shift_data.get(end_key)
        
        if start_time and end_time:
            if start_time >= end_time:
                errors.append(f"{day.capitalize()} end time must be after start time")
            else:
                working_days += 1
        elif start_time or end_time:
            errors.append(f"{day.capitalize()} requires both start and end times")
    
    if working_days == 0:
        errors.append("At least one working day must be configured for dynamic shifts")
    
    return errors

# Routes
@router.get("/", response_class=HTMLResponse)
async def attendance_home(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Attendance management home page"""
    return RedirectResponse(url="/attendance/shift-manager")

@router.get("/shift-manager", response_class=HTMLResponse)
async def shift_manager_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Display shift manager page"""
    try:
        shifts = db.query(Shift).order_by(Shift.created_at.desc()).all()
        
        context = {
            "request": request,
            "user": current_user,
            "shifts": shifts,
            "page_title": "Shift Manager"
        }
        return templates.TemplateResponse("staff/attendance/shift_manager.html", context)
        
    except Exception as e:
        logger.error(f"Error loading shift manager page: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading shift manager"
        )

@router.post("/shift-manager/standard")
async def create_standard_shift(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    weekday_start: str = Form(...),
    weekday_end: str = Form(...),
    weekend_policy: str = Form("none"),
    saturday_start: str = Form(""),
    saturday_end: str = Form(""),
    sunday_start: str = Form(""),
    sunday_end: str = Form(""),
    is_active: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new standard shift with proper weekend handling"""
    try:
        # Parse weekday times (required)
        weekday_start_time = parse_time_string(weekday_start)
        weekday_end_time = parse_time_string(weekday_end)
        
        if not weekday_start_time or not weekday_end_time:
            raise ValueError("Weekday start and end times are required")
        
        # Initialize weekend times
        saturday_start_time = None
        saturday_end_time = None
        sunday_start_time = None
        sunday_end_time = None
        
        # Handle weekend policy logic
        if weekend_policy == "both":
            # Both days work - Saturday times are required
            if not saturday_start or not saturday_end:
                raise ValueError("Weekend times are required when both days are working")
            saturday_start_time = parse_time_string(saturday_start)
            saturday_end_time = parse_time_string(saturday_end)
            # Copy Saturday times to Sunday
            sunday_start_time = saturday_start_time
            sunday_end_time = saturday_end_time
            
        elif weekend_policy == "saturday_only":
            # Only Saturday works
            if not saturday_start or not saturday_end:
                raise ValueError("Saturday times are required when Saturday is a working day")
            saturday_start_time = parse_time_string(saturday_start)
            saturday_end_time = parse_time_string(saturday_end)
            # Sunday remains None
            
        elif weekend_policy == "sunday_only":
            # Only Sunday works
            if not sunday_start or not sunday_end:
                raise ValueError("Sunday times are required when Sunday is a working day")
            sunday_start_time = parse_time_string(sunday_start)
            sunday_end_time = parse_time_string(sunday_end)
            # Saturday remains None
            
        # weekend_policy == "none" - both remain None
        
        # Validate times
        if weekday_end_time <= weekday_start_time:
            raise ValueError("Weekday end time must be after start time")
        
        if saturday_start_time and saturday_end_time and saturday_end_time <= saturday_start_time:
            raise ValueError("Saturday end time must be after start time")
            
        if sunday_start_time and sunday_end_time and sunday_end_time <= sunday_start_time:
            raise ValueError("Sunday end time must be after start time")
        
        # Create shift data
        shift_data = StandardShiftCreate(
            name=name,
            description=description,
            weekday_start=weekday_start_time,
            weekday_end=weekday_end_time,
            saturday_start=saturday_start_time,
            saturday_end=saturday_end_time,
            sunday_start=sunday_start_time,
            sunday_end=sunday_end_time,
            is_active=is_active
        )
        
        # Check if shift name already exists
        if check_shift_name_exists(db, shift_data.name):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="A shift with this name already exists"
            )
        
        # Create new shift
        new_shift = Shift(
            name=shift_data.name,
            description=shift_data.description,
            shift_type='standard',
            weekday_start=shift_data.weekday_start,
            weekday_end=shift_data.weekday_end,
            saturday_start=shift_data.saturday_start,
            saturday_end=shift_data.saturday_end,
            sunday_start=shift_data.sunday_start,
            sunday_end=shift_data.sunday_end,
            is_active=shift_data.is_active
        )
        
        db.add(new_shift)
        db.commit()
        db.refresh(new_shift)
        
        logger.info(f"Standard shift '{new_shift.name}' created by user {current_user.username}")
        
        return RedirectResponse(
            url=f"/attendance/shift-manager?success=Standard shift '{new_shift.name}' created successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except ValueError as e:
        logger.error(f"Validation error creating standard shift: {str(e)}")
        shifts = db.query(Shift).order_by(Shift.created_at.desc()).all()
        context = {
            "request": request,
            "user": current_user,
            "shifts": shifts,
            "error": str(e),
            "page_title": "Shift Manager"
        }
        return templates.TemplateResponse("staff/attendance/shift_manager.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating standard shift: {str(e)}")
        
        shifts = db.query(Shift).order_by(Shift.created_at.desc()).all()
        context = {
            "request": request,
            "user": current_user,
            "shifts": shifts,
            "error": "An unexpected error occurred. Please try again.",
            "page_title": "Shift Manager"
        }
        return templates.TemplateResponse("staff/attendance/shift_manager.html", context)

@router.post("/shift-manager/standard/{shift_id}")
async def update_standard_shift(
    shift_id: int,
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    weekday_start: str = Form(...),
    weekday_end: str = Form(...),
    weekend_policy: str = Form("none"),          # ✅ Add this
    saturday_start: str = Form(""),              # ✅ Fix these field names
    saturday_end: str = Form(""),                # ✅ Fix these field names
    sunday_start: str = Form(""),                # ✅ Add this
    sunday_end: str = Form(""),                  # ✅ Add this
    is_active: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an existing standard shift with proper weekend handling"""
    try:
        # Get existing shift
        shift = get_shift_by_id(db, shift_id)
        
        # Parse weekday times (required)
        weekday_start_time = parse_time_string(weekday_start)
        weekday_end_time = parse_time_string(weekday_end)
        
        if not weekday_start_time or not weekday_end_time:
            raise ValueError("Weekday start and end times are required")
        
        # Initialize weekend times
        saturday_start_time = None
        saturday_end_time = None
        sunday_start_time = None
        sunday_end_time = None
        
        # Handle weekend policy logic (SAME AS CREATE ROUTE)
        if weekend_policy == "both":
            # Both days work - Saturday times are required
            if not saturday_start or not saturday_end:
                raise ValueError("Weekend times are required when both days are working")
            saturday_start_time = parse_time_string(saturday_start)
            saturday_end_time = parse_time_string(saturday_end)
            # Copy Saturday times to Sunday
            sunday_start_time = saturday_start_time
            sunday_end_time = saturday_end_time
            
        elif weekend_policy == "saturday_only":
            # Only Saturday works
            if not saturday_start or not saturday_end:
                raise ValueError("Saturday times are required when Saturday is a working day")
            saturday_start_time = parse_time_string(saturday_start)
            saturday_end_time = parse_time_string(saturday_end)
            # Sunday remains None
            
        elif weekend_policy == "sunday_only":
            # Only Sunday works
            if not sunday_start or not sunday_end:
                raise ValueError("Sunday times are required when Sunday is a working day")
            sunday_start_time = parse_time_string(sunday_start)
            sunday_end_time = parse_time_string(sunday_end)
            # Saturday remains None
            
        # weekend_policy == "none" - both remain None
        
        # Validate times
        if weekday_end_time <= weekday_start_time:
            raise ValueError("Weekday end time must be after start time")
        
        if saturday_start_time and saturday_end_time and saturday_end_time <= saturday_start_time:
            raise ValueError("Saturday end time must be after start time")
            
        if sunday_start_time and sunday_end_time and sunday_end_time <= sunday_start_time:
            raise ValueError("Sunday end time must be after start time")
        
        # Create shift data for validation
        shift_data = StandardShiftCreate(
            name=name,
            description=description,
            weekday_start=weekday_start_time,
            weekday_end=weekday_end_time,
            saturday_start=saturday_start_time,
            saturday_end=saturday_end_time,
            sunday_start=sunday_start_time,
            sunday_end=sunday_end_time,
            is_active=is_active
        )
        
        # Check if shift name already exists (excluding current shift)
        if check_shift_name_exists(db, shift_data.name, exclude_id=shift_id):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="A shift with this name already exists"
            )
        
        # Update shift
        shift.name = shift_data.name
        shift.description = shift_data.description
        shift.weekday_start = shift_data.weekday_start
        shift.weekday_end = shift_data.weekday_end
        shift.saturday_start = shift_data.saturday_start
        shift.saturday_end = shift_data.saturday_end
        shift.sunday_start = shift_data.sunday_start
        shift.sunday_end = shift_data.sunday_end
        shift.is_active = shift_data.is_active
        shift.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(shift)
        
        logger.info(f"Standard shift '{shift.name}' updated by user {current_user.username}")
        
        return RedirectResponse(
            url=f"/attendance/shift-manager?success=Standard shift '{shift.name}' updated successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except ValueError as e:
        shifts = db.query(Shift).order_by(Shift.created_at.desc()).all()
        context = {
            "request": request,
            "user": current_user,
            "shifts": shifts,
            "error": str(e),
            "page_title": "Shift Manager"
        }
        return templates.TemplateResponse("staff/attendance/shift_manager.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating standard shift: {str(e)}")
        
        shifts = db.query(Shift).order_by(Shift.created_at.desc()).all()
        context = {
            "request": request,
            "user": current_user,
            "shifts": shifts,
            "error": "An unexpected error occurred. Please try again.",
            "page_title": "Shift Manager"
        }
        return templates.TemplateResponse("staff/attendance/shift_manager.html", context)

@router.post("/shift-manager/dynamic/{shift_id}")
async def update_dynamic_shift(
    shift_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an existing dynamic shift"""
    try:
        # Get existing shift
        shift = get_shift_by_id(db, shift_id)
        
        form_data = await request.form()
        
        # Extract basic information
        name = form_data.get('name', '').strip()
        description = form_data.get('description', '').strip()
        is_active = form_data.get('is_active') == 'true'
        
        if not name:
            raise ValueError("Shift name is required")
        
        # Extract and parse day times
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_times = {}
        
        for day in days:
            start_key = f'{day}_start'
            end_key = f'{day}_end'
            
            start_time_str = form_data.get(start_key, '').strip()
            end_time_str = form_data.get(end_key, '').strip()
            
            # Only process if both start and end are provided
            if start_time_str and end_time_str:
                day_times[start_key] = parse_time_string(start_time_str)
                day_times[end_key] = parse_time_string(end_time_str)
                
                # Validate that end time is after start time
                if day_times[end_key] <= day_times[start_key]:
                    raise ValueError(f"{day.capitalize()} end time must be after start time")
            else:
                day_times[start_key] = None
                day_times[end_key] = None
        
        # Check if at least one day is configured
        has_configured_day = any(
            day_times[f'{day}_start'] and day_times[f'{day}_end'] 
            for day in days
        )
        
        if not has_configured_day:
            raise ValueError("At least one day must be configured with valid times")
        
        # Check if shift name already exists (excluding current shift)
        if check_shift_name_exists(db, name, exclude_id=shift_id):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="A shift with this name already exists"
            )
        
        # Update shift
        shift.name = name
        shift.description = description
        shift.is_active = is_active
        shift.updated_at = datetime.utcnow()
        
        # Update day times
        for key, value in day_times.items():
            setattr(shift, key, value)
        
        db.commit()
        db.refresh(shift)
        
        logger.info(f"Dynamic shift '{shift.name}' updated by user {current_user.username}")
        
        return RedirectResponse(
            url=f"/attendance/shift-manager?success=Dynamic shift '{shift.name}' updated successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except ValueError as e:
        shifts = db.query(Shift).order_by(Shift.created_at.desc()).all()
        context = {
            "request": request,
            "user": current_user,
            "shifts": shifts,
            "error": str(e),
            "page_title": "Shift Manager"
        }
        return templates.TemplateResponse("staff/attendance/shift_manager.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating dynamic shift: {str(e)}")
        
        shifts = db.query(Shift).order_by(Shift.created_at.desc()).all()
        context = {
            "request": request,
            "user": current_user,
            "shifts": shifts,
            "error": "An unexpected error occurred. Please try again.",
            "page_title": "Shift Manager"
        }
        return templates.TemplateResponse("staff/attendance/shift_manager.html", context)


@router.post("/shift-manager/dynamic")
async def create_dynamic_shift(
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new dynamic shift"""
    try:
        form_data = await request.form()
        
        # Extract basic information
        name = form_data.get('name', '').strip()
        description = form_data.get('description', '').strip()
        is_active = form_data.get('is_active') == 'true'
        
        if not name:
            raise ValueError("Shift name is required")
        
        # Extract and parse day times
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_times = {}
        
        for day in days:
            start_key = f'{day}_start'
            end_key = f'{day}_end'
            
            start_time_str = form_data.get(start_key, '').strip()
            end_time_str = form_data.get(end_key, '').strip()
            
            # Only process if both start and end are provided
            if start_time_str and end_time_str:
                day_times[start_key] = parse_time_string(start_time_str)
                day_times[end_key] = parse_time_string(end_time_str)
                
                # Validate that end time is after start time
                if day_times[end_key] <= day_times[start_key]:
                    raise ValueError(f"{day.capitalize()} end time must be after start time")
            else:
                day_times[start_key] = None
                day_times[end_key] = None
        
        # Check if at least one day is configured
        has_configured_day = any(
            day_times[f'{day}_start'] and day_times[f'{day}_end'] 
            for day in days
        )
        
        if not has_configured_day:
            raise ValueError("At least one day must be configured with valid times")
        
        # Check if shift name already exists
        if check_shift_name_exists(db, name):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="A shift with this name already exists"
            )
        
        # Create new dynamic shift
        new_shift = Shift(
            name=name,
            description=description,
            shift_type='dynamic',
            is_active=is_active,
            **day_times  # Unpack all the day time fields
        )
        
        db.add(new_shift)
        db.commit()
        db.refresh(new_shift)
        
        logger.info(f"Dynamic shift '{new_shift.name}' created by user {current_user.username}")
        
        return RedirectResponse(
            url=f"/attendance/shift-manager?success=Dynamic shift '{new_shift.name}' created successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except ValueError as e:
        shifts = db.query(Shift).order_by(Shift.created_at.desc()).all()
        context = {
            "request": request,
            "user": current_user,
            "shifts": shifts,
            "error": str(e),
            "page_title": "Shift Manager"
        }
        return templates.TemplateResponse("staff/attendance/shift_manager.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating dynamic shift: {str(e)}")
        
        shifts = db.query(Shift).order_by(Shift.created_at.desc()).all()
        context = {
            "request": request,
            "user": current_user,
            "shifts": shifts,
            "error": "An unexpected error occurred. Please try again.",
            "page_title": "Shift Manager"
        }
        return templates.TemplateResponse("staff/attendance/shift_manager.html", context)

@router.delete("/shifts/{shift_id}")
async def delete_shift(
    shift_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a shift"""
    try:
        shift = get_shift_by_id(db, shift_id)
        
        # Check if shift has assigned employees
        if shift.assigned_employees_count > 0:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete shift '{shift.name}' as it has {shift.assigned_employees_count} assigned employees"
            )
        
        shift_name = shift.name
        db.delete(shift)
        db.commit()
        
        logger.info(f"Shift '{shift_name}' deleted by user {current_user.username}")
        
        return {
            "success": True, 
            "message": f"Shift '{shift_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting shift: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting shift"
        )

@router.post("/shifts/{shift_id}/toggle")
async def toggle_shift_status(
    shift_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Toggle shift active status"""
    try:
        shift = get_shift_by_id(db, shift_id)
        
        shift.is_active = not shift.is_active
        shift.updated_at = datetime.utcnow()
        db.commit()
        
        status_text = "activated" if shift.is_active else "deactivated"
        logger.info(f"Shift '{shift.name}' {status_text} by user {current_user.username}")
        
        return {
            "success": True, 
            "message": f"Shift '{shift.name}' {status_text} successfully", 
            "is_active": shift.is_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error toggling shift status: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating shift status"
        )

# API Endpoints
@router.get("/api/shifts", response_model=List[ShiftResponse])
async def get_shifts_api(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    shift_type: Optional[str] = Query(None, description="Filter by shift type"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """API endpoint to get shifts"""
    try:
        query = db.query(Shift)
        
        if shift_type:
            query = query.filter(Shift.shift_type == shift_type)
        
        if is_active is not None:
            query = query.filter(Shift.is_active == is_active)
        
        shifts = query.order_by(Shift.created_at.desc()).offset(offset).limit(limit).all()
        
        return [ShiftResponse.model_validate(shift) for shift in shifts]
        
    except Exception as e:
        logger.error(f"Error in shifts API: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching shifts"
        )

@router.get("/api/shifts/{shift_id}", response_model=ShiftResponse)
async def get_shift_api(
    shift_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """API endpoint to get a specific shift"""
    try:
        shift = get_shift_by_id(db, shift_id)
        return ShiftResponse.model_validate(shift)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in shift API: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching shift"
        )

@router.get("/api/stats")
async def get_attendance_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get attendance statistics"""
    try:
        total_shifts = db.query(Shift).count()
        active_shifts = db.query(Shift).filter(Shift.is_active == True).count()
        standard_shifts = db.query(Shift).filter(Shift.shift_type == 'standard').count()
        dynamic_shifts = db.query(Shift).filter(Shift.shift_type == 'dynamic').count()
        
        return {
            "total_shifts": total_shifts,
            "active_shifts": active_shifts,
            "inactive_shifts": total_shifts - active_shifts,
            "standard_shifts": standard_shifts,
            "dynamic_shifts": dynamic_shifts,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting attendance stats: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching statistics"
        )

# Health check
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy", 
            "service": "attendance_management",
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

logger.info("Attendance management router initialized successfully")