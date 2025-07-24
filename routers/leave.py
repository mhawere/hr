"""
Leave Management Router
Production-ready implementation with comprehensive leave management,
balance calculations, calendar views, reporting, and back dating support.
"""

import logging
import json
import os
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, date, timedelta
from calendar import monthrange
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc, extract

from fastapi import (
    APIRouter, Request, Form, Depends, HTTPException, UploadFile, File, 
    Query, BackgroundTasks
)
from fastapi import status as http_status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator

from models.database import get_db
from models.employee import Employee, User
from models.leave import Leave, LeaveType, PublicHoliday, LeaveBalance, LeaveStatusEnum
from models.attendance import ProcessedAttendance
from utils.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

ALLOWED_DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png'}
MAX_DOCUMENT_SIZE = 10 * 1024 * 1024
LEAVE_DOCUMENT_UPLOAD_DIR = Path("static/uploads/leave_documents")

WORKING_DAYS_PER_WEEK = 6
WORKING_DAYS_PER_MONTH = 2.5

class LeaveConfig:
    MAX_BACKDATE_MONTHS = 6
    BACKDATED_REQUIRES_APPROVAL = False
    AUTO_REPROCESS_ATTENDANCE = True
    WARN_ATTENDANCE_CONFLICTS = True

class LeaveRequest(BaseModel):
    employee_id: int
    leave_type_id: int
    start_date: date
    end_date: date
    reason: str
    comments: Optional[str] = ""
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        earliest_allowed = date.today() - timedelta(days=LeaveConfig.MAX_BACKDATE_MONTHS * 30)
        if v < earliest_allowed:
            raise ValueError(f'Leave dates cannot be more than {LeaveConfig.MAX_BACKDATE_MONTHS} months in the past (earliest: {earliest_allowed.strftime("%Y-%m-%d")})')
        return v
    
    @validator('end_date')
    def validate_end_after_start(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('End date must be after start date')
        return v

def ensure_leave_upload_directory():
    try:
        LEAVE_DOCUMENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(LEAVE_DOCUMENT_UPLOAD_DIR, 0o755)
        logger.info("Leave document upload directory initialized")
    except Exception as e:
        logger.error(f"Failed to create leave upload directory: {str(e)}")
        raise

def validate_leave_document(document: UploadFile) -> bool:
    if not document.filename:
        return False
    
    file_extension = Path(document.filename).suffix.lower()
    if file_extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Allowed: {', '.join(ALLOWED_DOCUMENT_EXTENSIONS)}"
        )
    
    document.file.seek(0, 2)
    file_size = document.file.tell()
    document.file.seek(0)
    
    if file_size > MAX_DOCUMENT_SIZE:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {MAX_DOCUMENT_SIZE // (1024*1024)}MB"
        )
    
    return True

def save_leave_document(document: UploadFile) -> str:
    if not document.filename:
        return ""
    
    validate_leave_document(document)
    ensure_leave_upload_directory()
    
    file_extension = Path(document.filename).suffix.lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = LEAVE_DOCUMENT_UPLOAD_DIR / unique_filename
    
    try:
        with open(file_path, "wb") as buffer:
            content = document.file.read()
            buffer.write(content)
        
        logger.info(f"Leave document saved: {unique_filename}")
        return f"uploads/leave_documents/{unique_filename}"
        
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        logger.error(f"Error saving leave document: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error saving document"
        )

def get_working_days_between_dates(start_date: date, end_date: date, public_holidays: List[date]) -> float:
    if start_date > end_date:
        return 0.0
    
    working_days = 0.0
    current_date = start_date
    
    while current_date <= end_date:
        if current_date.weekday() < 6:
            if current_date not in public_holidays:
                working_days += 1.0
        current_date += timedelta(days=1)
    
    return working_days

def check_leave_overlap(db: Session, employee_id: int, start_date: date, end_date: date, exclude_leave_id: Optional[int] = None) -> bool:
    query = db.query(Leave).filter(
        Leave.employee_id == employee_id,
        Leave.status == LeaveStatusEnum.ACTIVE,
        or_(
            and_(Leave.start_date <= start_date, Leave.end_date >= start_date),
            and_(Leave.start_date <= end_date, Leave.end_date >= end_date),
            and_(Leave.start_date >= start_date, Leave.end_date <= end_date)
        )
    )
    
    if exclude_leave_id:
        query = query.filter(Leave.id != exclude_leave_id)
    
    return query.first() is not None

def check_attendance_conflicts(db: Session, employee_id: int, start_date: date, end_date: date) -> Dict[str, Any]:
    existing_attendance = db.query(ProcessedAttendance).filter(
        ProcessedAttendance.employee_id == employee_id,
        ProcessedAttendance.date >= start_date,
        ProcessedAttendance.date <= end_date,
        ProcessedAttendance.status.in_(['present', 'late'])
    ).all()
    
    present_days = len([r for r in existing_attendance if r.is_present or r.status in ['present', 'late']])
    total_hours = sum(r.total_working_hours or 0 for r in existing_attendance)
    
    return {
        'conflicting_records': len(existing_attendance),
        'present_days': present_days,
        'total_hours': round(total_hours, 2),
        'dates': [r.date for r in existing_attendance]
    }

def calculate_employee_leave_balance(db: Session, employee_id: int, year: int) -> Dict[str, Any]:
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        return {}
    
    hire_date = employee.hire_date.date() if employee.hire_date else date(year, 1, 1)
    
    earliest_attendance = db.query(func.min(ProcessedAttendance.date)).filter(
        ProcessedAttendance.employee_id == employee_id,
        extract('year', ProcessedAttendance.date) == year
    ).scalar()
    
    if earliest_attendance:
        actual_start_date = min(hire_date, earliest_attendance)
        if earliest_attendance < hire_date:
            logger.info(f"Employee {employee_id}: Using attendance start date {earliest_attendance} instead of hire date {hire_date}")
            actual_start_date = earliest_attendance
        else:
            actual_start_date = hire_date
    else:
        actual_start_date = hire_date
    
    start_calc_date = max(actual_start_date, date(year, 1, 1))
    end_calc_date = min(date.today(), date(year, 12, 31))
    
    if start_calc_date > end_calc_date:
        return {}
    
    months_worked = 0
    current_month = start_calc_date.replace(day=1)
    today = date.today()
    
    while current_month.year == year and current_month <= today.replace(day=1):
        months_worked += 1
        current_month = add_months(current_month, 1)
    
    attendance_records = db.query(ProcessedAttendance).filter(
        ProcessedAttendance.employee_id == employee_id,
        ProcessedAttendance.date >= start_calc_date,
        ProcessedAttendance.date <= end_calc_date
    ).all()
    
    attendance_score = 0.0
    total_possible_working_days = 0
    
    public_holidays = [ph.date for ph in db.query(PublicHoliday).filter(
        PublicHoliday.date >= start_calc_date,
        PublicHoliday.date <= end_calc_date,
        PublicHoliday.is_active == True
    ).all()]
    
    attendance_by_date = {record.date: record for record in attendance_records}
    
    current_date = start_calc_date
    while current_date <= end_calc_date:
        if current_date.weekday() < 6 and current_date not in public_holidays:
            total_possible_working_days += 1
            
            if current_date in attendance_by_date:
                record = attendance_by_date[current_date]
                
                if (record.is_present or 
                    record.status.lower() in ['present', 'late', 'on_leave', 'public_holiday'] or 
                    record.total_working_hours > 0):
                    attendance_score += 1.0
                elif record.status.lower() in ['half_day', 'half day']:
                    attendance_score += 0.75
                else:
                    attendance_score += 0.5
            else:
                attendance_score += 0.5
        
        current_date += timedelta(days=1)
    
    attendance_percentage = (attendance_score / total_possible_working_days) if total_possible_working_days > 0 else 0
    attendance_percentage = max(attendance_percentage, 0.75)
    
    max_entitled_days = months_worked * WORKING_DAYS_PER_MONTH
    earned_days = max(max_entitled_days * 0.9, max_entitled_days * attendance_percentage)
    
    used_leave = db.query(func.sum(Leave.days_requested)).filter(
        Leave.employee_id == employee_id,
        Leave.status == LeaveStatusEnum.ACTIVE,
        extract('year', Leave.start_date) == year
    ).scalar() or 0.0
    
    remaining_earned = max(0, earned_days - used_leave)
    remaining_max = max(0, max_entitled_days - used_leave)
    
    actual_present_days = sum(1 for record in attendance_records 
                            if record.is_present or record.status.lower() in ['present', 'late'] 
                            or record.total_working_hours > 0)
    
    return {
        'employee_id': employee_id,
        'year': year,
        'months_worked': months_worked,
        'actual_start_date': actual_start_date.strftime('%Y-%m-%d'),
        'earliest_attendance': earliest_attendance.strftime('%Y-%m-%d') if earliest_attendance else None,
        'hire_date': hire_date.strftime('%Y-%m-%d'),
        'attendance_percentage': round(attendance_percentage * 100, 1),
        'total_possible_working_days': total_possible_working_days,
        'actual_working_days': actual_present_days,
        'attendance_score': round(attendance_score, 1),
        'max_entitled_days': round(max_entitled_days, 1),
        'earned_days': round(earned_days, 1),
        'used_days': round(used_leave, 1),
        'remaining_earned': round(remaining_earned, 1),
        'remaining_max': round(remaining_max, 1)
    }

def add_months(source_date: date, months: int) -> date:
    month = source_date.month - 1 + months
    year = source_date.year + month // 12
    month = month % 12 + 1
    day = min(source_date.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 2, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1])
    return date(year, month, day)

def update_leave_balance(db: Session, employee_id: int, leave_type_id: int, year: int):
    balance_data = calculate_employee_leave_balance(db, employee_id, year)
    
    if not balance_data:
        return
    
    existing_balance = db.query(LeaveBalance).filter(
        LeaveBalance.employee_id == employee_id,
        LeaveBalance.leave_type_id == leave_type_id,
        LeaveBalance.year == year
    ).first()
    
    if existing_balance:
        existing_balance.earned_days = balance_data['earned_days']
        existing_balance.max_entitled_days = balance_data['max_entitled_days']
        existing_balance.used_days = balance_data['used_days']
        existing_balance.remaining_days = balance_data['remaining_earned']
        existing_balance.last_calculated = datetime.utcnow()
    else:
        new_balance = LeaveBalance(
            employee_id=employee_id,
            leave_type_id=leave_type_id,
            year=year,
            earned_days=balance_data['earned_days'],
            max_entitled_days=balance_data['max_entitled_days'],
            used_days=balance_data['used_days'],
            remaining_days=balance_data['remaining_earned']
        )
        db.add(new_balance)

async def reprocess_attendance_for_leave(db: Session, employee_id: int, start_date: date, end_date: date):
    try:
        from services.attendance_service import AttendanceService
        attendance_service = AttendanceService(db)
        
        today = date.today()
        actual_end_date = min(end_date, today - timedelta(days=1))
        
        if start_date <= actual_end_date:
            await attendance_service._process_daily_attendance_for_employee(
                employee_id, start_date, actual_end_date
            )
            logger.info(f"Attendance reprocessed for employee {employee_id} from {start_date} to {actual_end_date}")
    except Exception as e:
        logger.error(f"Failed to reprocess attendance for employee {employee_id}: {e}")

@router.get("/employee/{employee_id}", response_class=HTMLResponse)
async def employee_leave_page(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        current_year = date.today().year
        balance_data = calculate_employee_leave_balance(db, employee_id, current_year)
        
        leave_types = db.query(LeaveType).filter(LeaveType.is_active == True).all()
        logger.info(f"Found {len(leave_types)} active leave types for employee {employee_id}")
        
        for lt in leave_types:
            logger.info(f"   - {lt.name}: {lt.max_days_per_year} days, Color: {lt.color}")
        
        if not leave_types:
            logger.warning(f"No active leave types found! This will cause empty dropdown.")
            all_leave_types = db.query(LeaveType).all()
            logger.info(f"Total leave types in database: {len(all_leave_types)}")
            for lt in all_leave_types:
                logger.info(f"   - {lt.name}: Active={lt.is_active}")
        
        leave_types_json = [
            {
                "id": lt.id,
                "name": lt.name,
                "max_days_per_year": lt.max_days_per_year,
                "color": lt.color,
                "description": lt.description
            }
            for lt in leave_types
        ]
        
        leaves = db.query(Leave).options(joinedload(Leave.leave_type)).filter(
            Leave.employee_id == employee_id
        ).order_by(desc(Leave.created_at)).all()
        
        public_holidays = db.query(PublicHoliday).filter(
            PublicHoliday.is_active == True,
            extract('year', PublicHoliday.date) == current_year
        ).order_by(PublicHoliday.date).all()
        
        public_holidays_json = [holiday.date.strftime('%Y-%m-%d') for holiday in public_holidays]
        
        today = date.today()
        active_leaves = [l for l in leaves if l.status == LeaveStatusEnum.ACTIVE and l.end_date >= today]
        past_leaves = [l for l in leaves if l.status in [LeaveStatusEnum.COMPLETED, LeaveStatusEnum.CANCELLED] or l.end_date < today]
        upcoming_leaves = [l for l in active_leaves if l.start_date > today]
        
        earliest_allowed = today - timedelta(days=LeaveConfig.MAX_BACKDATE_MONTHS * 30)
        
        context = {
            "request": request,
            "user": current_user,
            "employee": employee,
            "balance_data": balance_data,
            "leave_types": leave_types,
            "leave_types_json": leave_types_json,
            "leaves": leaves,
            "active_leaves": active_leaves,
            "past_leaves": past_leaves,
            "upcoming_leaves": upcoming_leaves,
            "public_holidays": public_holidays,
            "public_holidays_json": public_holidays_json,
            "current_year": current_year,
            "today": today,
            "earliest_allowed_date": earliest_allowed.strftime('%Y-%m-%d'),
            "max_backdate_months": LeaveConfig.MAX_BACKDATE_MONTHS,
            "page_title": f"{employee.first_name} {employee.last_name} - Leave Management"
        }
        
        return templates.TemplateResponse("staff/leave/staff_leave.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading employee leave page: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading leave page"
        )

@router.get("/employee/{employee_id}/calculation-breakdown")
async def get_calculation_breakdown(
    employee_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        current_year = date.today().year
        balance_data = calculate_employee_leave_balance(db, employee_id, current_year)
        
        if not balance_data:
            raise HTTPException(status_code=400, detail="Unable to calculate leave balance")
        
        hire_date = employee.hire_date.date() if employee.hire_date else date(current_year, 1, 1)
        
        earliest_attendance = db.query(func.min(ProcessedAttendance.date)).filter(
            ProcessedAttendance.employee_id == employee_id,
            extract('year', ProcessedAttendance.date) == current_year
        ).scalar()
        
        attendance_count = db.query(func.count(ProcessedAttendance.id)).filter(
            ProcessedAttendance.employee_id == employee_id,
            extract('year', ProcessedAttendance.date) == current_year
        ).scalar()
        
        breakdown_data = {
            'employee_name': f"{employee.first_name} {employee.last_name}",
            'employee_id': employee.employee_id,
            'hire_date': balance_data['hire_date'],
            'earliest_attendance': balance_data.get('earliest_attendance'),
            'actual_start_date': balance_data.get('actual_start_date'),
            'period': f"January 2025 to {date.today().strftime('%B %Y')}",
            'current_date': date.today().strftime('%Y-%m-%d'),
            'start_calc_date': balance_data.get('actual_start_date'),
            'end_calc_date': date.today().strftime('%Y-%m-%d'),
            'months_worked': balance_data['months_worked'],
            'total_possible_working_days': balance_data['total_possible_working_days'],
            'attendance_score': balance_data['attendance_score'],
            'actual_working_days': balance_data['actual_working_days'],
            'raw_attendance_percentage': balance_data['attendance_percentage'],
            'attendance_percentage': balance_data['attendance_percentage'],
            'max_entitled_days': balance_data['max_entitled_days'],
            'attendance_based_days': round(balance_data['max_entitled_days'] * balance_data['attendance_percentage'] / 100, 2),
            'generous_minimum_days': round(balance_data['max_entitled_days'] * 0.9, 2),
            'earned_days': balance_data['earned_days'],
            'used_days': balance_data['used_days'],
            'remaining_earned': balance_data['remaining_earned'],
            'debug_info': {
                'hire_date_from_db': str(employee.hire_date),
                'earliest_attendance_record': str(earliest_attendance) if earliest_attendance else 'None',
                'total_attendance_records': attendance_count,
                'calculation_year': current_year,
                'smart_date_detection': True,
                'date_source': 'attendance_records' if earliest_attendance and earliest_attendance < hire_date else 'hire_date'
            }
        }
        
        return {
            "success": True,
            "data": breakdown_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting calculation breakdown: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting calculation breakdown: {str(e)}"
        )

@router.post("/employee/{employee_id}/recalculate")
async def recalculate_employee_leave_balance(
    employee_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        current_year = date.today().year
        
        logger.info(f"Recalculating leave balance for employee {employee_id} by user {current_user.username}")
        balance_data = calculate_employee_leave_balance(db, employee_id, current_year)
        
        if not balance_data:
            raise HTTPException(status_code=400, detail="Unable to calculate leave balance")
        
        leave_types = db.query(LeaveType).filter(LeaveType.is_active == True).all()
        
        for leave_type in leave_types:
            existing_balance = db.query(LeaveBalance).filter(
                LeaveBalance.employee_id == employee_id,
                LeaveBalance.leave_type_id == leave_type.id,
                LeaveBalance.year == current_year
            ).first()
            
            if existing_balance:
                existing_balance.earned_days = balance_data['earned_days']
                existing_balance.max_entitled_days = balance_data['max_entitled_days']
                existing_balance.used_days = balance_data['used_days']
                existing_balance.remaining_days = balance_data['remaining_earned']
                existing_balance.last_calculated = datetime.utcnow()
                existing_balance.updated_at = datetime.utcnow()
            else:
                new_balance = LeaveBalance(
                    employee_id=employee_id,
                    leave_type_id=leave_type.id,
                    year=current_year,
                    earned_days=balance_data['earned_days'],
                    max_entitled_days=balance_data['max_entitled_days'],
                    used_days=balance_data['used_days'],
                    remaining_days=balance_data['remaining_earned'],
                    last_calculated=datetime.utcnow()
                )
                db.add(new_balance)
        
        db.commit()
        
        balance_data['last_calculated'] = datetime.utcnow()
        
        logger.info(f"Leave balance recalculated successfully for employee {employee_id}")
        
        return {
            "success": True,
            "message": "Leave balance recalculated successfully",
            "balance_data": balance_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error recalculating leave balance for employee {employee_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recalculating leave balance: {str(e)}"
        )

@router.post("/employee/{employee_id}/create")
async def create_leave(
    employee_id: int,
    request: Request,
    leave_type_id: int = Form(...),
    start_date: date = Form(...),
    end_date: date = Form(...),
    reason: str = Form(..., min_length=10),
    comments: str = Form(""),
    attachment: UploadFile = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        leave_type = db.query(LeaveType).filter(
            LeaveType.id == leave_type_id,
            LeaveType.is_active == True
        ).first()
        if not leave_type:
            raise HTTPException(status_code=400, detail="Invalid leave type")
        
        today = date.today()
        earliest_allowed = today - timedelta(days=LeaveConfig.MAX_BACKDATE_MONTHS * 30)
        
        if start_date < earliest_allowed:
            raise HTTPException(
                status_code=400, 
                detail=f"Leave cannot be more than {LeaveConfig.MAX_BACKDATE_MONTHS} months in the past. Earliest allowed: {earliest_allowed.strftime('%Y-%m-%d')}"
            )
        
        if end_date < start_date:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        
        is_backdated_request = start_date < today
        
        if is_backdated_request:
            logger.info(f"Processing backdated leave request: {start_date} to {end_date}")
            
            if LeaveConfig.WARN_ATTENDANCE_CONFLICTS:
                conflicts = check_attendance_conflicts(db, employee_id, start_date, end_date)
                if conflicts['conflicting_records'] > 0:
                    logger.warning(f"Backdated leave overlaps with {conflicts['conflicting_records']} existing attendance records")
        
        if check_leave_overlap(db, employee_id, start_date, end_date):
            raise HTTPException(status_code=400, detail="Leave dates overlap with existing leave")
        
        public_holidays = [ph.date for ph in db.query(PublicHoliday).filter(
            PublicHoliday.date >= start_date,
            PublicHoliday.date <= end_date,
            PublicHoliday.is_active == True
        ).all()]
        
        days_requested = get_working_days_between_dates(start_date, end_date, public_holidays)
        
        if days_requested <= 0:
            raise HTTPException(status_code=400, detail="No working days in the selected period")
        
        leave_year = start_date.year
        balance_data = calculate_employee_leave_balance(db, employee_id, leave_year)
        
        if is_backdated_request:
            current_used = balance_data.get('used_days', 0)
            projected_used = current_used + days_requested
            remaining_after = balance_data.get('earned_days', 0) - projected_used
            
            if remaining_after < -5.0:
                logger.warning(f"Backdated leave would result in significantly negative balance: {remaining_after}")
                if not LeaveConfig.BACKDATED_REQUIRES_APPROVAL:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Backdated leave would exceed balance by {abs(remaining_after):.1f} days"
                    )
            
            logger.info(f"Backdated leave impact: Current used: {current_used}, New total: {projected_used}, Remaining: {remaining_after}")
        else:
            if days_requested > balance_data.get('remaining_earned', 0):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient leave balance. Requested: {days_requested}, Available: {balance_data.get('remaining_earned', 0)}"
                )
        
        attachment_path = ""
        if attachment and attachment.filename:
            attachment_path = save_leave_document(attachment)
        
        backdated_prefix = "[BACKDATED] " if is_backdated_request else ""
        final_comments = f"{backdated_prefix}{comments.strip()}" if comments else (backdated_prefix.strip() if backdated_prefix else None)
        
        new_leave = Leave(
            employee_id=employee_id,
            leave_type_id=leave_type_id,
            start_date=start_date,
            end_date=end_date,
            days_requested=days_requested,
            reason=reason.strip(),
            comments=final_comments,
            attachment=attachment_path,
            created_by=current_user.id
        )
        
        db.add(new_leave)
        db.commit()
        db.refresh(new_leave)
        
        if is_backdated_request and LeaveConfig.AUTO_REPROCESS_ATTENDANCE:
            logger.info(f"Triggering attendance reprocessing for backdated leave period")
            try:
                await reprocess_attendance_for_leave(db, employee_id, start_date, min(end_date, today - timedelta(days=1)))
                logger.info(f"Attendance reprocessed for backdated leave period")
            except Exception as reprocess_error:
                logger.error(f"Failed to reprocess attendance for backdated leave: {reprocess_error}")
        
        update_leave_balance(db, employee_id, leave_type_id, leave_year)
        db.commit()
        
        success_message = f"{'Backdated l' if is_backdated_request else 'L'}eave request created successfully"
        logger.info(f"Leave created for employee {employee_id} by user {current_user.username}")
        
        return RedirectResponse(
            url=f"/leave/employee/{employee_id}?success={success_message}",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating leave: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating leave request"
        )

@router.get("/holidays/manage", response_class=HTMLResponse)
async def manage_public_holidays(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Public holidays management page"""
    try:
        context = {
            "request": request,
            "user": current_user,
            "page_title": "Public Holidays Management"
        }
        return templates.TemplateResponse("components/publicholiday.html", context)
        
    except Exception as e:
        logger.error(f"Error loading public holidays page: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading public holidays page"
        )

@router.get("/holidays/list")
async def list_public_holidays(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    year: Optional[int] = Query(None),
    status: Optional[str] = Query(None)
):
    """List all public holidays"""
    try:
        query = db.query(PublicHoliday)
        
        if year:
            start_date = date(year, 1, 1)
            end_date = date(year, 12, 31)
            query = query.filter(PublicHoliday.date.between(start_date, end_date))
        
        if status == "active":
            query = query.filter(PublicHoliday.is_active == True)
        elif status == "inactive":
            query = query.filter(PublicHoliday.is_active == False)
        
        holidays = query.order_by(PublicHoliday.date).all()
        
        return {
            "success": True,
            "holidays": [
                {
                    "id": holiday.id,
                    "date": holiday.date.isoformat(),
                    "name": holiday.name,
                    "description": holiday.description,
                    "is_active": holiday.is_active,
                    "created_at": holiday.created_at.isoformat()
                }
                for holiday in holidays
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching public holidays: {str(e)}")
        return {
            "success": False,
            "message": "Error fetching holidays",
            "holidays": []
        }

@router.post("/cancel/{leave_id}")
async def cancel_leave(
    leave_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        leave = db.query(Leave).filter(Leave.id == leave_id).first()
        if not leave:
            raise HTTPException(status_code=404, detail="Leave not found")
        
        if leave.status != LeaveStatusEnum.ACTIVE:
            raise HTTPException(status_code=400, detail="Leave cannot be cancelled")
        
        today = date.today()
        if leave.start_date <= today:
            raise HTTPException(status_code=400, detail="Cannot cancel leave that has already started")
        
        leave.status = LeaveStatusEnum.CANCELLED
        leave.updated_at = datetime.utcnow()
        
        update_leave_balance(db, leave.employee_id, leave.leave_type_id, leave.start_date.year)
        
        db.commit()
        
        logger.info(f"Leave {leave_id} cancelled by user {current_user.username}")
        
        return {"success": True, "message": "Leave cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error cancelling leave: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error cancelling leave"
        )

@router.get("/employee/{employee_id}/calendar", response_class=HTMLResponse)
async def leave_calendar(
    employee_id: int,
    request: Request,
    year: int = Query(None),
    month: int = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        today = date.today()
        target_year = year or today.year
        target_month = month or today.month
        
        month_start = date(target_year, target_month, 1)
        days_in_month = monthrange(target_year, target_month)[1]
        month_end = date(target_year, target_month, days_in_month)
        
        leaves = db.query(Leave).options(joinedload(Leave.leave_type)).filter(
            Leave.employee_id == employee_id,
            Leave.status == LeaveStatusEnum.ACTIVE,
            or_(
                and_(Leave.start_date <= month_start, Leave.end_date >= month_start),
                and_(Leave.start_date <= month_end, Leave.end_date >= month_end),
                and_(Leave.start_date >= month_start, Leave.end_date <= month_end)
            )
        ).all()
        
        public_holidays = db.query(PublicHoliday).filter(
            PublicHoliday.date >= month_start,
            PublicHoliday.date <= month_end,
            PublicHoliday.is_active == True
        ).all()
        
        calendar_data = []
        for day in range(1, days_in_month + 1):
            day_date = date(target_year, target_month, day)
            day_leaves = []
            day_holidays = []
            
            for leave in leaves:
                if leave.start_date <= day_date <= leave.end_date:
                    day_leaves.append(leave)
            
            for holiday in public_holidays:
                if holiday.date == day_date:
                    day_holidays.append(holiday)
            
            is_backdated = any(leave.start_date < leave.created_at.date() for leave in day_leaves)
            
            calendar_data.append({
                'date': day_date,
                'day': day,
                'is_weekend': day_date.weekday() >= 6,
                'is_today': day_date == today,
                'leaves': day_leaves,
                'holidays': day_holidays,
                'is_working_day': day_date.weekday() < 6 and not day_holidays,
                'is_backdated': is_backdated
            })
        
        prev_month = target_month - 1 if target_month > 1 else 12
        prev_year = target_year if target_month > 1 else target_year - 1
        next_month = target_month + 1 if target_month < 12 else 1
        next_year = target_year if target_month < 12 else target_year + 1
        
        context = {
            "request": request,
            "user": current_user,
            "employee": employee,
            "calendar_data": calendar_data,
            "target_year": target_year,
            "target_month": target_month,
            "month_name": month_start.strftime('%B'),
            "prev_month": prev_month,
            "prev_year": prev_year,
            "next_month": next_month,
            "next_year": next_year,
            "page_title": f"{employee.first_name} {employee.last_name} - Leave Calendar"
        }
        
        return templates.TemplateResponse("staff/leave/leave_calendar.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading leave calendar: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading calendar"
        )

@router.get("/reports", response_class=HTMLResponse)
async def leave_reports(
    request: Request,
    year: int = Query(None),
    department_id: int = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        target_year = year or date.today().year
        
        employee_query = db.query(Employee).filter(Employee.status.in_(['Active']))
        
        if department_id:
            employee_query = employee_query.filter(Employee.department_id == department_id)
        
        employees = employee_query.all()
        
        report_data = []
        total_entitled = 0
        total_used = 0
        total_remaining = 0
        
        for employee in employees:
            balance_data = calculate_employee_leave_balance(db, employee.id, target_year)
            
            leave_by_type = db.query(
                LeaveType.name,
                func.sum(Leave.days_requested).label('days_used')
            ).join(Leave).filter(
                Leave.employee_id == employee.id,
                Leave.status == LeaveStatusEnum.ACTIVE,
                extract('year', Leave.start_date) == target_year
            ).group_by(LeaveType.name).all()
            
            leave_breakdown = {lt.name: lt.days_used for lt in leave_by_type}
            
            report_data.append({
                'employee': employee,
                'balance_data': balance_data,
                'leave_breakdown': leave_breakdown
            })
            
            total_entitled += balance_data.get('max_entitled_days', 0)
            total_used += balance_data.get('used_days', 0)
            total_remaining += balance_data.get('remaining_earned', 0)
        
        departments = db.query(Employee.department_id, func.count(Employee.id).label('employee_count')).filter(
            Employee.status.in_(['Active'])
        ).group_by(Employee.department_id).all()
        
        leave_type_stats = db.query(
            LeaveType.name,
            func.count(Leave.id).label('request_count'),
            func.sum(Leave.days_requested).label('total_days')
        ).join(Leave).filter(
            extract('year', Leave.start_date) == target_year,
            Leave.status == LeaveStatusEnum.ACTIVE
        ).group_by(LeaveType.name).all()
        
        context = {
            "request": request,
            "user": current_user,
            "report_data": report_data,
            "target_year": target_year,
            "department_id": department_id,
            "total_entitled": round(total_entitled, 1),
            "total_used": round(total_used, 1),
            "total_remaining": round(total_remaining, 1),
            "leave_type_stats": leave_type_stats,
            "page_title": f"Leave Reports - {target_year}"
        }
        
        return templates.TemplateResponse("staff/leave/leave_reports.html", context)
        
    except Exception as e:
        logger.error(f"Error loading leave reports: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading reports"
        )

@router.get("/reports/backdated")
async def backdated_leave_report(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        backdated_leaves = db.query(Leave).join(Employee).filter(
            func.date(Leave.created_at) > Leave.start_date,
            Leave.status == LeaveStatusEnum.ACTIVE
        ).order_by(desc(Leave.created_at)).all()
        
        context = {
            "request": request,
            "user": current_user,
            "backdated_leaves": backdated_leaves,
            "page_title": "Backdated Leave Report"
        }
        
        return templates.TemplateResponse("staff/leave/backdated_report.html", context)
        
    except Exception as e:
        logger.error(f"Error loading backdated leave report: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading report")

@router.post("/holidays/add")
async def add_public_holiday(
    holiday_date: date = Form(...),
    name: str = Form(..., min_length=2),
    description: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        existing = db.query(PublicHoliday).filter(PublicHoliday.date == holiday_date).first()
        if existing:
            raise HTTPException(status_code=400, detail="Holiday already exists for this date")
        
        new_holiday = PublicHoliday(
            date=holiday_date,
            name=name.strip(),
            description=description.strip() if description else None
        )
        
        db.add(new_holiday)
        db.commit()
        
        logger.info(f"Public holiday added: {name} on {holiday_date} by user {current_user.username}")
        
        return {"success": True, "message": "Public holiday added successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding public holiday: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error adding holiday"
        )

@router.delete("/holidays/{holiday_id}")
async def delete_public_holiday(
    holiday_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        holiday = db.query(PublicHoliday).filter(PublicHoliday.id == holiday_id).first()
        if not holiday:
            raise HTTPException(status_code=404, detail="Holiday not found")
        
        db.delete(holiday)
        db.commit()
        
        logger.info(f"Public holiday deleted: {holiday.name} by user {current_user.username}")
        
        return {"success": True, "message": "Public holiday deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting public holiday: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting holiday"
        )

@router.get("/api/employee/{employee_id}/balance")
async def get_leave_balance_api(
    employee_id: int,
    year: int = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        target_year = year or date.today().year
        balance_data = calculate_employee_leave_balance(db, employee_id, target_year)
        
        return {
            "success": True,
            "data": balance_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting leave balance: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching leave balance"
        )

@router.get("/employee/{employee_id}/create-form", response_class=HTMLResponse)
async def show_create_leave_form(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        current_year = date.today().year
        balance_data = calculate_employee_leave_balance(db, employee_id, current_year)
        
        leave_types = db.query(LeaveType).filter(LeaveType.is_active == True).all()
        
        public_holidays = db.query(PublicHoliday).filter(
            PublicHoliday.is_active == True,
            extract('year', PublicHoliday.date) == current_year
        ).order_by(PublicHoliday.date).all()
        
        public_holidays_json = [holiday.date.strftime('%Y-%m-%d') for holiday in public_holidays]
        
        today = date.today()
        earliest_allowed = today - timedelta(days=LeaveConfig.MAX_BACKDATE_MONTHS * 30)
        max_future_date = date(current_year + 1, 12, 31)
        
        context = {
            "request": request,
            "user": current_user,
            "employee": employee,
            "balance_data": balance_data,
            "leave_types": leave_types,
            "public_holidays_json": public_holidays_json,
            "earliest_allowed_date": earliest_allowed.strftime('%Y-%m-%d'),
            "max_future_date": max_future_date.strftime('%Y-%m-%d'),
            "form_title": "Create Leave Request",
            "form_description": f"Submit a new leave request for {employee.first_name} {employee.last_name}",
            "form_action": f"/leave/employee/{employee_id}/create",
            "page_title": "Create Leave Request"
        }
        
        return templates.TemplateResponse("staff/leave/leave_form.html", context)
        
    except Exception as e:
        logger.error(f"Error showing leave form: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading form")

# Add edit form route
@router.get("/edit/{leave_id}", response_class=HTMLResponse)
async def show_edit_leave_form(
    leave_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        leave = db.query(Leave).filter(Leave.id == leave_id).first()
        if not leave:
            raise HTTPException(status_code=404, detail="Leave not found")
        
        employee = leave.employee
        current_year = leave.start_date.year
        balance_data = calculate_employee_leave_balance(db, employee.id, current_year)
        
        leave_types = db.query(LeaveType).filter(LeaveType.is_active == True).all()
        
        public_holidays = db.query(PublicHoliday).filter(
            PublicHoliday.is_active == True,
            extract('year', PublicHoliday.date) == current_year
        ).order_by(PublicHoliday.date).all()
        
        public_holidays_json = [holiday.date.strftime('%Y-%m-%d') for holiday in public_holidays]
        
        today = date.today()
        earliest_allowed = today - timedelta(days=LeaveConfig.MAX_BACKDATE_MONTHS * 30)
        max_future_date = date(current_year + 1, 12, 31)
        
        context = {
            "request": request,
            "user": current_user,
            "employee": employee,
            "leave": leave,
            "balance_data": balance_data,
            "leave_types": leave_types,
            "public_holidays_json": public_holidays_json,
            "earliest_allowed_date": earliest_allowed.strftime('%Y-%m-%d'),
            "max_future_date": max_future_date.strftime('%Y-%m-%d'),
            "selected_leave_type": leave.leave_type_id,
            "start_date": leave.start_date.strftime('%Y-%m-%d'),
            "end_date": leave.end_date.strftime('%Y-%m-%d'),
            "reason": leave.reason,
            "comments": leave.comments or "",
            "form_title": "Edit Leave Request",
            "form_description": f"Update leave request for {employee.first_name} {employee.last_name}",
            "form_action": f"/leave/update/{leave_id}",
            "page_title": "Edit Leave Request"
        }
        
        return templates.TemplateResponse("staff/leave/leave_form.html", context)
        
    except Exception as e:
        logger.error(f"Error showing edit form: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading form")

# Add update route
@router.post("/update/{leave_id}")
async def update_leave(
    leave_id: int,
    request: Request,
    leave_type_id: int = Form(...),
    start_date: date = Form(...),
    end_date: date = Form(...),
    reason: str = Form(..., min_length=10),
    comments: str = Form(""),
    attachment: UploadFile = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        leave = db.query(Leave).filter(Leave.id == leave_id).first()
        if not leave:
            raise HTTPException(status_code=404, detail="Leave not found")
        
        if leave.status != LeaveStatusEnum.ACTIVE:
            raise HTTPException(status_code=400, detail="Cannot edit completed or cancelled leave")
        
        # Validate dates
        today = date.today()
        if leave.start_date <= today:
            raise HTTPException(status_code=400, detail="Cannot edit leave that has already started")
        
        # Check for overlaps (excluding current leave)
        if check_leave_overlap(db, leave.employee_id, start_date, end_date, exclude_leave_id=leave_id):
            raise HTTPException(status_code=400, detail="Leave dates overlap with existing leave")
        
        # Calculate days
        public_holidays = [ph.date for ph in db.query(PublicHoliday).filter(
            PublicHoliday.date >= start_date,
            PublicHoliday.date <= end_date,
            PublicHoliday.is_active == True
        ).all()]
        
        days_requested = get_working_days_between_dates(start_date, end_date, public_holidays)
        
        # Update leave
        leave.leave_type_id = leave_type_id
        leave.start_date = start_date
        leave.end_date = end_date
        leave.days_requested = days_requested
        leave.reason = reason.strip()
        leave.comments = comments.strip() if comments else None
        leave.updated_at = datetime.utcnow()
        
        # Handle attachment update
        if attachment and attachment.filename:
            attachment_path = save_leave_document(attachment)
            leave.attachment = attachment_path
        
        db.commit()
        
        # Update balance
        update_leave_balance(db, leave.employee_id, leave_type_id, start_date.year)
        db.commit()
        
        logger.info(f"Leave {leave_id} updated by user {current_user.username}")
        
        return RedirectResponse(
            url=f"/leave/employee/{leave.employee_id}?success=Leave updated successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating leave: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating leave")

@router.get("/download/{leave_id}/attachment")
async def download_leave_attachment(
    leave_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        leave = db.query(Leave).filter(Leave.id == leave_id).first()
        if not leave or not leave.attachment:
            raise HTTPException(status_code=404, detail="Attachment not found")
        
        file_path = Path("static") / leave.attachment
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        filename = leave.attachment.split('/')[-1]
        return FileResponse(
            path=file_path,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading leave attachment: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error downloading file"
        )

try:
    ensure_leave_upload_directory()
    logger.info("Leave management router initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize leave management router: {str(e)}")