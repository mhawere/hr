"""
Biometric Attendance Router
Handles biometric data synchronization and attendance management
ALL API CONNECTIONS ARE READ-ONLY
"""

import logging
import asyncio
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from fastapi import Form
from fastapi.responses import StreamingResponse
from io import BytesIO
import io
import traceback

from fastapi import (
    APIRouter, Request, Depends, HTTPException, Query, BackgroundTasks
)
from fastapi import status as http_status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from models.database import get_db
from models.employee import User, Employee, Department
from models.custom_fields import CustomField  # Import from existing file
from models.attendance import RawAttendance, ProcessedAttendance, AttendanceSyncLog
from models.report_template import ReportTemplate
from services.attendance_service import AttendanceService
from services.biometric_service import BiometricAPIService
from utils.auth import get_current_user
from routers.attendance import Shift
from services.attendance_badge_service import AttendanceBadgeService

# Import the staff attendance service
try:
    from services.staff_att import StaffAttendanceService
except ImportError:
    logger.warning("StaffAttendanceService not found, some routes may not work")
    StaffAttendanceService = None

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/biometric", tags=["biometric"])
templates = Jinja2Templates(directory="templates")

# Pydantic models
class AttendanceRecord(BaseModel):
    date: date
    check_in_time: Optional[datetime]
    check_out_time: Optional[datetime]
    expected_start_time: Optional[str]
    expected_end_time: Optional[str]
    total_working_hours: float
    is_present: bool
    is_late: bool
    is_early_departure: bool
    late_minutes: int
    status: str
    shift_name: Optional[str] = None

class AttendanceSummary(BaseModel):
    total_days: int
    present_days: int
    absent_days: int
    late_days: int
    total_hours: float
    attendance_percentage: float

class SyncResult(BaseModel):
    success: bool
    message: str
    records_fetched: Optional[int] = 0
    records_saved: Optional[int] = 0
    days_processed: Optional[int] = 0
    sync_type: Optional[str] = None

class ReportRequest(BaseModel):
    start_date: str
    end_date: str

# Helper Functions
def get_employee_by_id(db: Session, employee_id: int) -> Employee:
    """Get employee by ID with error handling"""
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Employee with ID {employee_id} not found"
        )
    return employee

def get_current_month_dates():
    """Get first and last day of current month"""
    today = date.today()
    first_day = date(today.year, today.month, 1)
    
    # Get last day of month
    if today.month == 12:
        last_day = date(today.year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(today.year, today.month + 1, 1) - timedelta(days=1)
    
    return first_day, min(last_day, today) # Don't go beyond today

# Routes
@router.get("/", response_class=HTMLResponse)
async def biometric_home(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Biometric system home page"""
    return RedirectResponse(url="/staff/view")

@router.get("/employee/{employee_id}/attendance", response_class=HTMLResponse)
async def employee_attendance_page(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Display employee attendance page"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Default to current month if no dates provided
        if not start_date or not end_date:
            first_day, last_day = get_current_month_dates()
            start_date = first_day.isoformat()
            end_date = last_day.isoformat()
        else:
            first_day = datetime.fromisoformat(start_date).date()
            last_day = datetime.fromisoformat(end_date).date()
        
        # Check if employee has biometric ID
        if not employee.biometric_id:
            context = {
                "request": request,
                "user": current_user,
                "employee": employee,
                "error": "No biometric ID assigned to this employee",
                "start_date": first_day,
                "end_date": last_day,
                "records": [],
                "summary": {
                    'total_days': 0,
                    'present_days': 0,
                    'absent_days': 0,
                    'late_days': 0,
                    'total_hours': 0,
                    'attendance_percentage': 0
                },
                "page_title": "Attendance - " + (employee.first_name or 'Employee') + " " + (employee.last_name or '')
            }
            return templates.TemplateResponse("staff/attendance/staff_attendance.html", context)
        
        # Get attendance data
        attendance_service = AttendanceService(db)
        attendance_data = attendance_service.get_employee_attendance_summary(
            employee_id=employee_id,
            start_date=first_day,
            end_date=last_day
        )
        
        # Format records for template
        formatted_records = []
        for record in attendance_data['records']:
            shift_name = None
            if record.shift:
                shift_name = record.shift.name
            
            formatted_record = {
                'date': record.date,
                'check_in_time': record.check_in_time,
                'check_out_time': record.check_out_time,
                'expected_start_time': record.expected_start_time.strftime('%H:%M') if record.expected_start_time else None,
                'expected_end_time': record.expected_end_time.strftime('%H:%M') if record.expected_end_time else None,
                'total_working_hours': record.total_working_hours,
                'is_present': record.is_present,
                'is_late': record.is_late,
                'is_early_departure': record.is_early_departure,
                'late_minutes': record.late_minutes,
                'early_departure_minutes': record.early_departure_minutes,
                'status': record.status,
                'shift_name': shift_name,
                'notes': record.notes
            }
            formatted_records.append(formatted_record)
        
        # Check last sync status
        last_sync = db.query(AttendanceSyncLog).filter(
            AttendanceSyncLog.employee_id == employee_id
        ).order_by(desc(AttendanceSyncLog.created_at)).first()
        
        context = {
            "request": request,
            "user": current_user,
            "employee": employee,
            "records": formatted_records,
            "summary": attendance_data['summary'],
            "start_date": first_day,
            "end_date": last_day,
            "last_sync": last_sync,
            "page_title": "Attendance - " + employee.first_name + " " + employee.last_name
        }
        
        return templates.TemplateResponse("staff/attendance/staff_attendance.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading attendance page for employee {employee_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading attendance data"
        )

# NEW SERVER-SIDE ATTENDANCE ROUTES (HTMX COMPATIBLE)
@router.get("/staff/attendance/{employee_id}", response_class=HTMLResponse)
async def staff_attendance_page(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    period: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Main staff attendance page with server-side rendering"""
    try:
        if StaffAttendanceService is None:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="StaffAttendanceService not available"
            )
        
        service = StaffAttendanceService(db)
        
        # Parse date range
        if period:
            start_date_obj, end_date_obj = service.parse_filter_dates(period)
        elif start_date and end_date:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            start_date_obj, end_date_obj = service.get_current_month_dates()
        
        # Get attendance data
        attendance_data = service.get_attendance_data(employee_id, start_date_obj, end_date_obj)
        
        context = {
            "request": request,
            "user": current_user,
            **attendance_data,
            "page_title": "Attendance - Employee"


        }
        
        return templates.TemplateResponse("staff/attendance/staff_attendance.html", context)
        
    except Exception as e:
        logger.error(f"Error loading staff attendance page: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading attendance page"
        )

@router.get("/staff/attendance/{employee_id}/filter")
async def filter_attendance(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    period: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Filter attendance data and return content fragment"""
    try:
        if StaffAttendanceService is None:
            return HTMLResponse('<div class="error">StaffAttendanceService not available</div>')
        
        service = StaffAttendanceService(db)
        
        # Parse date range
        if period:
            start_date_obj, end_date_obj = service.parse_filter_dates(period)
        elif start_date and end_date:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            start_date_obj, end_date_obj = service.get_current_month_dates()
        
        # Get filtered attendance data
        attendance_data = service.get_attendance_data(employee_id, start_date_obj, end_date_obj)
        
        context = {
            "request": request,
            **attendance_data
        }
        
        return templates.TemplateResponse("staff/attendance/fragments/attendance_content.html", context)
        
    except Exception as e:
        logger.error(f"Error filtering attendance: {str(e)}")
        return HTMLResponse(f'<div class="error">Error filtering data: {str(e)}</div>')

@router.get("/staff/attendance/{employee_id}/exception-form")
async def exception_form_modal(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    date: str = Query(...)
):
    """Show exception form modal"""
    try:
        if StaffAttendanceService is None:
            return HTMLResponse('<div class="error">StaffAttendanceService not available</div>')
        
        service = StaffAttendanceService(db)
        employee = service.get_employee_by_id(employee_id)
        
        # Format date for display
        try:
            parsed_date = datetime.strptime(date, '%Y-%m-%d').date()
            formatted_date = parsed_date.strftime('%A, %B %d, %Y')
        except ValueError:
            formatted_date = date
        
        context = {
            "request": request,
            "employee": employee,
            "exception_date": date,
            "formatted_date": formatted_date
        }
        
        return templates.TemplateResponse("staff/attendance/modals/exception_form.html", context)
        
    except Exception as e:
        logger.error(f"Error showing exception form: {str(e)}")
        return HTMLResponse(f'<div class="error">Error loading form: {str(e)}</div>')

@router.post("/staff/employee/{employee_id}/add-exception")
async def add_exception_server_side(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    date: str = Form(...),
    reason: str = Form(...),
    category: str = Form("")
):
    """Add attendance exception server-side"""
    try:
        if StaffAttendanceService is None:
            return HTMLResponse('<div class="error">StaffAttendanceService not available</div>')
        
        service = StaffAttendanceService(db)
        
        result = service.add_attendance_exception(
            employee_id=employee_id,
            exception_date=date,
            reason=reason,
            category=category,
            current_user_id=current_user.id
        )
        
        if result['success']:
            # Return updated attendance content
            parsed_date = datetime.strptime(date, '%Y-%m-%d').date()
            # Get current date range (you might want to store this in session)
            start_date_obj, end_date_obj = service.get_current_month_dates()
            
            attendance_data = service.get_attendance_data(employee_id, start_date_obj, end_date_obj)
            
            context = {
                "request": request,
                **attendance_data
            }
            
            response = templates.TemplateResponse("staff/attendance/fragments/attendance_content.html", context)
            response.headers["HX-Trigger"] = json.dumps({
                "showNotification": {
                    "message": result['message'],
                    "type": "success"
                },
                "closeModal": True
            })
            return response
        else:
            # Return error in modal
            return HTMLResponse(
                f'<div class="error" style="color: var(--danger-color); padding: 1rem;">{result["message"]}</div>'
            )
            
    except Exception as e:
        logger.error(f"Error adding exception: {str(e)}")
        return HTMLResponse(
            f'<div class="error" style="color: var(--danger-color); padding: 1rem;">Error adding exception: {str(e)}</div>'
        )

@router.delete("/staff/employee/{employee_id}/remove-exception")
async def remove_exception_server_side(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    date: str = Query(...)
):
    """Remove attendance exception server-side"""
    try:
        if StaffAttendanceService is None:
            response = HTMLResponse('<div class="error">StaffAttendanceService not available</div>')
            response.headers["HX-Trigger"] = json.dumps({
                "showNotification": {
                    "message": "Service not available",
                    "type": "error"
                }
            })
            return response
        
        service = StaffAttendanceService(db)
        
        result = service.remove_attendance_exception(employee_id, date)
        
        if result['success']:
            # Return updated attendance content
            parsed_date = datetime.strptime(date, '%Y-%m-%d').date()
            # Get current date range
            start_date_obj, end_date_obj = service.get_current_month_dates()
            
            attendance_data = service.get_attendance_data(employee_id, start_date_obj, end_date_obj)
            
            context = {
                "request": request,
                **attendance_data
            }
            
            response = templates.TemplateResponse("staff/attendance/fragments/attendance_content.html", context)
            response.headers["HX-Trigger"] = json.dumps({
                "showNotification": {
                    "message": result['message'],
                    "type": "success"
                }
            })
            return response
        else:
            response = HTMLResponse(f'<div class="error">Error: {result["message"]}</div>')
            response.headers["HX-Trigger"] = json.dumps({
                "showNotification": {
                    "message": result['message'],
                    "type": "error"
                }
            })
            return response
            
    except Exception as e:
        logger.error(f"Error removing exception: {str(e)}")
        response = HTMLResponse(f'<div class="error">Error removing exception: {str(e)}</div>')
        response.headers["HX-Trigger"] = json.dumps({
            "showNotification": {
                "message": f"Error removing exception: {str(e)}",
                "type": "error"
            }
        })
        return response

@router.post("/staff/attendance/{employee_id}/recalculate")
async def recalculate_stats_server_side(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    start_date: str = Form(...),
    end_date: str = Form(...)
):
    """Recalculate attendance statistics server-side"""
    try:
        if StaffAttendanceService is None:
            response = HTMLResponse('<div class="error">StaffAttendanceService not available</div>')
            response.headers["HX-Trigger"] = json.dumps({
                "showNotification": {
                    "message": "Service not available",
                    "type": "error"
                }
            })
            return response
        
        service = StaffAttendanceService(db)
        
        # Parse dates
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        result = await service.recalculate_stats(employee_id, start_date_obj, end_date_obj)
        
        if result['success']:
            # Return updated stats fragment
            attendance_data = service.get_attendance_data(employee_id, start_date_obj, end_date_obj)
            
            context = {
                "request": request,
                "summary": attendance_data['summary'],
                "employee": attendance_data['employee']
            }
            
            response = templates.TemplateResponse("staff/attendance/fragments/stats_cards.html", context)
            response.headers["HX-Trigger"] = json.dumps({
                "showNotification": {
                    "message": result['message'],
                    "type": "success"
                }
            })
            return response
        else:
            response = HTMLResponse(f'<div class="error">Error: {result["message"]}</div>')
            response.headers["HX-Trigger"] = json.dumps({
                "showNotification": {
                    "message": result['message'],
                    "type": "error"
                }
            })
            return response
            
    except Exception as e:
        logger.error(f"Error recalculating stats: {str(e)}")
        response = HTMLResponse(f'<div class="error">Error recalculating stats: {str(e)}</div>')
        response.headers["HX-Trigger"] = json.dumps({
            "showNotification": {
                "message": f"Error recalculating stats: {str(e)}",
                "type": "error"
            }
        })
        return response

# ORIGINAL BIOMETRIC ROUTES CONTINUE HERE...

@router.get("/employee/{employee_id}/debug")
async def debug_employee_attendance(
    employee_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug employee attendance data"""
    try:
        attendance_service = AttendanceService(db)
        debug_info = attendance_service.get_employee_debug_info(employee_id)
        return debug_info
        
    except Exception as e:
        logger.error(f"Error getting debug info for employee {employee_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching debug information"
        )

@router.post("/test-api-connection-detailed")
async def test_biometric_api_connection_detailed(
    current_user: User = Depends(get_current_user)
):
    """Enhanced test of biometric API connection"""
    try:
        logger.info(f"🧪 Testing biometric API connection by user {current_user.username}")
        
        async with BiometricAPIService() as api:
            # Test basic connection
            connection_test = await api.test_connection()
            
            if connection_test['success']:
                # Test with a specific employee if possible
                test_data = await api.fetch_all_recent_data(days_back=1)
                
                return {
                    'success': True,
                    'message': 'Successfully connected to biometric API',
                    'connection_test': connection_test,
                    'sample_records': len(test_data),
                    'test_timestamp': datetime.now().isoformat()
                }
            else:
                return connection_test
        
    except Exception as e:
        logger.error(f"Error testing API connection: {str(e)}")
        return {
            'success': False,
            'message': f'API connection test failed: {str(e)}',
            'api_status': 'Failed'
        }

@router.post("/employee/{employee_id}/sync")
async def sync_employee_attendance(
    employee_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    force_full: bool = Query(False, description="Force full month sync")
):
    """Sync attendance data for specific employee with support for range sync"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        if not employee.biometric_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Employee has no biometric ID assigned"
                }
            )
        
        # Check if this is a range sync request
        sync_type = "incremental"  # default
        start_date = None
        end_date = None
        
        # Try to get JSON body for range sync
        try:
            body = await request.json()
            if body.get('sync_type') == 'range':
                sync_type = "range"
                start_date = datetime.strptime(body['start_date'], '%Y-%m-%d').date()
                end_date = datetime.strptime(body['end_date'], '%Y-%m-%d').date()
                
                # Validate date range
                if start_date > end_date:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "message": "Start date must be before end date"
                        }
                    )
                
                if start_date > date.today():
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "message": "Start date cannot be in the future"
                        }
                    )
                
                logger.info(f"Range sync requested: {start_date} to {end_date}")
            elif body.get('sync_type') == 'full':
                sync_type = "full"
                force_full = True
                
        except Exception:
            # No JSON body or invalid format, proceed with regular sync
            pass
        
        logger.info(f"Starting {sync_type} attendance sync for employee {employee_id}")
        
        # Run sync with appropriate parameters
        attendance_service = AttendanceService(db)
        
        if sync_type == "range":
            result = await attendance_service.sync_employee_attendance_range(
                employee_id=employee_id,
                start_date=start_date,
                end_date=end_date
            )
        else:
            result = await attendance_service.sync_employee_attendance(
                employee_id=employee_id,
                force_full_sync=force_full
            )
        
        logger.info(f"{sync_type.capitalize()} sync completed for employee {employee_id}: {result}")
        
        # ✅ NEW: Get updated last sync info after successful sync
        if result.get("success"):
            try:
                updated_last_sync = db.query(AttendanceSyncLog).filter(
                    AttendanceSyncLog.employee_id == employee_id
                ).order_by(desc(AttendanceSyncLog.created_at)).first()
                
                if updated_last_sync and updated_last_sync.last_sync_date:
                    result["last_sync_info"] = {
                        "last_sync_date": updated_last_sync.last_sync_date.isoformat(),
                        "sync_status": updated_last_sync.sync_status,
                        "formatted_date": updated_last_sync.last_sync_date.strftime('%B %d, %Y at %I:%M %p'),
                        "records_fetched": updated_last_sync.records_fetched,
                        "records_processed": updated_last_sync.records_processed
                    }
                    logger.info(f"✅ Added last sync info to response: {result['last_sync_info']['formatted_date']}")
                else:
                    logger.warning(f"⚠️ No last sync record found for employee {employee_id}")
                    
            except Exception as sync_info_error:
                logger.error(f"Error fetching updated sync info: {sync_info_error}")
                # Don't fail the whole request if we can't get sync info
                pass
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing attendance for employee {employee_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Sync failed: {str(e)}"
            }
        )

@router.post("/process-attendance-badges")
async def process_attendance_badges(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    start_date: Optional[str] = Query("2024-01-10", description="Start date for badge processing")
):
    """
    Process attendance badges for all employees
    """
    try:
        # Parse start date
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError:
            start_date_obj = date(2024, 1, 10)
        
        logger.info(f"🏆 Starting attendance badge processing by user {current_user.username}")
        
        badge_service = AttendanceBadgeService(db)
        results = await badge_service.process_all_attendance_badges(start_date_obj)
        
        return JSONResponse({
            "success": True,
            "message": "Attendance badge processing completed successfully",
            "results": results,
            "start_date": start_date_obj.isoformat(),
            "processed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing attendance badges: {str(e)}")
        return JSONResponse({
            "success": False,
            "message": f"Badge processing failed: {str(e)}"
        }, status_code=500)

@router.post("/employee/{employee_id}/process-attendance-badges")
async def process_employee_attendance_badges(
    employee_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    start_date: Optional[str] = Query("2024-01-10", description="Start date for badge processing")
):
    """
    Process attendance badges for a specific employee
    """
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Parse start date
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError:
            start_date_obj = date(2024, 1, 10)
        
        logger.info(f"🏆 Processing attendance badges for employee {employee_id} by user {current_user.username}")
        
        badge_service = AttendanceBadgeService(db)
        results = await badge_service.process_employee_attendance_badges(employee_id, start_date_obj)
        
        return JSONResponse({
            "success": True,
            "message": f"Attendance badges processed for {employee.first_name} {employee.last_name}",
            "employee_id": employee_id,
            "results": results,
            "start_date": start_date_obj.isoformat(),
            "processed_at": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing attendance badges for employee {employee_id}: {str(e)}")
        return JSONResponse({
            "success": False,
            "message": f"Badge processing failed: {str(e)}"
        }, status_code=500)

@router.get("/attendance-badge-report")
async def attendance_badge_report(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Show attendance badge report page
    """
    try:
        # Import badge models here to avoid circular imports
        from models.badge import Badge, EmployeeBadge, BadgeCategory
        
        # Get attendance badge statistics
        total_employees = db.query(Employee).filter(
            Employee.biometric_id.isnot(None),
            Employee.biometric_id != "").count()
       
       # Get badge counts
        perfect_month_badge = db.query(Badge).filter(Badge.name == "Perfect Month").first()
        punctuality_badge = db.query(Badge).filter(Badge.name == "Punctuality Champion").first()
        superstar_badge = db.query(Badge).filter(Badge.name == "Attendance Superstar").first()
        
        badge_stats = {}
        
        if perfect_month_badge:
            badge_stats['perfect_month'] = db.query(EmployeeBadge).filter(
                EmployeeBadge.badge_id == perfect_month_badge.id
            ).count()
        
        if punctuality_badge:
            badge_stats['punctuality_champion'] = db.query(EmployeeBadge).filter(
                EmployeeBadge.badge_id == punctuality_badge.id
            ).count()
        
        if superstar_badge:
            badge_stats['attendance_superstar'] = db.query(EmployeeBadge).filter(
                EmployeeBadge.badge_id == superstar_badge.id
            ).count()
        
        # Get recent badge awards
        recent_badges = db.query(EmployeeBadge).join(Badge).join(Employee).filter(
            Badge.category == BadgeCategory.ATTENDANCE
        ).order_by(desc(EmployeeBadge.earned_date)).limit(20).all()
        
        context = {
            "request": request,
            "user": current_user,
            "total_employees": total_employees,
            "badge_stats": badge_stats,
            "recent_badges": recent_badges,
            "page_title": "Attendance Badge Report"
        }
        
        return templates.TemplateResponse("biometric/attendance_badge_report.html", context)
        
    except Exception as e:
        logger.error(f"Error loading badge report: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading badge report"
        )

@router.get("/api/employee/{employee_id}/attendance")
async def get_employee_attendance_api(
   employee_id: int,
   start_date: date = Query(...),
   end_date: date = Query(...),
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """API endpoint to get employee attendance data"""
   try:
       employee = get_employee_by_id(db, employee_id)
       
       attendance_service = AttendanceService(db)
       data = attendance_service.get_employee_attendance_summary(
           employee_id=employee_id,
           start_date=start_date,
           end_date=end_date
       )
       
       # Format for API response
       records = []
       for record in data['records']:
           records.append({
               'date': record.date.isoformat(),
               'check_in_time': record.check_in_time.isoformat() if record.check_in_time else None,
               'check_out_time': record.check_out_time.isoformat() if record.check_out_time else None,
               'expected_start_time': record.expected_start_time.strftime('%H:%M') if record.expected_start_time else None,
               'expected_end_time': record.expected_end_time.strftime('%H:%M') if record.expected_end_time else None,
               'total_working_hours': record.total_working_hours,
               'is_present': record.is_present,
               'is_late': record.is_late,
               'is_early_departure': record.is_early_departure,
               'late_minutes': record.late_minutes,
               'status': record.status,
               'shift_name': record.shift.name if record.shift else None
           })
       
       return {
           'employee': {
               'id': employee.id,
               'name': f"{employee.first_name} {employee.last_name}",
               'biometric_id': employee.biometric_id,
               'employee_id': employee.employee_id
           },
           'period': {
               'start_date': start_date.isoformat(),
               'end_date': end_date.isoformat()
           },
           'records': records,
           'summary': data['summary']
       }
       
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error in attendance API for employee {employee_id}: {str(e)}")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Error fetching attendance data"
       )

@router.get("/api/employee/{employee_id}/sync-status")
async def get_sync_status(
   employee_id: int,
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Get last sync status for employee"""
   try:
       employee = get_employee_by_id(db, employee_id)
       
       last_sync = db.query(AttendanceSyncLog).filter(
           AttendanceSyncLog.employee_id == employee_id
       ).order_by(desc(AttendanceSyncLog.created_at)).first()
       
       if not last_sync:
           return {
               'has_sync': False,
               'message': 'No sync history found'
           }
       
       return {
           'has_sync': True,
           'last_sync_date': last_sync.last_sync_date.isoformat(),
           'sync_status': last_sync.sync_status,
           'records_fetched': last_sync.records_fetched,
           'records_processed': last_sync.records_processed,
           'sync_range': {
               'start': last_sync.sync_start_date.isoformat(),
               'end': last_sync.sync_end_date.isoformat()
           },
           'error_message': last_sync.error_message,
           'created_at': last_sync.created_at.isoformat()
       }
       
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error getting sync status for employee {employee_id}: {str(e)}")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Error fetching sync status"
       )

@router.post("/employee/{employee_id}/recalculate")
async def recalculate_attendance_stats(
   employee_id: int,
   request: Request,
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Recalculate attendance statistics from existing database records"""
   try:
       # Get request data
       data = await request.json()
       start_month = data.get('start_month')
       end_month = data.get('end_month')
       year = data.get('year')
       
       logger.info(f"🧮 Recalculating stats for employee {employee_id} by user {current_user.username}")
       logger.info(f"📅 Filter parameters: {start_month}/{end_month}/{year}")
       
       # Validate employee exists
       employee = get_employee_by_id(db, employee_id)
       
       # Calculate date range
       if start_month and end_month and year:
           start_date = date(int(year), int(start_month), 1)
           # Get last day of end month
           if int(end_month) == 12:
               end_date = date(int(year) + 1, 1, 1) - timedelta(days=1)
           else:
               end_date = date(int(year), int(end_month) + 1, 1) - timedelta(days=1)
       else:
           # Default to current month
           today = date.today()
           start_date = date(today.year, today.month, 1)
           end_date = today
       
       logger.info(f"📊 Recalculating for date range: {start_date} to {end_date}")
       
       # Use AttendanceService to recalculate
       attendance_service = AttendanceService(db)
       result = await attendance_service.recalculate_employee_stats(
           employee_id=employee_id,
           start_date=start_date,
           end_date=end_date
       )
       
       logger.info(f"✅ Stats recalculated successfully for employee {employee_id}")
       
       return JSONResponse({
           'success': True,
           'summary': result['summary'],
           'calculation_details': result['calculation_details'],
           'records_analyzed': result['records_analyzed'],
           'message': 'Statistics recalculated successfully',
           'date_range': {
               'start_date': start_date.isoformat(),
               'end_date': end_date.isoformat()
           },
           'recalculated_at': datetime.now().isoformat(),
           'recalculated_by': current_user.username
       })
       
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"💥 Error recalculating stats for employee {employee_id}: {str(e)}")
       logger.error(traceback.format_exc())
       
       return JSONResponse({
           'success': False,
           'message': f'Recalculation failed: {str(e)}',
           'error_type': type(e).__name__
       }, status_code=500)

@router.post("/test-api-connection")
async def test_biometric_api_connection(
   current_user: User = Depends(get_current_user)
):
   """Test connection to biometric API (READ-ONLY)"""
   try:
       logger.info(f"Testing biometric API connection by user {current_user.username}")
       
       async with BiometricAPIService() as api:
           # Test authentication
           if not api.token:
               return {
                   'success': False,
                   'message': 'Failed to authenticate with biometric API'
               }
           
           # Test basic data fetch (READ-ONLY)
           test_data = await api.fetch_all_recent_data(days_back=1)
           
           return {
               'success': True,
               'message': 'Successfully connected to biometric API',
               'api_status': 'Connected',
               'sample_records': len(test_data),
               'test_timestamp': datetime.now().isoformat()
           }
       
   except Exception as e:
       logger.error(f"Error testing API connection: {str(e)}")
       return {
           'success': False,
           'message': f'API connection failed: {str(e)}',
           'api_status': 'Disconnected'
       }

@router.get("/api/dashboard-stats")
async def get_biometric_dashboard_stats(
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Get dashboard statistics for biometric system"""
   try:
       # Get employees with biometric IDs
       employees_with_biometric = db.query(Employee).filter(
           Employee.biometric_id.isnot(None),
           Employee.biometric_id != ""
       ).count()
       
       total_employees = db.query(Employee).count()
       
       # Get today's attendance stats
       today = date.today()
       todays_attendance = db.query(ProcessedAttendance).filter(
           ProcessedAttendance.date == today
       ).count()
       
       present_today = db.query(ProcessedAttendance).filter(
           and_(
               ProcessedAttendance.date == today,
               ProcessedAttendance.is_present == True
           )
       ).count()
       
       late_today = db.query(ProcessedAttendance).filter(
           and_(
               ProcessedAttendance.date == today,
               ProcessedAttendance.is_late == True
           )
       ).count()
       
       # Recent sync stats
       recent_syncs = db.query(AttendanceSyncLog).filter(
           AttendanceSyncLog.created_at >= datetime.now() - timedelta(days=7)
       ).count()
       
       failed_syncs = db.query(AttendanceSyncLog).filter(
           and_(
               AttendanceSyncLog.created_at >= datetime.now() - timedelta(days=7),
               AttendanceSyncLog.sync_status == "failed"
           )
       ).count()
       
       return {
           'employees': {
               'total': total_employees,
               'with_biometric': employees_with_biometric,
               'without_biometric': total_employees - employees_with_biometric
           },
           'today_attendance': {
               'total_records': todays_attendance,
               'present': present_today,
               'late': late_today,
               'absent': employees_with_biometric - present_today
           },
           'sync_stats': {
               'recent_syncs_7_days': recent_syncs,
               'failed_syncs_7_days': failed_syncs,
               'success_rate': round((recent_syncs - failed_syncs) / recent_syncs * 100, 1) if recent_syncs > 0 else 0
           },
           'timestamp': datetime.now().isoformat()
       }
       
   except Exception as e:
       logger.error(f"Error getting dashboard stats: {str(e)}")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Error fetching dashboard statistics"
       )

@router.post("/staff/attendance/{employee_id}/filter")
async def filter_attendance_ajax(
   employee_id: int,
   request: Request,
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Handle AJAX filter requests for attendance data"""
   try:
       data = await request.json()
       start_month = data.get('start_month')
       end_month = data.get('end_month')
       year = data.get('year')
       
       # Calculate date range
       if start_month and end_month and year:
           start_date = date(int(year), int(start_month), 1)
           # Get last day of end month
           if int(end_month) == 12:
               end_date = date(int(year) + 1, 1, 1) - timedelta(days=1)
           else:
               end_date = date(int(year), int(end_month) + 1, 1) - timedelta(days=1)
       else:
           # Default to current month
           today = date.today()
           start_date = date(today.year, today.month, 1)
           end_date = today
       
       # Get filtered attendance summary
       attendance_service = AttendanceService(db)
       attendance_data = attendance_service.get_employee_attendance_summary(
           employee_id=employee_id,
           start_date=start_date,
           end_date=end_date
       )
       
       # Format records for frontend (same as in the main route)
       formatted_records = []
       for record in attendance_data['records']:
           shift_name = None
           if record.shift:
               shift_name = record.shift.name
           
           formatted_record = {
               'date': record.date.isoformat(),
               'check_in_time': record.check_in_time.isoformat() if record.check_in_time else None,
               'check_out_time': record.check_out_time.isoformat() if record.check_out_time else None,
               'expected_start_time': record.expected_start_time.strftime('%H:%M') if record.expected_start_time else None,
               'expected_end_time': record.expected_end_time.strftime('%H:%M') if record.expected_end_time else None,
               'total_working_hours': record.total_working_hours,
               'is_present': record.is_present,
               'is_late': record.is_late,
               'is_early_departure': record.is_early_departure,
               'late_minutes': record.late_minutes,
               'early_departure_minutes': record.early_departure_minutes,
               'status': record.status,
               'shift_name': shift_name,
               'has_exception': has_exception,
               'exception_info': exception_info
           }
           formatted_records.append(formatted_record)
       
       return JSONResponse({
           'success': True,
           'summary': attendance_data['summary'],
           'records': formatted_records,
           'date_range': {
               'start_date': start_date.isoformat(),
               'end_date': end_date.isoformat()
           },
           'reload_table': False
       })
       
   except Exception as e:
       logger.error(f"Filter error: {str(e)}")
       return JSONResponse({
           'success': False, 
           'message': str(e)
       }, status_code=500)
   

@router.get("/reports", response_class=HTMLResponse)
async def attendance_reports_builder(
   request: Request,
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Attendance Report Builder Page"""
   try:
       context = {
           "request": request,
           "user": current_user,
           "page_title": "Attendance Report Builder"
       }
       
       return templates.TemplateResponse("staff/attendance/attendance_reports.html", context)
       
   except Exception as e:
       logger.error(f"Error loading report builder: {str(e)}")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Error loading report builder"
       )

@router.post("/generate-custom-report")
async def generate_custom_report(
   request: Request,
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Generate custom report based on drag-and-drop configuration"""
   try:
       report_config = await request.json()
       sections = report_config.get('sections', [])
       
       if not sections:
           raise HTTPException(
               status_code=http_status.HTTP_400_BAD_REQUEST,
               detail="No report sections configured"
           )
       
       # Parse report configuration
       report_data = await process_report_configuration(sections, db)
       
       # Generate PDF
       pdf_buffer = await generate_custom_pdf_report(report_data, current_user)
       
       # Return PDF response
       return StreamingResponse(
           BytesIO(pdf_buffer),
           media_type="application/pdf",
           headers={
               "Content-Disposition": f"attachment; filename=custom-attendance-report-{date.today().isoformat()}.pdf",
               "Cache-Control": "no-cache",
               "Pragma": "no-cache"
           }
       )
       
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error generating custom report: {str(e)}")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Error generating custom report"
       )

@router.post("/export-custom-report")
async def export_custom_report_excel(
   request: Request,
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Export custom report to Excel"""
   try:
       report_config = await request.json()
       sections = report_config.get('sections', [])
       
       if not sections:
           raise HTTPException(
               status_code=http_status.HTTP_400_BAD_REQUEST,
               detail="No report sections configured"
           )
       
       # Process report configuration
       report_data = await process_report_configuration(sections, db)
       
       # Generate Excel file
       excel_buffer = await generate_excel_report(report_data, current_user)
       
       # Return Excel response
       return StreamingResponse(
           BytesIO(excel_buffer),
           media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
           headers={
               "Content-Disposition": f"attachment; filename=custom-attendance-report-{date.today().isoformat()}.xlsx",
               "Cache-Control": "no-cache",
               "Pragma": "no-cache"
           }
       )
       
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error exporting to Excel: {str(e)}")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Error exporting to Excel"
       )

@router.post("/save-report-template")
async def save_report_template(
   request: Request,
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Save report template"""
   try:
       template_data = await request.json()
       
       name = template_data.get('name')
       description = template_data.get('description', '')
       sections = template_data.get('sections', [])
       
       if not name or not sections:
           raise HTTPException(
               status_code=http_status.HTTP_400_BAD_REQUEST,
               detail="Template name and sections are required"
           )
       
       # Save template to database (you'll need to create ReportTemplate model)
       template = ReportTemplate(
           name=name,
           description=description,
           configuration=json.dumps(sections),
           created_by=current_user.id,
           created_at=datetime.now()
       )
       
       db.add(template)
       db.commit()
       
       return JSONResponse({
           "success": True,
           "message": "Template saved successfully",
           "template_id": template.id
       })
       
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error saving template: {str(e)}")
       return JSONResponse({
           "success": False,
           "message": "Failed to save template"
       }, status_code=500)

@router.get("/report-templates")
async def get_report_templates(
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Get saved report templates"""
   try:
       templates = db.query(ReportTemplate).filter(
           ReportTemplate.created_by == current_user.id
       ).order_by(ReportTemplate.created_at.desc()).all()
       
       return [{
           "id": template.id,
           "name": template.name,
           "description": template.description,
           "configuration": json.loads(template.configuration),
           "created_at": template.created_at.isoformat()
       } for template in templates]
       
   except Exception as e:
       logger.error(f"Error getting templates: {str(e)}")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Error retrieving templates"
       )

@router.get("/api/departments")
async def get_departments_for_reports(
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Get departments for report filtering"""
   try:
       departments = db.query(Department).filter(
           Department.is_active == True
       ).order_by(Department.name).all()
       
       return [{
           "id": dept.id,
           "name": dept.name,
           "code": dept.code,
           "employee_count": len(dept.employees) if dept.employees else 0
       } for dept in departments]
       
   except Exception as e:
       logger.error(f"Error getting departments: {str(e)}")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Error retrieving departments"
       )

# Helper functions for report processing

async def process_report_configuration(sections: List[Dict], db: Session) -> Dict[str, Any]:
   """Process report configuration and fetch data"""
   report_data = {
       'title': 'Custom Attendance Report',
       'generated_at': datetime.now(),
       'sections': []
   }
   
   # Default date range
   start_date = date.today() - timedelta(days=30)
   end_date = date.today()
   
   # Extract date range from sections
   for section in sections:
       if section.get('type') == 'date-range':
           # You would extract date configuration from the section
           pass
   
   for section in sections:
       section_type = section.get('type')
       section_data = await process_section(section_type, section, db, start_date, end_date)
       report_data['sections'].append(section_data)
   
   return report_data

async def process_section(section_type: str, section_config: Dict, db: Session, 
                        start_date: date, end_date: date) -> Dict[str, Any]:
   """Process individual report section"""
   
   if section_type == 'summary-stats':
       return await get_summary_statistics(db, start_date, end_date)
   
   elif section_type == 'late-arrivals':
       threshold = section_config.get('config', {}).get('late_threshold', 15)
       return await get_late_arrivals_data(db, start_date, end_date, threshold)
   
   elif section_type == 'department':
       dept_ids = section_config.get('config', {}).get('departments', [])
       return await get_department_wise_data(db, start_date, end_date, dept_ids)
   
   elif section_type == 'absent-employees':
       return await get_absent_employees_data(db, start_date, end_date)
   
   elif section_type == 'early-departures':
       return await get_early_departures_data(db, start_date, end_date)
   
   elif section_type == 'overtime':
       return await get_overtime_data(db, start_date, end_date)
   
   else:
       return {
           'type': section_type,
           'title': section_type.replace('-', ' ').title(),
           'data': {},
           'message': 'Section type not implemented yet'
       }

async def get_summary_statistics(db: Session, start_date: date, end_date: date) -> Dict[str, Any]:
   """Get summary statistics for the report"""
   
   # Total employees with biometric IDs
   total_employees = db.query(Employee).filter(
       Employee.biometric_id.isnot(None),
       Employee.biometric_id != ""
   ).count()
   
   # Attendance statistics for date range
   attendance_records = db.query(ProcessedAttendance).filter(
       and_(
           ProcessedAttendance.date >= start_date,
           ProcessedAttendance.date <= end_date
       )
   ).all()
   
   present_count = sum(1 for record in attendance_records if record.is_present)
   absent_count = len(attendance_records) - present_count
   late_count = sum(1 for record in attendance_records if record.is_late)
   
   total_hours = sum(record.total_working_hours or 0 for record in attendance_records)
   avg_hours = total_hours / len(attendance_records) if attendance_records else 0
   
   attendance_rate = (present_count / len(attendance_records) * 100) if attendance_records else 0
   
   return {
       'type': 'summary-stats',
       'title': 'Summary Statistics',
       'data': {
           'total_employees': total_employees,
           'present_count': present_count,
           'absent_count': absent_count,
           'late_count': late_count,
           'total_hours': round(total_hours, 2),
           'avg_hours': round(avg_hours, 2),
           'attendance_rate': round(attendance_rate, 2),
           'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
       }
   }

async def get_late_arrivals_data(db: Session, start_date: date, end_date: date, 
                              threshold_minutes: int = 15) -> Dict[str, Any]:
   """Get late arrivals data"""
   
   late_records = db.query(ProcessedAttendance).join(Employee).filter(
       and_(
           ProcessedAttendance.date >= start_date,
           ProcessedAttendance.date <= end_date,
           ProcessedAttendance.is_late == True,
           ProcessedAttendance.late_minutes >= threshold_minutes
       )
   ).order_by(ProcessedAttendance.late_minutes.desc()).all()
   
   late_arrivals = []
   for record in late_records:
       employee = record.employee
       department_name = employee.department.name if employee.department else 'No Department'
       
       late_arrivals.append({
           'employee_id': employee.employee_id,
           'employee_name': f"{employee.first_name} {employee.last_name}",
           'department': department_name,
           'date': record.date.strftime('%Y-%m-%d'),
           'check_in_time': record.check_in_time.strftime('%H:%M') if record.check_in_time else 'N/A',
           'expected_time': record.expected_start_time.strftime('%H:%M') if record.expected_start_time else 'N/A',
           'late_minutes': record.late_minutes or 0
       })
   
   return {
       'type': 'late-arrivals',
       'title': f'Late Arrivals (>{threshold_minutes} minutes)',
       'data': {
           'records': late_arrivals,
           'total_count': len(late_arrivals),
           'threshold_minutes': threshold_minutes
       }
   }

async def get_department_wise_data(db: Session, start_date: date, end_date: date, 
                                dept_ids: List[int] = None) -> Dict[str, Any]:
   """Get department-wise attendance data"""
   
   query = db.query(Department).filter(Department.is_active == True)
   
   if dept_ids:
       query = query.filter(Department.id.in_(dept_ids))
   
   departments = query.all()
   
   dept_data = []
   for dept in departments:
       # Get employees in this department
       dept_employees = db.query(Employee).filter(
           Employee.department_id == dept.id,
           Employee.biometric_id.isnot(None)
       ).all()
       
       if not dept_employees:
           continue
       
       employee_ids = [emp.id for emp in dept_employees]
       
       # Get attendance records for these employees
       attendance_records = db.query(ProcessedAttendance).filter(
           and_(
               ProcessedAttendance.employee_id.in_(employee_ids),
               ProcessedAttendance.date >= start_date,
               ProcessedAttendance.date <= end_date
           )
       ).all()
       
       present_count = sum(1 for record in attendance_records if record.is_present)
       total_records = len(attendance_records)
       attendance_rate = (present_count / total_records * 100) if total_records > 0 else 0
       
       late_count = sum(1 for record in attendance_records if record.is_late)
       avg_hours = sum(record.total_working_hours or 0 for record in attendance_records) / total_records if total_records > 0 else 0
       
       dept_data.append({
           'department_name': dept.name,
           'department_code': dept.code,
           'total_employees': len(dept_employees),
           'total_records': total_records,
           'present_count': present_count,
           'absent_count': total_records - present_count,
           'late_count': late_count,
           'attendance_rate': round(attendance_rate, 2),
           'avg_hours': round(avg_hours, 2)
       })
   
   return {
       'type': 'department',
       'title': 'Department-wise Analysis',
       'data': {
           'departments': dept_data,
           'total_departments': len(dept_data)
       }
   }

async def get_absent_employees_data(db: Session, start_date: date, end_date: date) -> Dict[str, Any]:
   """Get absent employees data"""
   
   absent_records = db.query(ProcessedAttendance).join(Employee).filter(
       and_(
           ProcessedAttendance.date >= start_date,
           ProcessedAttendance.date <= end_date,
           ProcessedAttendance.is_present == False
       )
   ).order_by(ProcessedAttendance.date.desc()).all()
   
   absent_employees = []
   for record in absent_records:
       employee = record.employee
       department_name = employee.department.name if employee.department else 'No Department'
       
       absent_employees.append({
           'employee_id': employee.employee_id,
           'employee_name': f"{employee.first_name} {employee.last_name}",
           'department': department_name,
           'date': record.date.strftime('%Y-%m-%d'),
           'reason': 'Not recorded',  # You can add reason field if available
           'shift': record.shift.name if record.shift else 'No Shift'
       })
   
   return {
       'type': 'absent-employees',
       'title': 'Absent Employees',
       'data': {
           'records': absent_employees,
           'total_count': len(absent_employees)
       }
   }

async def get_early_departures_data(db: Session, start_date: date, end_date: date) -> Dict[str, Any]:
   """Get early departures data"""
   
   early_departure_records = db.query(ProcessedAttendance).join(Employee).filter(
       and_(
           ProcessedAttendance.date >= start_date,
           ProcessedAttendance.date <= end_date,
           ProcessedAttendance.is_early_departure == True
       )
   ).order_by(ProcessedAttendance.early_departure_minutes.desc()).all()
   
   early_departures = []
   for record in early_departure_records:
       employee = record.employee
       department_name = employee.department.name if employee.department else 'No Department'
       
       early_departures.append({
           'employee_id': employee.employee_id,
           'employee_name': f"{employee.first_name} {employee.last_name}",
           'department': department_name,
           'date': record.date.strftime('%Y-%m-%d'),
           'check_out_time': record.check_out_time.strftime('%H:%M') if record.check_out_time else 'N/A',
           'expected_time': record.expected_end_time.strftime('%H:%M') if record.expected_end_time else 'N/A',
           'early_minutes': record.early_departure_minutes or 0
       })
   
   return {
       'type': 'early-departures',
       'title': 'Early Departures',
       'data': {
           'records': early_departures,
           'total_count': len(early_departures)
       }
   }

async def get_overtime_data(db: Session, start_date: date, end_date: date) -> Dict[str, Any]:
   """Get overtime data"""
   
   # Define standard working hours (8 hours)
   standard_hours = 8.0
   
   overtime_records = db.query(ProcessedAttendance).join(Employee).filter(
       and_(
           ProcessedAttendance.date >= start_date,
           ProcessedAttendance.date <= end_date,
           ProcessedAttendance.total_working_hours > standard_hours
       )
   ).order_by(ProcessedAttendance.total_working_hours.desc()).all()
   
   overtime_data = []
   for record in overtime_records:
       employee = record.employee
       department_name = employee.department.name if employee.department else 'No Department'
       
       overtime_hours = (record.total_working_hours or 0) - standard_hours
       
       overtime_data.append({
           'employee_id': employee.employee_id,
           'employee_name': f"{employee.first_name} {employee.last_name}",
           'department': department_name,
           'date': record.date.strftime('%Y-%m-%d'),
           'total_hours': round(record.total_working_hours or 0, 2),
           'overtime_hours': round(overtime_hours, 2),
           'check_in': record.check_in_time.strftime('%H:%M') if record.check_in_time else 'N/A',
           'check_out': record.check_out_time.strftime('%H:%M') if record.check_out_time else 'N/A'
       })
   
   total_overtime = sum(item['overtime_hours'] for item in overtime_data)
   
   return {
       'type': 'overtime',
       'title': 'Overtime Analysis',
       'data': {
           'records': overtime_data,
           'total_count': len(overtime_data),
           'total_overtime_hours': round(total_overtime, 2),
           'standard_hours': standard_hours
       }
   }

async def generate_custom_pdf_report(report_data: Dict[str, Any], current_user: User) -> bytes:
   """Generate PDF report from custom configuration"""
   
   try:
       from weasyprint import HTML
       import io
       
       # Prepare context for template
       context = {
           'report_data': report_data,
           'generated_by': current_user.username,
           'generated_at': datetime.now(),
           'company_name': 'IUEA',
           'logo_url': get_logo_url()
       }
       
       # Create HTML content
       html_content = f"""
       <!DOCTYPE html>
       <html>
       <head>
           <meta charset="utf-8">
           <title>Custom Attendance Report</title>
           <style>
               body {{
                   font-family: Arial, sans-serif;
                   line-height: 1.6;
                   margin: 0;
                   padding: 20px;
               }}
               .header {{
                   text-align: center;
                   border-bottom: 2px solid #6b1e1e;
                   padding-bottom: 20px;
                   margin-bottom: 30px;
               }}
               .logo {{
                   max-height: 60px;
                   margin-bottom: 10px;
               }}
               .report-title {{
                   color: #6b1e1e;
                   font-size: 24px;
                   font-weight: bold;
                   margin: 10px 0;
               }}
               .report-subtitle {{
                   color: #666;
                   font-size: 14px;
               }}
               .section {{
                   margin-bottom: 30px;
                   page-break-inside: avoid;
               }}
               .section-title {{
                   color: #6b1e1e;
                   font-size: 18px;
                   font-weight: bold;
                   border-bottom: 1px solid #ddd;
                   padding-bottom: 5px;
                   margin-bottom: 15px;
               }}
               .stats-grid {{
                   display: grid;
                   grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                   gap: 15px;
                   margin-bottom: 20px;
               }}
               .stat-card {{
                   background: #f8f9fa;
                   padding: 15px;
                   border-radius: 5px;
                   text-align: center;
                   border-left: 4px solid #6b1e1e;
               }}
               .stat-value {{
                   font-size: 24px;
                   font-weight: bold;
                   color: #6b1e1e;
               }}
               .stat-label {{
                   font-size: 12px;
                   color: #666;
                   text-transform: uppercase;
               }}
               table {{
                   width: 100%;
                   border-collapse: collapse;
                   margin-top: 10px;
               }}
               th, td {{
                   padding: 8px;
                   text-align: left;
                   border-bottom: 1px solid #ddd;
               }}
               th {{
                   background-color: #f8f9fa;
                   font-weight: bold;
                   color: #6b1e1e;
               }}
               tr:nth-child(even) {{
                   background-color: #f9f9f9;
               }}
               .footer {{
                   text-align: center;
                   margin-top: 40px;
                   padding-top: 20px;
                   border-top: 1px solid #ddd;
                   color: #666;
                   font-size: 12px;
               }}
           </style>
       </head>
       <body>
           <div class="header">
               {f'<img src="{context["logo_url"]}" class="logo" alt="Logo">' if context["logo_url"] else ''}
               <div class="report-title">{report_data['title']}</div>
               <div class="report-subtitle">
                   Generated on {context['generated_at'].strftime('%Y-%m-%d %H:%M:%S')} by {context['generated_by']}
               </div>
           </div>
       """
       
       # Add sections to HTML
       for section in report_data['sections']:
           html_content += generate_section_html(section)
       
       html_content += f"""
           <div class="footer">
               <p>{context['company_name']} - Attendance Management System</p>
               <p>This report was automatically generated. Please verify data accuracy.</p>
           </div>
       </body>
       </html>
       """
       
       # Generate PDF
       pdf_buffer = io.BytesIO()
       HTML(string=html_content, base_url=".").write_pdf(pdf_buffer)
       pdf_buffer.seek(0)
       
       return pdf_buffer.read()
       
   except Exception as e:
       logger.error(f"Error generating PDF: {str(e)}")
       raise

def generate_section_html(section: Dict[str, Any]) -> str:
   """Generate HTML for a report section"""
   
   section_type = section.get('type')
   section_title = section.get('title', 'Unknown Section')
   section_data = section.get('data', {})
   
   html = f'<div class="section"><div class="section-title">{section_title}</div>'
   
   if section_type == 'summary-stats':
       html += '<div class="stats-grid">'
       stats = section_data
       for key, value in stats.items():
           if key != 'date_range':
               label = key.replace('_', ' ').title()
               html += f'''
                   <div class="stat-card">
                       <div class="stat-value">{value}</div>
                       <div class="stat-label">{label}</div>
                   </div>
               '''
       html += '</div>'
       if 'date_range' in stats:
           html += f'<p><strong>Report Period:</strong> {stats["date_range"]}</p>'
   
   elif section_type in ['late-arrivals', 'absent-employees', 'early-departures', 'overtime']:
       records = section_data.get('records', [])
       total_count = section_data.get('total_count', 0)
       
       html += f'<p><strong>Total Records:</strong> {total_count}</p>'
       
       if records:
           html += '<table><thead><tr>'
           
           # Table headers based on section type
           if section_type == 'late-arrivals':
               html += '<th>Employee</th><th>Department</th><th>Date</th><th>Check-in</th><th>Expected</th><th>Late By (mins)</th>'
           elif section_type == 'absent-employees':
               html += '<th>Employee</th><th>Department</th><th>Date</th><th>Shift</th>'
           elif section_type == 'early-departures':
               html += '<th>Employee</th><th>Department</th><th>Date</th><th>Check-out</th><th>Expected</th><th>Early By (mins)</th>'
           elif section_type == 'overtime':
               html += '<th>Employee</th><th>Department</th><th>Date</th><th>Total Hours</th><th>Overtime Hours</th>'
           
           html += '</tr></thead><tbody>'
           
           # Table rows
           for record in records:
               html += '<tr>'
               
               if section_type == 'late-arrivals':
                   html += f'''
                       <td>{record['employee_name']}</td>
                       <td>{record['department']}</td>
                       <td>{record['date']}</td>
                       <td>{record['check_in_time']}</td>
                       <td>{record['expected_time']}</td>
                       <td>{record['late_minutes']}</td>
                   '''
               elif section_type == 'absent-employees':
                   html += f'''
                       <td>{record['employee_name']}</td>
                       <td>{record['department']}</td>
                       <td>{record['date']}</td>
                       <td>{record['shift']}</td>
                   '''
               elif section_type == 'early-departures':
                   html += f'''
                       <td>{record['employee_name']}</td>
                       <td>{record['department']}</td>
                       <td>{record['date']}</td>
                       <td>{record['check_out_time']}</td>
                       <td>{record['expected_time']}</td>
                       <td>{record['early_minutes']}</td>
                   '''
               elif section_type == 'overtime':
                   html += f'''
                       <td>{record['employee_name']}</td>
                       <td>{record['department']}</td>
                       <td>{record['date']}</td>
                       <td>{record['total_hours']}</td>
                       <td>{record['overtime_hours']}</td>
                   '''
               
               html += '</tr>'
           
           html += '</tbody></table>'
       else:
           html += '<p>No records found for this section.</p>'
   
   elif section_type == 'department':
       departments = section_data.get('departments', [])
       html += f'<p><strong>Total Departments:</strong> {section_data.get("total_departments", 0)}</p>'
       
       if departments:
           html += '''
               <table>
                   <thead>
                       <tr>
                           <th>Department</th>
                           <th>Code</th>
                           <th>Employees</th>
                           <th>Present</th>
                           <th>Absent</th>
                           <th>Late</th>
                           <th>Attendance Rate</th>
                           <th>Avg Hours</th>
                       </tr>
                   </thead>
                   <tbody>
           '''
           
           for dept in departments:
               html += f'''
                   <tr>
                       <td>{dept['department_name']}</td>
                       <td>{dept['department_code']}</td>
                       <td>{dept['total_employees']}</td>
                       <td>{dept['present_count']}</td>
                       <td>{dept['absent_count']}</td>
                       <td>{dept['late_count']}</td>
                       <td>{dept['attendance_rate']}%</td>
                       <td>{dept['avg_hours']}</td>
                   </tr>
               '''
           
           html += '</tbody></table>'
   
   html += '</div>'
   return html

async def generate_excel_report(report_data: Dict[str, Any], current_user: User) -> bytes:
   """Generate Excel report from custom configuration"""
   
   try:
       import xlsxwriter
       
       output = BytesIO()
       workbook = xlsxwriter.Workbook(output, {'in_memory': True})
       
       # Define formats
       header_format = workbook.add_format({
           'bold': True,
           'font_color': 'white',
           'bg_color': '#6b1e1e',
           'border': 1
       })
       
       title_format = workbook.add_format({
           'bold': True,
           'font_size': 16,
           'font_color': '#6b1e1e'
       })
       
       data_format = workbook.add_format({
           'border': 1,
           'align': 'left'
       })
       
       number_format = workbook.add_format({
           'border': 1,
           'num_format': '0.00'
       })
       
       # Create summary worksheet
       summary_sheet = workbook.add_worksheet('Summary')
       summary_sheet.write('A1', report_data['title'], title_format)
       summary_sheet.write('A2', f"Generated on: {report_data['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}")
       summary_sheet.write('A3', f"Generated by: {current_user.username}")
       
       row = 5
       
       # Add each section to appropriate worksheet
       for section in report_data['sections']:
           section_type = section.get('type')
           section_title = section.get('title', 'Unknown Section')
           section_data = section.get('data', {})
           
           if section_type == 'summary-stats':
               summary_sheet.write(f'A{row}', section_title, header_format)
               row += 2
               
               stats = section_data
               for key, value in stats.items():
                   if key != 'date_range':
                       label = key.replace('_', ' ').title()
                       summary_sheet.write(f'A{row}', label)
                       summary_sheet.write(f'B{row}', value, number_format if isinstance(value, (int, float)) else data_format)
                       row += 1
               
               row += 2
           
           elif section_type in ['late-arrivals', 'absent-employees', 'early-departures', 'overtime']:
               # Create separate worksheet for detailed data
               sheet = workbook.add_worksheet(section_title[:30])  # Excel sheet name limit
               
               records = section_data.get('records', [])
               
               if records:
                   sheet.write('A1', section_title, title_format)
                   sheet.write('A2', f"Total Records: {section_data.get('total_count', 0)}")
                   
                   # Headers
                   row = 4
                   col = 0
                   
                   if section_type == 'late-arrivals':
                       headers = ['Employee ID', 'Employee Name', 'Department', 'Date', 'Check-in Time', 'Expected Time', 'Late Minutes']
                   elif section_type == 'absent-employees':
                       headers = ['Employee ID', 'Employee Name', 'Department', 'Date', 'Shift']
                   elif section_type == 'early-departures':
                       headers = ['Employee ID', 'Employee Name', 'Department', 'Date', 'Check-out Time', 'Expected Time', 'Early Minutes']
                   elif section_type == 'overtime':
                       headers = ['Employee ID', 'Employee Name', 'Department', 'Date', 'Total Hours', 'Overtime Hours']
                   
                   for header in headers:
                       sheet.write(row, col, header, header_format)
                       col += 1
                   
                   # Data rows
                   row += 1
                   for record in records:
                       col = 0
                       
                       if section_type == 'late-arrivals':
                           data = [
                               record['employee_id'], record['employee_name'], record['department'],
                               record['date'], record['check_in_time'], record['expected_time'], record['late_minutes']
                           ]
                       elif section_type == 'absent-employees':
                           data = [
                               record['employee_id'], record['employee_name'], record['department'],
                               record['date'], record['shift']
                           ]
                       elif section_type == 'early-departures':
                           data = [
                               record['employee_id'], record['employee_name'], record['department'],
                               record['date'], record['check_out_time'], record['expected_time'], record['early_minutes']
                           ]
                       elif section_type == 'overtime':
                           data = [
                               record['employee_id'], record['employee_name'], record['department'],
                               record['date'], record['total_hours'], record['overtime_hours']
                           ]
                       
                       for value in data:
                           format_to_use = number_format if isinstance(value, (int, float)) else data_format
                           sheet.write(row, col, value, format_to_use)
                           col += 1
                       row += 1
                   
                   # Auto-adjust column widths
                   for i, header in enumerate(headers):
                       sheet.set_column(i, i, max(len(header) + 2, 12))
           
           elif section_type == 'department':
               # Department analysis worksheet
               dept_sheet = workbook.add_worksheet('Department Analysis')
               dept_sheet.write('A1', section_title, title_format)
               
               departments = section_data.get('departments', [])
               
               if departments:
                   headers = ['Department', 'Code', 'Total Employees', 'Present', 'Absent', 'Late', 'Attendance Rate (%)', 'Avg Hours']
                   
                   row = 3
                   for col, header in enumerate(headers):
                       dept_sheet.write(row, col, header, header_format)
                   
                   row += 1
                   for dept in departments:
                       data = [
                           dept['department_name'], dept['department_code'], dept['total_employees'],
                           dept['present_count'], dept['absent_count'], dept['late_count'],
                           dept['attendance_rate'], dept['avg_hours']
                       ]
                       
                       for col, value in enumerate(data):
                           format_to_use = number_format if isinstance(value, (int, float)) else data_format
                           dept_sheet.write(row, col, value, format_to_use)
                       row += 1
                   
                   # Auto-adjust column widths
                   for i, header in enumerate(headers):
                       dept_sheet.set_column(i, i, max(len(header) + 2, 15))
       
       workbook.close()
       output.seek(0)
       
       return output.read()
       
   except Exception as e:
       logger.error(f"Error generating Excel: {str(e)}")
       raise

@router.post("/employee/{employee_id}/report")
async def generate_attendance_report(
   employee_id: int,
   request_data: ReportRequest,  # Accept JSON instead of Form data
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db)
):
   """Generate PDF attendance report for employee with comprehensive error handling"""
   try:
       # Step 1: Validate employee
       logger.info(f"🏗️ Starting report generation for employee {employee_id}")
       employee = get_employee_by_id(db, employee_id)
       
       # Step 2: Parse and validate dates from the request
       try:
           start_date = datetime.strptime(request_data.start_date, '%Y-%m-%d').date()
           end_date = datetime.strptime(request_data.end_date, '%Y-%m-%d').date()
       except ValueError:
           raise HTTPException(
               status_code=http_status.HTTP_400_BAD_REQUEST,
               detail="Invalid date format. Please use YYYY-MM-DD format."
           )
       
       # Validate date range
       if end_date < start_date:
           raise HTTPException(
               status_code=http_status.HTTP_400_BAD_REQUEST,
               detail="End date must be equal to or after start date."
           )
       
       if start_date > date.today():
           raise HTTPException(
               status_code=http_status.HTTP_400_BAD_REQUEST,
               detail="Start date cannot be in the future."
           )
       
       logger.info(f"📅 Report date range: {start_date} to {end_date}")
       
       # Step 3: Get attendance data
       logger.info(f"📊 Fetching attendance data...")
       attendance_service = AttendanceService(db)
       attendance_data = attendance_service.get_employee_attendance_summary(
           employee_id=employee_id,
           start_date=start_date,
           end_date=end_date
       )
       
       logger.info(f"📋 Found {len(attendance_data['records'])} attendance records")
       
       # Step 4: Handle logo path with multiple fallback options
       logo_url = get_logo_url()
       logger.info(f"🖼️ Logo URL: {logo_url or 'No logo found'}")
       
       # Step 5: Format records for template
       formatted_records = []
       for record in attendance_data['records']:
           try:
               shift_name = record.shift.name if record.shift else 'No Shift'
               
               formatted_record = {
                   'date': record.date,
                   'day_name': record.date.strftime('%A'),
                   'check_in_time': record.check_in_time,
                   'check_out_time': record.check_out_time,
                   'expected_start_time': record.expected_start_time,
                   'expected_end_time': record.expected_end_time,
                   'total_working_hours': record.total_working_hours or 0,
                   'is_present': record.is_present,
                   'is_late': record.is_late,
                   'is_early_departure': record.is_early_departure,
                   'late_minutes': record.late_minutes or 0,
                   'early_departure_minutes': record.early_departure_minutes or 0,
                   'status': record.status,
                   'shift_name': shift_name
               }
               formatted_records.append(formatted_record)
               
           except Exception as record_error:
               logger.warning(f"⚠️ Error formatting record for {record.date}: {record_error}")
               continue
       
       logger.info(f"✅ Successfully formatted {len(formatted_records)} records")
       
       # Step 6: Prepare report period description
       start_month_name = start_date.strftime('%B')
       end_month_name = end_date.strftime('%B')
       
       if start_date.year == end_date.year and start_date.month == end_date.month:
           report_period = f"{start_month_name} {start_date.year}"
       else:
           report_period = f"{start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}"
       
       # Step 7: Prepare template context
       context = {
           "employee": employee,
           "records": formatted_records,
           "summary": attendance_data['summary'],
           "start_date": start_date,
           "end_date": end_date,
           "report_period": report_period,
           "generated_at": datetime.now(),
           "generated_by": current_user.username,
           "department_name": employee.department.name if employee.department else 'No Department',
           "company_name": "IUEA",
           "report_title": f"Attendance Report - {employee.first_name} {employee.last_name}",
           "logo_url": logo_url
       }
       
       logger.info(f"📝 Template context prepared for {employee.first_name} {employee.last_name}")
       
       # Step 8: Render HTML template
       try:
           html_content = templates.get_template("reports/attendance_report.html").render(context)
           logger.info("✅ HTML template rendered successfully")
       except Exception as template_error:
           logger.error(f"❌ Template rendering failed: {template_error}")
           raise HTTPException(
               status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
               detail="Error rendering report template"
           )
       
       # Step 9: Generate PDF
       try:
           from weasyprint import HTML
           import io
           
           logger.info("🔄 Generating PDF...")
           pdf_buffer = io.BytesIO()
           
           # Generate PDF with proper configuration
           HTML(string=html_content, base_url=".").write_pdf(
               pdf_buffer,
               stylesheets=None,
               presentational_hints=True
           )
           
           pdf_buffer.seek(0)
           logger.info("✅ PDF generated successfully")
           
       except Exception as pdf_error:
           logger.error(f"❌ PDF generation failed: {pdf_error}")
           raise HTTPException(
               status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
               detail="Error generating PDF report"
           )
       
       # Step 10: Prepare filename and response
       # Clean filename - remove special characters
       safe_first_name = "".join(c for c in employee.first_name if c.isalnum())
       safe_last_name = "".join(c for c in employee.last_name if c.isalnum())
       safe_period = report_period.replace(' ', '_').replace(',', '')
       
       filename = f"Attendance_Report_{safe_first_name}_{safe_last_name}_{safe_period}.pdf"
       
       # Log successful generation
       logger.info(f"🎉 Report generated successfully for employee {employee.employee_id} by user {current_user.username}")
       logger.info(f"📄 Filename: {filename}")
       logger.info(f"📊 Report contains {len(formatted_records)} attendance records")
       
       # Step 11: Return PDF response
       return StreamingResponse(
           io.BytesIO(pdf_buffer.read()),
           media_type="application/pdf",
           headers={
               "Content-Disposition": f"attachment; filename={filename}",
               "Cache-Control": "no-cache",
               "Pragma": "no-cache"
           }
       )
       
   except HTTPException:
       # Re-raise HTTP exceptions
       raise
       
   except Exception as e:
       # Handle unexpected errors
       logger.error(f"💥 Unexpected error generating report for employee {employee_id}: {str(e)}")
       logger.error(f"Error type: {type(e).__name__}")
       logger.error(f"Traceback: {traceback.format_exc()}")
       
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="An unexpected error occurred while generating the report"
       )

def get_logo_url() -> str:
   """Get logo URL with multiple fallback options"""
   import os
   from pathlib import Path
   
   # Try multiple possible logo locations
   logo_paths = [
       Path("static/images/logo.png"),
       Path("static/images/logo.jpg"),
       Path("static/images/logo.jpeg"),
       Path("static/assets/logo.png"),
       Path("static/logo.png"),
       Path("images/logo.png"),
       Path("logo.png")
   ]
   
   for logo_path in logo_paths:
       if logo_path.exists() and logo_path.is_file():
           try:
               # Convert to absolute path for WeasyPrint
               absolute_path = logo_path.resolve()
               logo_url = f"file://{absolute_path}"
               
               # Verify file is readable
               with open(absolute_path, 'rb') as f:
                   # Try to read first few bytes to ensure it's accessible
                   f.read(10)
               
               logger.info(f"✅ Logo found: {absolute_path}")
               return logo_url
               
           except Exception as e:
               logger.warning(f"⚠️ Logo file exists but not readable: {logo_path} - {e}")
               continue
   
   logger.warning("⚠️ No logo file found in any of the expected locations")
   return ""

# ADDITIONAL UTILITY ROUTES

@router.get("/employee/{employee_id}/export-excel")
async def export_employee_attendance_excel(
   employee_id: int,
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db),
   start_date: Optional[str] = Query(None),
   end_date: Optional[str] = Query(None)
):
   """Export employee attendance to Excel"""
   try:
       import xlsxwriter
       
       employee = get_employee_by_id(db, employee_id)
       
       # Parse dates
       if start_date and end_date:
           start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
           end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
       else:
           start_date_obj, end_date_obj = get_current_month_dates()
       
       # Get attendance data
       attendance_service = AttendanceService(db)
       attendance_data = attendance_service.get_employee_attendance_summary(
           employee_id=employee_id,
           start_date=start_date_obj,
           end_date=end_date_obj
       )
       
       # Create Excel file
       output = BytesIO()
       workbook = xlsxwriter.Workbook(output, {'in_memory': True})
       
       # Define formats
       header_format = workbook.add_format({
           'bold': True,
           'font_color': 'white',
           'bg_color': '#6b1e1e',
           'border': 1
       })
       
       title_format = workbook.add_format({
           'bold': True,
           'font_size': 16,
           'font_color': '#6b1e1e'
       })
       
       data_format = workbook.add_format({
           'border': 1
       })
       
       present_format = workbook.add_format({
           'border': 1,
           'bg_color': '#d4edda'
       })
       
       absent_format = workbook.add_format({
           'border': 1,
           'bg_color': '#f8d7da'
       })
       
       late_format = workbook.add_format({
           'border': 1,
           'bg_color': '#fff3cd'
       })
       
       # Create main worksheet
       worksheet = workbook.add_worksheet('Attendance Report')
       
       # Add title and employee info
       worksheet.write('A1', f'Attendance Report - {employee.first_name} {employee.last_name}', title_format)
       worksheet.write('A2', f'Employee ID: {employee.employee_id}')
       worksheet.write('A3', f'Department: {employee.department.name if employee.department else "No Department"}')
       worksheet.write('A4', f'Period: {start_date_obj.strftime("%B %d, %Y")} to {end_date_obj.strftime("%B %d, %Y")}')
       worksheet.write('A5', f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
       
       # Add summary statistics
       summary = attendance_data['summary']
       row = 7
       worksheet.write(f'A{row}', 'Summary Statistics', header_format)
       row += 1
       worksheet.write(f'A{row}', 'Total Days')
       worksheet.write(f'B{row}', summary['total_days'])
       row += 1
       worksheet.write(f'A{row}', 'Present Days')
       worksheet.write(f'B{row}', summary['present_days'])
       row += 1
       worksheet.write(f'A{row}', 'Absent Days')
       worksheet.write(f'B{row}', summary['absent_days'])
       row += 1
       worksheet.write(f'A{row}', 'Late Days')
       worksheet.write(f'B{row}', summary['late_days'])
       row += 1
       worksheet.write(f'A{row}', 'Total Hours')
       worksheet.write(f'B{row}', round(summary['total_hours'], 2))
       row += 1
       worksheet.write(f'A{row}', 'Attendance Rate')
       worksheet.write(f'B{row}', f"{round(summary['attendance_percentage'], 2)}%")
       
       # Add detailed records
       row += 3
       worksheet.write(f'A{row}', 'Detailed Records', header_format)
       row += 1
       
       # Headers for detailed records
       headers = ['Date', 'Day', 'Check In', 'Check Out', 'Expected In', 'Expected Out', 
                 'Hours', 'Status', 'Late Minutes', 'Shift']
       
       for col, header in enumerate(headers):
           worksheet.write(row, col, header, header_format)
       
       row += 1
       
       # Data rows
       for record in attendance_data['records']:
           col = 0
           
           # Determine row format based on status
           if record.is_present:
               if record.is_late:
                   row_format = late_format
               else:
                   row_format = present_format
           else:
               row_format = absent_format
           
           worksheet.write(row, col, record.date.strftime('%Y-%m-%d'), row_format)
           col += 1
           worksheet.write(row, col, record.date.strftime('%A'), row_format)
           col += 1
           worksheet.write(row, col, record.check_in_time.strftime('%H:%M') if record.check_in_time else 'N/A', row_format)
           col += 1
           worksheet.write(row, col, record.check_out_time.strftime('%H:%M') if record.check_out_time else 'N/A', row_format)
           col += 1
           worksheet.write(row, col, record.expected_start_time.strftime('%H:%M') if record.expected_start_time else 'N/A', row_format)
           col += 1
           worksheet.write(row, col, record.expected_end_time.strftime('%H:%M') if record.expected_end_time else 'N/A', row_format)
           col += 1
           worksheet.write(row, col, round(record.total_working_hours or 0, 2), row_format)
           col += 1
           worksheet.write(row, col, record.status, row_format)
           col += 1
           worksheet.write(row, col, record.late_minutes or 0, row_format)
           col += 1
           worksheet.write(row, col, record.shift.name if record.shift else 'No Shift', row_format)
           
           row += 1
       
       # Auto-adjust column widths
       for i, header in enumerate(headers):
           worksheet.set_column(i, i, max(len(header) + 2, 12))
       
       workbook.close()
       output.seek(0)
       
       # Prepare filename
       safe_first_name = "".join(c for c in employee.first_name if c.isalnum())
       safe_last_name = "".join(c for c in employee.last_name if c.isalnum())
       filename = f"Attendance_{safe_first_name}_{safe_last_name}_{start_date_obj.strftime('%Y%m%d')}_{end_date_obj.strftime('%Y%m%d')}.xlsx"
       
       return StreamingResponse(
           BytesIO(output.read()),
           media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
           headers={
               "Content-Disposition": f"attachment; filename={filename}",
               "Cache-Control": "no-cache",
               "Pragma": "no-cache"
           }
       )
       
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error exporting Excel for employee {employee_id}: {str(e)}")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Error generating Excel export"
       )

@router.get("/sync-all-employees")
async def sync_all_employees_attendance(
   background_tasks: BackgroundTasks,
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db),
   force_full: bool = Query(False, description="Force full sync for all employees")
):
   """Sync attendance for all employees with biometric IDs"""
   try:
       # Get all employees with biometric IDs
       employees = db.query(Employee).filter(
           Employee.biometric_id.isnot(None),
           Employee.biometric_id != ""
       ).all()
       
       if not employees:
           return JSONResponse({
               "success": False,
               "message": "No employees found with biometric IDs"
           })
       
       logger.info(f"🔄 Starting bulk sync for {len(employees)} employees by user {current_user.username}")
       
       # Initialize results tracking
       results = {
           "total_employees": len(employees),
           "successful_syncs": 0,
           "failed_syncs": 0,
           "employees_processed": [],
           "errors": []
       }
       
       attendance_service = AttendanceService(db)
       
       # Process each employee
       for employee in employees:
           try:
               logger.info(f"🔄 Syncing employee {employee.id}: {employee.first_name} {employee.last_name}")
               
               # Sync this employee
               sync_result = await attendance_service.sync_employee_attendance(
                   employee_id=employee.id,
                   force_full_sync=force_full
               )
               
               if sync_result.get("success"):
                   results["successful_syncs"] += 1
                   results["employees_processed"].append({
                       "employee_id": employee.id,
                       "name": f"{employee.first_name} {employee.last_name}",
                       "status": "success",
                       "records_fetched": sync_result.get("records_fetched", 0),
                       "records_saved": sync_result.get("records_saved", 0)
                   })
                   logger.info(f"✅ Successfully synced employee {employee.id}")
               else:
                   results["failed_syncs"] += 1
                   error_msg = sync_result.get("message", "Unknown error")
                   results["errors"].append({
                       "employee_id": employee.id,
                       "name": f"{employee.first_name} {employee.last_name}",
                       "error": error_msg
                   })
                   logger.error(f"❌ Failed to sync employee {employee.id}: {error_msg}")
               
           except Exception as e:
               results["failed_syncs"] += 1
               error_msg = str(e)
               results["errors"].append({
                   "employee_id": employee.id,
                   "name": f"{employee.first_name} {employee.last_name}",
                   "error": error_msg
               })
               logger.error(f"❌ Exception syncing employee {employee.id}: {error_msg}")
       
       # Calculate success rate
       success_rate = (results["successful_syncs"] / results["total_employees"] * 100) if results["total_employees"] > 0 else 0
       
       logger.info(f"✅ Bulk sync completed: {results['successful_syncs']}/{results['total_employees']} successful ({success_rate:.1f}%)")
       
       return JSONResponse({
           "success": True,
           "message": f"Bulk sync completed. {results['successful_syncs']}/{results['total_employees']} employees synced successfully.",
           "results": results,
           "success_rate": round(success_rate, 1),
           "sync_type": "full" if force_full else "incremental",
           "completed_at": datetime.now().isoformat()
       })
       
   except Exception as e:
       logger.error(f"Error in bulk sync: {str(e)}")
       return JSONResponse({
           "success": False,
           "message": f"Bulk sync failed: {str(e)}"
       }, status_code=500)

@router.get("/attendance-summary")
async def get_attendance_summary(
   current_user: User = Depends(get_current_user),
   db: Session = Depends(get_db),
   date_filter: Optional[str] = Query("today", description="today, week, month")
):
   """Get attendance summary for dashboard"""
   try:
       today = date.today()
       
       if date_filter == "today":
           start_date = today
           end_date = today
       elif date_filter == "week":
           # Get start of current week (Monday)
           days_since_monday = today.weekday()
           start_date = today - timedelta(days=days_since_monday)
           end_date = today
       elif date_filter == "month":
           start_date = date(today.year, today.month, 1)
           end_date = today
       else:
           start_date = today
           end_date = today
       
       # Get employees with biometric IDs
       employees_with_biometric = db.query(Employee).filter(
           Employee.biometric_id.isnot(None),
           Employee.biometric_id != ""
       ).all()
       
       total_employees = len(employees_with_biometric)
       
       if total_employees == 0:
           return {
               "date_range": f"{start_date.isoformat()} to {end_date.isoformat()}",
               "total_employees": 0,
               "summary": {
                   "present": 0,
                   "absent": 0,
                   "late": 0,
                   "on_time": 0
               },
               "attendance_rate": 0,
               "departments": []
           }
       
       # Get attendance records for the period
       attendance_records = db.query(ProcessedAttendance).filter(
           and_(
               ProcessedAttendance.date >= start_date,
               ProcessedAttendance.date <= end_date,
               ProcessedAttendance.employee_id.in_([emp.id for emp in employees_with_biometric])
           )
       ).all()
       
       # Calculate summary statistics
       present_count = sum(1 for record in attendance_records if record.is_present)
       late_count = sum(1 for record in attendance_records if record.is_late)
       on_time_count = sum(1 for record in attendance_records if record.is_present and not record.is_late)
       
       # For today, absent = employees without records
       if date_filter == "today":
           employees_with_records = {record.employee_id for record in attendance_records}
           absent_count = total_employees - len(employees_with_records)
       else:
           absent_count = len(attendance_records) - present_count
       
       attendance_rate = (present_count / len(attendance_records) * 100) if attendance_records else 0
       
       # Get department-wise breakdown
       departments = db.query(Department).filter(Department.is_active == True).all()
       dept_summary = []
       
       for dept in departments:
           dept_employees = [emp for emp in employees_with_biometric if emp.department_id == dept.id]
           if not dept_employees:
               continue
           
           dept_employee_ids = [emp.id for emp in dept_employees]
           dept_records = [record for record in attendance_records if record.employee_id in dept_employee_ids]
           
           dept_present = sum(1 for record in dept_records if record.is_present)
           dept_late = sum(1 for record in dept_records if record.is_late)
           
           if date_filter == "today":
               dept_employees_with_records = {record.employee_id for record in dept_records}
               dept_absent = len(dept_employees) - len(dept_employees_with_records)
           else:
               dept_absent = len(dept_records) - dept_present
           
           dept_rate = (dept_present / len(dept_records) * 100) if dept_records else 0
           
           dept_summary.append({
               "department_name": dept.name,
               "total_employees": len(dept_employees),
               "present": dept_present,
               "absent": dept_absent,
               "late": dept_late,
               "attendance_rate": round(dept_rate, 1)
           })
       
       return {
           "date_range": f"{start_date.isoformat()} to {end_date.isoformat()}",
           "total_employees": total_employees,
           "summary": {
               "present": present_count,
               "absent": absent_count,
               "late": late_count,
               "on_time": on_time_count
           },
           "attendance_rate": round(attendance_rate, 1),
           "departments": dept_summary
       }
       
   except Exception as e:
       logger.error(f"Error getting attendance summary: {str(e)}")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail="Error fetching attendance summary"
       )

# Health check
@router.get("/health")
async def health_check():
   """Health check endpoint"""
   try:
       return {
           "status": "healthy", 
           "service": "biometric_attendance",
           "timestamp": datetime.utcnow().isoformat(),
           "api_mode": "READ_ONLY",
           "version": "1.0.0"
       }
   except Exception as e:
       logger.error(f"Health check failed: {str(e)}")
       return {
           "status": "unhealthy",
           "error": str(e),
           "timestamp": datetime.utcnow().isoformat()
       }

logger.info("Biometric attendance router initialized successfully")