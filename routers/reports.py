"""
Production-Ready Reports Router
Comprehensive report builder with drag-and-drop functionality
Supports detailed, summary, analytical, and personnel reports
ALL IMPROVEMENTS INCLUDED - PRODUCTION READY
"""

import logging
import json
import io
import traceback
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc, text
from fastapi import APIRouter, Request, Depends, HTTPException, BackgroundTasks
from fastapi import status as http_status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import xlsxwriter
from pathlib import Path
from models.leave import Leave, LeaveStatusEnum, PublicHoliday, LeaveType
from sqlalchemy.orm import joinedload




# Import all required models
from models.database import get_db
from models.employee import (
    User, Employee, Department, 
    StatusEnum, ContractStatusEnum, GenderEnum, 
    MaritalStatusEnum, EmploymentTypeEnum, BloodGroupEnum
)
from models.attendance import ProcessedAttendance, AttendanceSyncLog
from models.report_template import ReportTemplate
from utils.auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])
templates = Jinja2Templates(directory="templates")

# Pydantic models
class ReportComponent(BaseModel):
    id: str
    type: str
    title: str
    config: Dict[str, Any]

class ReportRequest(BaseModel):
    title: str
    description: Optional[str] = ""
    type: str  # detailed, summary, analytical, personnel
    components: List[ReportComponent]
    filters: Dict[str, Any]

class TemplateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    components: List[ReportComponent]
    settings: Dict[str, Any]

# Routes
@router.get("/", response_class=HTMLResponse)
async def report_builder_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Report Builder Page"""
    try:
        context = {
            "request": request,
            "user": current_user,
            "page_title": "Report Builder"
        }
        
        return templates.TemplateResponse("reports/reports.html", context)
        
    except Exception as e:
        logger.error(f"Error loading report builder: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading report builder"
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
            "code": dept.code or dept.name[:3].upper(),
            "employee_count": len([emp for emp in dept.employees if emp and emp.biometric_id]) if dept.employees else 0
        } for dept in departments]
        
    except Exception as e:
        logger.error(f"Error getting departments: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving departments"
        )

@router.get("/api/employees")
async def get_employees_for_reports(
    department_id: Optional[int] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get employees for report filtering"""
    try:
        query = db.query(Employee)
        
        if department_id:
            query = query.filter(Employee.department_id == department_id)
            
        if status and status != 'all':
            try:
                query = query.filter(Employee.status == StatusEnum(status))
            except ValueError:
                logger.warning(f"Invalid status filter: {status}")
        
        employees = query.order_by(Employee.first_name, Employee.last_name).all()
        
        return [{
            "id": emp.id,
            "employee_id": emp.employee_id,
            "name": f"{emp.first_name} {emp.last_name}",
            "department": emp.department.name if emp.department else "No Department",
            "position": emp.position or "No Position"
        } for emp in employees]
        
    except Exception as e:
        logger.error(f"Error getting employees: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving employees"
        )

@router.post("/generate")
async def generate_custom_report(
    report_request: ReportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate custom PDF report based on configuration"""
    try:
        logger.info(f"Generating custom report: {report_request.title} by user {current_user.username}")
        
        # Validate report request
        if not report_request.components:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="No report components specified"
            )
        
        # Process report configuration and get data
        report_data = await process_report_configuration(report_request, db)
        
        # Generate PDF
        pdf_buffer = await generate_pdf_report(report_data, current_user)
        
        # Create filename
        safe_title = "".join(c for c in report_request.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title.replace(' ', '_')}_{date.today().isoformat()}.pdf"
        
        logger.info(f"Successfully generated PDF report: {filename}")
        
        return StreamingResponse(
            io.BytesIO(pdf_buffer),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating custom report: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating custom report"
        )
def generate_section_html_landscape(section: Dict[str, Any]) -> str:
    """Generate HTML for a report section - LANDSCAPE OPTIMIZED"""
    
    section_type = section.get('type')
    section_title = section.get('title', 'Unknown Section')
    section_data = section.get('data', {})
    
    # Handle errors
    if section.get('error'):
        return f'''
            <div class="section">
                <div class="section-title">{section_title}</div>
                <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 15px; color: #721c24;">
                    <strong>Error:</strong> {section['error']}
                </div>
            </div>
        '''
    
    html = f'<div class="section"><div class="section-title">{section_title}</div>'
    
    # Route to specific optimized handlers
    if section_type == 'header':
        return ''  # Skip header sections
    elif section_type == 'summary-stats':
        html += _generate_summary_stats_html_landscape(section_data)
    elif section_type == 'individual-records':
        html += _generate_individual_records_html_landscape(section_data)
    elif section_type == 'late-arrivals':
        html += _generate_late_arrivals_html_landscape(section_data)
    elif section_type == 'absent-employees':
        html += _generate_absent_employees_html_landscape(section_data)
    elif section_type == 'early-departures':
        html += _generate_early_departures_html_landscape(section_data)
    elif section_type == 'personnel-list':
        html += _generate_personnel_list_html_landscape(section_data)
    # 🚨 ADD THESE MISSING HANDLERS:
    elif section_type == 'department-summary':
        html += _generate_department_summary_html_landscape(section_data)
    elif section_type == 'department-comparison':
        html += _generate_department_comparison_html_landscape(section_data)
    elif section_type == 'attendance-overview':
        html += _generate_attendance_overview_html_landscape(section_data)
    elif section_type == 'trends':
        html += _generate_trends_html_landscape(section_data)
    elif section_type == 'performance-metrics':
        html += _generate_performance_metrics_html_landscape(section_data)
    elif section_type == 'overtime':
        html += _generate_overtime_html_landscape(section_data)
    # 🆕 ADD LEAVE/HOLIDAY HANDLERS:
    elif section_type == 'leave-balance':
        html += _generate_leave_balance_html_landscape(section_data)
    elif section_type == 'leave-utilization':
        html += _generate_leave_utilization_html_landscape(section_data)
    elif section_type == 'leave-calendar':
        html += _generate_leave_calendar_html_landscape(section_data)
    elif section_type == 'holiday-impact':
        html += _generate_holiday_impact_html_landscape(section_data)
    elif section_type == 'working-days-analysis':
        html += _generate_working_days_analysis_html_landscape(section_data)
    elif section_type == 'staff-wise-summary':
        html += _generate_staff_wise_summary_html_landscape(section_data)
    else:
        html += f'<div style="background: #fff3cd; padding: 15px; border-radius: 6px;">Section type "{section_type}" not implemented yet.</div>'
    
    html += '</div>'
    return html

@router.post("/export-excel")
async def export_custom_report_excel(
    report_request: ReportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export custom report to Excel"""
    try:
        logger.info(f"Exporting custom report to Excel: {report_request.title} by user {current_user.username}")
        
        # Validate report request
        if not report_request.components:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="No report components specified"
            )
        
        # Process report configuration and get data
        report_data = await process_report_configuration(report_request, db)
        
        # Generate Excel file
        excel_buffer = await generate_excel_report(report_data, current_user)
        
        # Create filename
        safe_title = "".join(c for c in report_request.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title.replace(' ', '_')}_{date.today().isoformat()}.xlsx"
        
        logger.info(f"Successfully generated Excel report: {filename}")
        
        return StreamingResponse(
            io.BytesIO(excel_buffer),
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
        logger.error(f"Error exporting to Excel: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error exporting to Excel"
        )

@router.post("/save-template")
async def save_report_template(
    template_request: TemplateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save report template"""
    try:
        # Validate template request
        if not template_request.name or not template_request.name.strip():
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Template name is required"
            )
        
        if not template_request.components:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Template must have at least one component"
            )
        
        # Check if template name already exists for this user
        existing = db.query(ReportTemplate).filter(
            ReportTemplate.created_by == current_user.id,
            ReportTemplate.name == template_request.name.strip()
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Template name already exists"
            )
        
        # Save template to database
        template = ReportTemplate(
            name=template_request.name.strip(),
            description=template_request.description.strip() if template_request.description else "",
            configuration=json.dumps([comp.dict() for comp in template_request.components]),
            settings=json.dumps(template_request.settings) if template_request.settings else "{}",
            created_by=current_user.id,
            created_at=datetime.now()
        )
        
        db.add(template)
        db.commit()
        
        logger.info(f"Template '{template_request.name}' saved by user {current_user.username}")
        
        return JSONResponse({
            "success": True,
            "message": "Template saved successfully",
            "template_id": template.id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving template: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": "Failed to save template"
        }, status_code=500)

@router.get("/templates")
async def get_report_templates(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get saved report templates"""
    try:
        templates = db.query(ReportTemplate).filter(
            ReportTemplate.created_by == current_user.id
        ).order_by(ReportTemplate.created_at.desc()).all()
        
        result = []
        for template in templates:
            try:
                configuration = json.loads(template.configuration) if template.configuration else []
                settings = json.loads(template.settings) if template.settings else {}
                
                result.append({
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "configuration": configuration,
                    "settings": settings,
                    "created_at": template.created_at.isoformat()
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in template {template.id}: {e}")
                continue
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting templates: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving templates"
        )

@router.delete("/templates/{template_id}")
async def delete_report_template(
    template_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete report template"""
    try:
        template = db.query(ReportTemplate).filter(
            ReportTemplate.id == template_id,
            ReportTemplate.created_by == current_user.id
        ).first()
        
        if not template:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        template_name = template.name
        db.delete(template)
        db.commit()
        
        logger.info(f"Template '{template_name}' deleted by user {current_user.username}")
        
        return JSONResponse({
            "success": True,
            "message": "Template deleted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting template: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": "Failed to delete template"
        }, status_code=500)

# Helper Functions
async def process_report_configuration(report_request: ReportRequest, db: Session) -> Dict[str, Any]:
    """Process report configuration and fetch data"""
    
    try:
        logger.info(f"Processing report configuration for: {report_request.title}")
        logger.info(f"Report type: {report_request.type}")
        logger.info(f"Filters: {report_request.filters}")
        
        # Calculate date range with better error handling
        start_date, end_date = calculate_date_range(report_request.filters.get('dateRange', 'last_month'))
        
        # Apply custom date range if specified
        if report_request.filters.get('dateRange') == 'custom':
            custom_start = report_request.filters.get('startDate')
            custom_end = report_request.filters.get('endDate')
            if custom_start and custom_end:
                try:
                    start_date = datetime.fromisoformat(custom_start).date()
                    end_date = datetime.fromisoformat(custom_end).date()
                    
                    # Validate date range
                    if start_date > end_date:
                        raise ValueError("Start date must be before end date")
                    if start_date > date.today():
                        raise ValueError("Start date cannot be in the future")
                        
                except ValueError as e:
                    logger.warning(f"Invalid custom date range: {e}")
                    # Fall back to default range
                    start_date, end_date = calculate_date_range('last_month')
        
        logger.info(f"Date range: {start_date} to {end_date}")
        
        report_data = {
            'title': report_request.title,
            'description': report_request.description,
            'type': report_request.type,
            'generated_at': datetime.now(),
            'date_range': {
                'start': start_date,
                'end': end_date,
                'formatted': f"{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"
            },
            'filters': report_request.filters,
            'sections': []
        }
        
        # Process each component
        for component in report_request.components:
            try:
                logger.info(f"Processing component: {component.type} - {component.title}")
                section_data = await process_component(component, db, start_date, end_date, report_request.filters)
                report_data['sections'].append(section_data)
                logger.info(f"Successfully processed component: {component.type}")
            except Exception as e:
                logger.error(f"Error processing component {component.type}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Add error section
                report_data['sections'].append({
                    'type': component.type,
                    'title': component.title,
                    'error': str(e),
                    'data': {
                        'error_details': f"Failed to process component: {str(e)}"
                    }
                })
        
        logger.info(f"Completed processing report configuration. Sections: {len(report_data['sections'])}")
        return report_data
        
    except Exception as e:
        logger.error(f"Error processing report configuration: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def calculate_date_range(date_range_type: str) -> tuple:
    """Calculate start and end dates based on range type"""
    today = date.today()
    
    try:
        logger.info(f"Calculating date range for: {date_range_type}")
        
        if date_range_type == 'this_week':
            start_date = today - timedelta(days=today.weekday())
            end_date = start_date + timedelta(days=6)
        elif date_range_type == 'this_month' or date_range_type == 'current_month':
            start_date = date(today.year, today.month, 1)
            end_date = today
        elif date_range_type == 'last_month':
            if today.month == 1:
                start_date = date(today.year - 1, 12, 1)
                end_date = date(today.year, 1, 1) - timedelta(days=1)
            else:
                start_date = date(today.year, today.month - 1, 1)
                end_date = date(today.year, today.month, 1) - timedelta(days=1)
        elif date_range_type == 'last_3_months':
            start_date = today - timedelta(days=90)
            end_date = today
        elif date_range_type == 'last_6_months':
            start_date = today - timedelta(days=180)
            end_date = today
        elif date_range_type == 'current_year':
            start_date = date(today.year, 1, 1)
            end_date = today
        else:  # default to last month
            if today.month == 1:
                start_date = date(today.year - 1, 12, 1)
                end_date = date(today.year, 1, 1) - timedelta(days=1)
            else:
                start_date = date(today.year, today.month - 1, 1)
                end_date = date(today.year, today.month, 1) - timedelta(days=1)
        
        # Ensure end_date is not in the future
        end_date = min(end_date, today)
        
        logger.info(f"Calculated date range: {start_date} to {end_date}")
        return start_date, end_date
        
    except Exception as e:
        logger.error(f"Error calculating date range for {date_range_type}: {str(e)}")
        # Return safe default
        start_date = date(today.year, today.month, 1)
        return start_date, today

def calculate_shift_hours(start_time, end_time) -> float:
    """Calculate working hours from shift start and end times"""
    try:
        from datetime import datetime, timedelta
        
        # Convert to datetime objects for calculation
        start_dt = datetime.combine(datetime.today().date(), start_time)
        end_dt = datetime.combine(datetime.today().date(), end_time)
        
        # Handle overnight shifts
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        
        # Calculate hours (no break deduction as requested)
        duration = end_dt - start_dt
        hours = duration.total_seconds() / 3600
        
        return round(hours, 2)
        
    except Exception as e:
        logger.warning(f"Error calculating shift hours: {e}")
        return 8.0  # Default

def weekday_name(weekday_num: int) -> str:
    """Convert weekday number to name"""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return days[weekday_num] if 0 <= weekday_num <= 6 else 'Unknown'

def calculate_annual_leave_balance(employee: Employee, db: Session, as_of_date: date) -> float:
    """Calculate annual leave balance - YOU NEED TO IMPLEMENT"""
    # Based on your company policy
    # Usually: (Months worked / 12) * Annual entitlement - Leave taken
    pass

def calculate_sick_leave_balance(employee: Employee, db: Session, as_of_date: date) -> float:
    """Calculate sick leave balance - YOU NEED TO IMPLEMENT"""
    pass

def get_leave_taken_in_period(employee_id: int, db: Session, start_date: date, end_date: date) -> Dict:
    """Get leave taken by type in period - YOU NEED TO IMPLEMENT"""
    leave_records = db.query(Leave).filter(
        and_(
            Leave.employee_id == employee_id,
            Leave.start_date <= end_date,
            Leave.end_date >= start_date,
            Leave.status == LeaveStatusEnum.ACTIVE
        )
    ).all()
    
    # Calculate overlapping days and group by type
    pass

async def process_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process individual report component"""
    
    component_type = component.type
    
    try:
        if component_type == 'header':
            return await process_header_component(component, start_date, end_date)
        elif component_type == 'summary-stats':
            return await process_summary_stats_component(component, db, start_date, end_date, filters)
        elif component_type == 'attendance-overview':
            return await process_attendance_overview_component(component, db, start_date, end_date, filters)
        elif component_type == 'late-arrivals':
            return await process_late_arrivals_component(component, db, start_date, end_date, filters)
        elif component_type == 'early-departures':
            return await process_early_departures_component(component, db, start_date, end_date, filters)
        elif component_type == 'absent-employees':
            return await process_absent_employees_component(component, db, start_date, end_date, filters)
        elif component_type == 'overtime':
            return await process_overtime_component(component, db, start_date, end_date, filters)
        elif component_type == 'individual-records':
            return await process_individual_records_component(component, db, start_date, end_date, filters)
        elif component_type == 'department-summary':
            return await process_department_summary_component(component, db, start_date, end_date, filters)
        elif component_type == 'department-comparison':
            return await process_department_comparison_component(component, db, start_date, end_date, filters)
        elif component_type == 'trends':
            return await process_trends_component(component, db, start_date, end_date, filters)
        elif component_type == 'performance-metrics':
            return await process_performance_metrics_component(component, db, start_date, end_date, filters)
        elif component_type == 'personnel-list':
            return await process_personnel_list_component(component, db, start_date, end_date, filters)
        # Add these to your existing component processing:
        elif component_type == 'staff-wise-summary':
            return await process_staff_wise_summary_component(component, db, start_date, end_date, filters)
        elif component_type == 'leave-balance':
            return await process_leave_balance_component(component, db, start_date, end_date, filters)
        elif component_type == 'leave-utilization':
            return await process_leave_utilization_component(component, db, start_date, end_date, filters)
        elif component_type == 'leave-calendar':
            return await process_leave_calendar_component(component, db, start_date, end_date, filters)
        elif component_type == 'holiday-impact':
            return await process_holiday_impact_component(component, db, start_date, end_date, filters)
        elif component_type == 'working-days-analysis':
            return await process_working_days_analysis_component(component, db, start_date, end_date, filters)
        else:
            return {
                'type': component_type,
                'title': component.title,
                'data': {},
                'message': f'Component type "{component_type}" not implemented yet'
            }
    except Exception as e:
        logger.error(f"Error processing component {component_type}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'type': component_type,
            'title': component.title,
            'error': str(e),
            'data': {
                'error_details': f"Component processing failed: {str(e)}"
            }
        }

async def process_header_component(component: ReportComponent, start_date: date, end_date: date) -> Dict[str, Any]:
    """Process header component"""
    return {
        'type': 'header',
        'title': component.title,
        'data': {
            'report_title': component.config.get('title', 'Report'),
            'show_logo': component.config.get('showLogo', True),
            'show_date': component.config.get('showDate', True),
            'date_range': f"{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}",
            'logo_url': get_logo_url()
        }
    }

async def process_summary_stats_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process summary statistics component"""
    
    try:
        logger.info("Processing summary stats component")
        
        # Base query for attendance records
        query = db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date
            )
        )
        
        # Apply department filter with improved logic
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                # Convert to integers, handling both string and int IDs
                dept_ids = []
                for d in dept_filter:
                    if isinstance(d, str) and d.isdigit():
                        dept_ids.append(int(d))
                    elif isinstance(d, int):
                        dept_ids.append(d)
                
                if dept_ids:
                    query = query.join(Employee).filter(Employee.department_id.in_(dept_ids))
                    logger.info(f"Applied department filter: {dept_ids}")
            except Exception as e:
                logger.warning(f"Invalid department filter in summary stats: {dept_filter}, error: {e}")
        
        attendance_records = query.all()
        logger.info(f"Found {len(attendance_records)} attendance records")
        
        # Calculate statistics
        total_records = len(attendance_records)
        present_count = sum(1 for record in attendance_records if record.is_present)
        absent_count = total_records - present_count
        late_count = sum(1 for record in attendance_records if getattr(record, 'is_late', False))
        early_departure_count = sum(1 for record in attendance_records if getattr(record, 'is_early_departure', False))
        
        total_hours = sum(record.total_working_hours or 0 for record in attendance_records)
        avg_hours = total_hours / total_records if total_records > 0 else 0
        
        attendance_rate = (present_count / total_records * 100) if total_records > 0 else 0
        punctuality_rate = ((present_count - late_count) / present_count * 100) if present_count > 0 else 0
        
        # Get unique employees count
        unique_employees = len(set(record.employee_id for record in attendance_records if record.employee_id))
        
        # If no data found, create a meaningful message
        if total_records == 0:
            logger.warning("No attendance records found for the specified criteria")
            return {
                'type': 'summary-stats',
                'title': component.title,
                'data': {
                    'total_records': 0,
                    'unique_employees': 0,
                    'present_count': 0,
                    'absent_count': 0,
                    'late_count': 0,
                    'early_departure_count': 0,
                    'total_hours': 0,
                    'avg_hours': 0,
                    'attendance_rate': 0,
                    'punctuality_rate': 0,
                    'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    'show_charts': component.config.get('showCharts', True),
                    'metrics': component.config.get('metrics', ['total', 'present', 'absent', 'late']),
                    'message': 'No attendance data found for the selected period and criteria. Please check your date range and department filters.'
                }
            }
        
        logger.info(f"Summary stats: Total: {total_records}, Present: {present_count}, Attendance Rate: {attendance_rate:.2f}%")
        
        return {
            'type': 'summary-stats',
            'title': component.title,
            'data': {
                'total_records': total_records,
                'unique_employees': unique_employees,
                'present_count': present_count,
                'absent_count': absent_count,
                'late_count': late_count,
                'early_departure_count': early_departure_count,
                'total_hours': round(total_hours, 2),
                'avg_hours': round(avg_hours, 2),
                'attendance_rate': round(attendance_rate, 2),
                'punctuality_rate': round(punctuality_rate, 2),
                'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'show_charts': component.config.get('showCharts', True),
                'metrics': component.config.get('metrics', ['total', 'present', 'absent', 'late'])
            }
        }
    except Exception as e:
        logger.error(f"Error processing summary stats: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'type': 'summary-stats',
            'title': component.title,
            'error': str(e),
            'data': {
                'message': f'Error processing summary statistics: {str(e)}'
            }
        }


async def process_individual_records_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process individual attendance records - FIXED VERSION"""
    
    try:
        limit = component.config.get('limit', 1000)
        show_all_fields = component.config.get('showAllFields', False)
        
        logger.info(f"=== INDIVIDUAL RECORDS PROCESSING (FIXED) ===")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Limit: {limit}")
        logger.info(f"Filters: {filters}")
        
        # Build the exact same query as summary stats but with eager loading
        base_query = db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date
            )
        )
        
        # First, get count without joins to verify data exists
        total_count_check = base_query.count()
        logger.info(f"Base query count (no joins): {total_count_check}")
        
        # Apply department filter exactly like summary stats
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = []
                for d in dept_filter:
                    if isinstance(d, str) and d.isdigit():
                        dept_ids.append(int(d))
                    elif isinstance(d, int):
                        dept_ids.append(d)
                
                if dept_ids:
                    base_query = base_query.join(Employee).filter(Employee.department_id.in_(dept_ids))
                    filtered_count = base_query.count()
                    logger.info(f"Applied department filter: {dept_ids}")
                    logger.info(f"Filtered query count: {filtered_count}")
            except Exception as e:
                logger.warning(f"Department filter error: {e}")
        
        # Get the total count after filtering
        total_filtered_count = base_query.count()
        logger.info(f"Total filtered count: {total_filtered_count}")
        
        if total_filtered_count == 0:
            return {
                'type': 'individual-records',
                'title': component.title,
                'data': {
                    'records': [],
                    'total_count': 0,
                    'message': f"No attendance records found for the period {start_date} to {end_date} with applied filters",
                    'debug_info': {
                        'total_in_db': db.query(ProcessedAttendance).count(),
                        'date_range': f"{start_date} to {end_date}",
                        'filters_applied': filters,
                        'base_count': total_count_check,
                        'filtered_count': total_filtered_count
                    }
                }
            }
        
        # Apply ordering and limit
        attendance_records = base_query.order_by(
            ProcessedAttendance.date.desc(),
            ProcessedAttendance.id.desc()  # Add secondary sort for consistency
        ).limit(limit).all()
        
        logger.info(f"Retrieved {len(attendance_records)} attendance records after limit")
        
        # Process records with proper error handling
        individual_records = []
        
        for i, attendance_record in enumerate(attendance_records):
            try:
                # Get employee info using a separate query to avoid join issues
                employee = db.query(Employee).filter(Employee.id == attendance_record.employee_id).first()
                
                if employee:
                    employee_name = f"{employee.first_name} {employee.last_name}"
                    department_name = employee.department.name if employee.department else "No Department"
                    position = employee.position or "No Position"
                    employee_id_display = employee.employee_id or f"EMP_{employee.id}"
                else:
                    employee_name = f"Employee ID {attendance_record.employee_id} (Not Found)"
                    department_name = "Unknown Department"
                    position = "Unknown Position"
                    employee_id_display = f"MISSING_{attendance_record.employee_id}"
                    logger.warning(f"Employee not found for attendance record: employee_id={attendance_record.employee_id}")
                
                # Determine status with better logic
                status = 'Absent'
                if attendance_record.is_present:
                    if getattr(attendance_record, 'is_late', False) and getattr(attendance_record, 'is_early_departure', False):
                        status = 'Late & Early Departure'
                    elif getattr(attendance_record, 'is_late', False):
                        status = 'Late'
                    elif getattr(attendance_record, 'is_early_departure', False):
                        status = 'Early Departure'
                    else:
                        status = 'Present'
                
                # Format times safely
                check_in_time = 'N/A'
                check_out_time = 'N/A'
                
                if attendance_record.check_in_time:
                    check_in_time = attendance_record.check_in_time.strftime('%H:%M')
                if attendance_record.check_out_time:
                    check_out_time = attendance_record.check_out_time.strftime('%H:%M')
                
                # Generate comments
                comments = 'On Time'
                if getattr(attendance_record, 'is_late', False):
                    late_mins = getattr(attendance_record, 'late_minutes', 0) or 0
                    comments = f"Late: {late_mins} mins"
                elif not attendance_record.is_present:
                    comments = 'Absent'
                elif getattr(attendance_record, 'is_early_departure', False):
                    early_mins = getattr(attendance_record, 'early_departure_minutes', 0) or 0
                    comments = f"Early departure: {early_mins} mins"
                
                # Create the record
                individual_record = {
                    'employee_id': employee_id_display,
                    'employee_name': employee_name,
                    'department': department_name,
                    'position': position,
                    'date': attendance_record.date.strftime('%Y-%m-%d'),
                    'day_name': attendance_record.date.strftime('%A'),
                    'check_in_time': check_in_time,
                    'check_out_time': check_out_time,
                    'total_hours': round(attendance_record.total_working_hours or 0, 2),
                    'expected_hours': round(calculate_expected_hours(attendance_record) or 8.0, 2),
                    'status': status,
                    'comments': comments,
                    'is_present': attendance_record.is_present,
                    'is_late': getattr(attendance_record, 'is_late', False),
                    'is_early_departure': getattr(attendance_record, 'is_early_departure', False)
                }
                
                individual_records.append(individual_record)
                
                # Log first few records for debugging
                if i < 3:
                    logger.info(f"✅ Record {i+1}: {employee_name} - {attendance_record.date} - {status}")
                
            except Exception as record_error:
                logger.error(f"Error processing attendance record {attendance_record.id}: {str(record_error)}")
                logger.error(f"Record data: employee_id={getattr(attendance_record, 'employee_id', 'N/A')}, date={getattr(attendance_record, 'date', 'N/A')}")
                continue
        
        logger.info(f"✅ SUCCESS: Processed {len(individual_records)} individual records out of {len(attendance_records)} retrieved")
        
        return {
            'type': 'individual-records',
            'title': component.title,
            'data': {
                'records': individual_records,
                'total_count': len(individual_records),
                'total_available': total_filtered_count,
                'message': f"Successfully retrieved {len(individual_records)} attendance records with actual names",
                'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'limit_applied': limit,
                'has_more': total_filtered_count > len(individual_records)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ ERROR in individual records: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'type': 'individual-records',
            'title': component.title,
            'error': str(e),
            'data': {
                'records': [],
                'total_count': 0,
                'message': f"Error processing records: {str(e)}",
                'debug_info': {
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ ERROR in individual records: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'type': 'individual-records',
            'title': component.title,
            'error': str(e),
            'data': {
                'records': [],
                'total_count': 0,
                'message': f"Error processing records: {str(e)}",
                'debug_info': {
                    'error_type': type(e).__name__,
                    'error_details': str(e)
                }
            }
        }
    
# Debug endpoints for testing
@router.get("/debug/individual-records")
async def debug_individual_records(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint for individual records component"""
    try:
        from datetime import date, timedelta
        
        # Use same date range as your report
        end_date = date.today()
        start_date = end_date - timedelta(days=90)  # Last 3 months
        
        # Test the exact query used in individual records
        base_query = db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date
            )
        )
        
        total_count = base_query.count()
        
        # Test with limit
        limited_records = base_query.order_by(
            ProcessedAttendance.date.desc()
        ).limit(100).all()
        
        # Test employee relationships
        employee_test = []
        for record in limited_records[:5]:  # Test first 5
            employee = db.query(Employee).filter(Employee.id == record.employee_id).first()
            employee_test.append({
                'attendance_id': record.id,
                'employee_id': record.employee_id,
                'date': record.date.isoformat(),
                'employee_found': employee is not None,
                'employee_name': f"{employee.first_name} {employee.last_name}" if employee else "Not Found"
            })
        
        return {
            'debug_info': {
                'date_range': f"{start_date} to {end_date}",
                'total_records_in_range': total_count,
                'limited_records_retrieved': len(limited_records),
                'first_5_employee_tests': employee_test,
                'database_total_attendance': db.query(ProcessedAttendance).count(),
                'database_total_employees': db.query(Employee).count()
            }
        }
        
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
@router.get("/debug/query-test")
async def debug_query_test(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Simple query test"""
    try:
        from datetime import date, timedelta
        
        end_date = date.today()
        start_date = end_date - timedelta(days=90)
        
        # Test basic query
        total_attendance = db.query(ProcessedAttendance).count()
        range_attendance = db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date
            )
        ).count()
        
        # Test with limit
        sample_records = db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date
            )
        ).limit(10).all()
        
        return {
            'total_attendance_records': total_attendance,
            'range_attendance_records': range_attendance,
            'sample_records_count': len(sample_records),
            'date_range': f"{start_date} to {end_date}",
            'sample_dates': [r.date.isoformat() for r in sample_records[:5]]
        }
        
    except Exception as e:
        return {'error': str(e)}

async def process_late_arrivals_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process late arrivals component - WITH ACTUAL NAMES"""
    
    try:
        threshold = component.config.get('threshold', 15)
        
        query = db.query(ProcessedAttendance).join(Employee).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date,
                ProcessedAttendance.is_late == True,
                ProcessedAttendance.late_minutes >= threshold
            )
        )
        
        # Apply filters
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = [int(d) for d in dept_filter if str(d).isdigit()]
                if dept_ids:
                    query = query.filter(Employee.department_id.in_(dept_ids))
            except (ValueError, TypeError):
                logger.warning(f"Invalid department filter in late arrivals: {dept_filter}")
        
        late_records = query.order_by(ProcessedAttendance.late_minutes.desc()).all()
        
        # Process records with ACTUAL NAMES
        late_arrivals = []
        for record in late_records:
            try:
                employee = record.employee
                department_name = employee.department.name if employee.department else 'No Department'
                
                late_arrivals.append({
                    'employee_id': employee.employee_id,
                    'employee_name': f"{employee.first_name} {employee.last_name}",  # ACTUAL NAME
                    'department': department_name,  # ACTUAL DEPARTMENT NAME
                    'position': employee.position or 'No Position',
                    'date': record.date.strftime('%Y-%m-%d'),
                    'day_name': record.date.strftime('%A'),
                    'check_in_time': record.check_in_time.strftime('%H:%M') if record.check_in_time else 'N/A',
                    'expected_time': record.expected_start_time.strftime('%H:%M') if record.expected_start_time else 'N/A',
                    'late_minutes': record.late_minutes or 0
                })
            except Exception as e:
                logger.warning(f"Error processing late arrival record: {e}")
                continue
        
        # Group by employee for summary
        employee_summary = {}
        for arrival in late_arrivals:
            emp_id = arrival['employee_id']
            if emp_id not in employee_summary:
                employee_summary[emp_id] = {
                    'employee_name': arrival['employee_name'],  # ACTUAL NAME
                    'department': arrival['department'],        # ACTUAL DEPARTMENT
                    'late_count': 0,
                    'total_late_minutes': 0,
                    'worst_lateness': 0
                }
            
            employee_summary[emp_id]['late_count'] += 1
            employee_summary[emp_id]['total_late_minutes'] += arrival['late_minutes']
            employee_summary[emp_id]['worst_lateness'] = max(
                employee_summary[emp_id]['worst_lateness'], 
                arrival['late_minutes']
            )
        
        return {
            'type': 'late-arrivals',
            'title': component.title,
            'data': {
                'records': late_arrivals,
                'total_count': len(late_arrivals),
                'threshold_minutes': threshold,
                'show_details': component.config.get('showDetails', True),
                'group_by_department': component.config.get('groupByDepartment', False),
                'employee_summary': list(employee_summary.values()),
                'avg_lateness': round(sum(r['late_minutes'] for r in late_arrivals) / len(late_arrivals), 2) if late_arrivals else 0
            }
        }
    except Exception as e:
        logger.error(f"Error processing late arrivals: {str(e)}")
        return {
            'type': 'late-arrivals',
            'title': component.title,
            'error': str(e),
            'data': {}
        }

async def process_early_departures_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process early departures component - WITH ACTUAL NAMES"""
    
    try:
        threshold = component.config.get('threshold', 30)  # minutes
        
        query = db.query(ProcessedAttendance).join(Employee).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date,
                ProcessedAttendance.is_early_departure == True,
                ProcessedAttendance.early_departure_minutes >= threshold
            )
        )
        
        # Apply filters
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = [int(d) for d in dept_filter if str(d).isdigit()]
                if dept_ids:
                    query = query.filter(Employee.department_id.in_(dept_ids))
            except (ValueError, TypeError):
                logger.warning(f"Invalid department filter in early departures: {dept_filter}")
        
        early_records = query.order_by(ProcessedAttendance.early_departure_minutes.desc()).all()
        
        # Process records with ACTUAL NAMES
        early_departures = []
        for record in early_records:
            try:
                employee = record.employee
                department_name = employee.department.name if employee.department else 'No Department'
                
                early_departures.append({
                    'employee_id': employee.employee_id,
                    'employee_name': f"{employee.first_name} {employee.last_name}",  # ACTUAL NAME
                    'department': department_name,  # ACTUAL DEPARTMENT NAME
                    'position': employee.position or 'No Position',
                    'date': record.date.strftime('%Y-%m-%d'),
                    'day_name': record.date.strftime('%A'),
                    'check_out_time': record.check_out_time.strftime('%H:%M') if record.check_out_time else 'N/A',
                    'expected_time': record.expected_end_time.strftime('%H:%M') if record.expected_end_time else 'N/A',
                    'early_minutes': record.early_departure_minutes or 0
                })
            except Exception as e:
                logger.warning(f"Error processing early departure record: {e}")
                continue
        
        return {
            'type': 'early-departures',
            'title': component.title,
            'data': {
                'records': early_departures,
                'total_count': len(early_departures),
                'threshold_minutes': threshold,
                'avg_early_minutes': round(sum(r['early_minutes'] for r in early_departures) / len(early_departures), 2) if early_departures else 0
            }
        }
    except Exception as e:
        logger.error(f"Error processing early departures: {str(e)}")
        return {
            'type': 'early-departures',
            'title': component.title,
            'error': str(e),
            'data': {}
        }

async def process_absent_employees_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process absent employees component - WITH ACTUAL NAMES"""
    
    try:
        query = db.query(ProcessedAttendance).join(Employee).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date,
                ProcessedAttendance.is_present == False
            )
        )
        
        # Apply filters
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = [int(d) for d in dept_filter if str(d).isdigit()]
                if dept_ids:
                    query = query.filter(Employee.department_id.in_(dept_ids))
            except (ValueError, TypeError):
                logger.warning(f"Invalid department filter in absent employees: {dept_filter}")
        
        absent_records = query.order_by(ProcessedAttendance.date.desc()).all()
        
        # Process records with ACTUAL NAMES
        absences = []
        for record in absent_records:
            try:
                employee = record.employee
                department_name = employee.department.name if employee.department else 'No Department'
                
                absences.append({
                    'employee_id': employee.employee_id,
                    'employee_name': f"{employee.first_name} {employee.last_name}",  # ACTUAL NAME
                    'department': department_name,  # ACTUAL DEPARTMENT NAME
                    'position': employee.position or 'No Position',
                    'date': record.date.strftime('%Y-%m-%d'),
                    'day_name': record.date.strftime('%A'),
                    'absence_type': getattr(record, 'absence_reason', 'Unspecified') or 'Unspecified'
                })
            except Exception as e:
                logger.warning(f"Error processing absence record: {e}")
                continue
        
        # Group by employee for summary
        employee_summary = {}
        for absence in absences:
            emp_id = absence['employee_id']
            if emp_id not in employee_summary:
                employee_summary[emp_id] = {
                    'employee_name': absence['employee_name'],  # ACTUAL NAME
                    'department': absence['department'],        # ACTUAL DEPARTMENT
                    'absence_count': 0,
                    'absence_dates': []
                }
            
            employee_summary[emp_id]['absence_count'] += 1
            employee_summary[emp_id]['absence_dates'].append(absence['date'])
        
        return {
            'type': 'absent-employees',
            'title': component.title,
            'data': {
                'records': absences,
                'total_count': len(absences),
                'unique_employees': len(employee_summary),
                'employee_summary': list(employee_summary.values())
            }
        }
    except Exception as e:
        logger.error(f"Error processing absent employees: {str(e)}")
        return {
            'type': 'absent-employees',
            'title': component.title,
            'error': str(e),
            'data': {}
        }

async def process_overtime_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process overtime component - WITH ACTUAL NAMES"""
    
    try:
        threshold = component.config.get('threshold', 1.0)  # hours
        
        query = db.query(ProcessedAttendance).join(Employee).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date,
                ProcessedAttendance.overtime_hours >= threshold
            )
        )
        
        # Apply filters
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = [int(d) for d in dept_filter if str(d).isdigit()]
                if dept_ids:
                    query = query.filter(Employee.department_id.in_(dept_ids))
            except (ValueError, TypeError):
                logger.warning(f"Invalid department filter in overtime: {dept_filter}")
        
        overtime_records = query.order_by(ProcessedAttendance.overtime_hours.desc()).all()
        
        # Process records with ACTUAL NAMES
        overtime_list = []
        for record in overtime_records:
            try:
                employee = record.employee
                department_name = employee.department.name if employee.department else 'No Department'
                
                overtime_list.append({
                    'employee_id': employee.employee_id,
                    'employee_name': f"{employee.first_name} {employee.last_name}",  # ACTUAL NAME
                    'department': department_name,  # ACTUAL DEPARTMENT NAME
                    'position': employee.position or 'No Position',
                    'date': record.date.strftime('%Y-%m-%d'),
                    'day_name': record.date.strftime('%A'),
                    'regular_hours': round(record.expected_working_hours or 0, 2),
                    'total_hours': round(record.total_working_hours or 0, 2),
                    'overtime_hours': round(record.overtime_hours or 0, 2)
                })
            except Exception as e:
                logger.warning(f"Error processing overtime record: {e}")
                continue
        
        # Calculate totals
        total_overtime = sum(r['overtime_hours'] for r in overtime_list)
        avg_overtime = total_overtime / len(overtime_list) if overtime_list else 0
        
        return {
            'type': 'overtime',
            'title': component.title,
            'data': {
                'records': overtime_list,
                'total_count': len(overtime_list),
                'threshold_hours': threshold,
                'total_overtime_hours': round(total_overtime, 2),
                'avg_overtime_hours': round(avg_overtime, 2)
            }
        }
    except Exception as e:
        logger.error(f"Error processing overtime: {str(e)}")
        return {
            'type': 'overtime',
            'title': component.title,
            'error': str(e),
            'data': {}
        }

async def process_personnel_list_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process personnel list component with comprehensive filtering - WITH ACTUAL NAMES"""
    
    try:
        logger.info("Processing personnel list component")
        
        # Get base query for employees
        query = db.query(Employee)
        
        # Apply department filter
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = [int(d) for d in dept_filter if str(d).isdigit()]
                if dept_ids:
                    query = query.filter(Employee.department_id.in_(dept_ids))
                    logger.info(f"Applied department filter: {dept_ids}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid department filter: {dept_filter}, error: {e}")
        
        # Apply personnel criteria from filters
        personnel_criteria = filters.get('personnelCriteria', {})
        hire_date_range = filters.get('hireDateRange', {})
        
        logger.info(f"Personnel criteria: {personnel_criteria}")
        logger.info(f"Hire date range: {hire_date_range}")
        
        # Apply gender filter
        if personnel_criteria.get('gender') and personnel_criteria['gender'] != 'all':
            try:
                gender_value = personnel_criteria['gender'].upper()
                if gender_value in ['MALE', 'FEMALE', 'OTHER']:
                    query = query.filter(Employee.gender == GenderEnum(gender_value.title()))
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid gender filter: {personnel_criteria['gender']}, error: {e}")
        
        # Apply employment status filter
        if personnel_criteria.get('status') and personnel_criteria['status'] != 'all':
            try:
                status_map = {
                    'active': StatusEnum.ACTIVE,
                    'not_active': StatusEnum.NOT_ACTIVE
                }
                if personnel_criteria['status'] in status_map:
                    query = query.filter(Employee.status == status_map[personnel_criteria['status']])
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid status filter: {personnel_criteria['status']}, error: {e}")
        
        # Apply contract status filter
        if personnel_criteria.get('contract') and personnel_criteria['contract'] != 'all':
            try:
                contract_map = {
                    'active': ContractStatusEnum.ACTIVE,
                    'expired': ContractStatusEnum.EXPIRED,
                    'suspended': ContractStatusEnum.SUSPENDED,
                    'canceled': ContractStatusEnum.CANCELED
                }
                if personnel_criteria['contract'] in contract_map:
                    query = query.filter(Employee.contract_status == contract_map[personnel_criteria['contract']])
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid contract filter: {personnel_criteria['contract']}, error: {e}")
        
        # Apply employment type filter
        if personnel_criteria.get('employment_type') and personnel_criteria['employment_type'] != 'all':
            try:
                employment_map = {
                    'full_time': EmploymentTypeEnum.FULL_TIME,
                    'part_time': EmploymentTypeEnum.PART_TIME,
                    'contract': EmploymentTypeEnum.CONTRACT,
                    'internship': EmploymentTypeEnum.INTERNSHIP
                }
                if personnel_criteria['employment_type'] in employment_map:
                    query = query.filter(Employee.employment_type == employment_map[personnel_criteria['employment_type']])
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid employment type filter: {personnel_criteria['employment_type']}, error: {e}")
        
        # Apply marital status filter
        if personnel_criteria.get('marital') and personnel_criteria['marital'] != 'all':
            try:
                marital_map = {
                    'single': MaritalStatusEnum.SINGLE,
                    'married': MaritalStatusEnum.MARRIED,
                    'divorced': MaritalStatusEnum.DIVORCED,
                    'widowed': MaritalStatusEnum.WIDOWED,
                    'separated': MaritalStatusEnum.SEPARATED
                }
                if personnel_criteria['marital'] in marital_map:
                    query = query.filter(Employee.marital_status == marital_map[personnel_criteria['marital']])
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid marital status filter: {personnel_criteria['marital']}, error: {e}")
        
        # Apply age filter
        if personnel_criteria.get('age') and personnel_criteria['age'] != 'all':
            try:
                today = date.today()
                if personnel_criteria['age'] == 'under_30':
                    cutoff_date = date(today.year - 30, today.month, today.day)
                    query = query.filter(Employee.date_of_birth > cutoff_date)
                elif personnel_criteria['age'] == '30_50':
                    start_date_filter = date(today.year - 50, today.month, today.day)
                    end_date_filter = date(today.year - 30, today.month, today.day)
                    query = query.filter(
                        Employee.date_of_birth >= start_date_filter,
                        Employee.date_of_birth <= end_date_filter
                    )
                elif personnel_criteria['age'] == 'over_50':
                    cutoff_date = date(today.year - 50, today.month, today.day)
                    query = query.filter(Employee.date_of_birth < cutoff_date)
            except Exception as e:
                logger.warning(f"Error applying age filter: {e}")
        
        # Apply hire date range filter
        if hire_date_range.get('from'):
            try:
                from_date = datetime.fromisoformat(hire_date_range['from']).date()
                query = query.filter(Employee.start_of_employment >= from_date)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid hire from date: {hire_date_range['from']}, error: {e}")
        
        if hire_date_range.get('to'):
            try:
                to_date = datetime.fromisoformat(hire_date_range['to']).date()
                query = query.filter(Employee.start_of_employment <= to_date)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid hire to date: {hire_date_range['to']}, error: {e}")
        
        # Execute query and get results
        employees = query.order_by(Employee.first_name, Employee.last_name).all()
        
        logger.info(f"Found {len(employees)} employees matching criteria")
        
        # Process employees data with ACTUAL NAMES
        employee_records = []
        for emp in employees:
            try:
                # Calculate age if date of birth is available
                age = None
                if emp.date_of_birth:
                    today = date.today()
                    age = today.year - emp.date_of_birth.year - ((today.month, today.day) < (emp.date_of_birth.month, emp.date_of_birth.day))
                
                # Format employment dates
                hire_date_str = emp.start_of_employment.strftime('%Y-%m-%d') if emp.start_of_employment else 'N/A'
                end_date_str = emp.end_of_employment.strftime('%Y-%m-%d') if emp.end_of_employment else 'Current'
                
                employee_record = {
                    'employee_id': emp.employee_id or 'N/A',
                    'full_name': f"{emp.first_name} {emp.last_name}",  # ACTUAL FULL NAME
                    'first_name': emp.first_name or 'N/A',            # ACTUAL FIRST NAME
                    'last_name': emp.last_name or 'N/A',              # ACTUAL LAST NAME
                    'email': emp.email or 'N/A',
                    'phone': emp.phone or 'N/A',
                    'department': emp.department.name if emp.department else 'No Department',  # ACTUAL DEPARTMENT NAME
                    'department_code': emp.department.code if emp.department else 'N/A',
                    'position': emp.position or 'N/A',                # ACTUAL POSITION
                    'employment_status': emp.status.value if emp.status else 'N/A',
                    'contract_status': emp.contract_status.value if emp.contract_status else 'N/A',
                    'employment_type': emp.employment_type.value if emp.employment_type else 'N/A',
                    'gender': emp.gender.value if emp.gender else 'N/A',
                    'age': age,
                    'date_of_birth': emp.date_of_birth.strftime('%Y-%m-%d') if emp.date_of_birth else 'N/A',
                    'marital_status': emp.marital_status.value if emp.marital_status else 'N/A',
                    'nationality': emp.nationality or 'N/A',
                    'national_id': emp.national_id_number or 'N/A',
                    'passport_number': emp.passport_number or 'N/A',
                    'address': emp.address or 'N/A',
                    'emergency_contact': emp.emergency_contact or 'N/A',
                    'hire_date': hire_date_str,
                    'end_date': end_date_str,
                    'biometric_id': emp.biometric_id or 'N/A',
                    'tin_number': emp.tin_number or 'N/A',
                    'nssf_number': emp.nssf_number or 'N/A',
                    'bank_name': emp.bank_name or 'N/A',
                    'branch_name': emp.branch_name or 'N/A',
                    'account_title': emp.account_title or 'N/A',
                    'account_number': emp.account_number or 'N/A',
                    'religion': emp.religion or 'N/A',
                    'blood_group': emp.blood_group.value if emp.blood_group else 'N/A'
                }
                employee_records.append(employee_record)
                
            except Exception as e:
                logger.error(f"Error processing employee {emp.id}: {str(e)}")
                continue
        
        # Get report type from component config
        report_type = component.config.get('reportType', 'directory')
        
        # Generate summary statistics
        total_employees = len(employee_records)
        
        # Gender breakdown
        gender_stats = {}
        for record in employee_records:
            gender = record['gender']
            gender_stats[gender] = gender_stats.get(gender, 0) + 1
        
        # Department breakdown
        dept_stats = {}
        for record in employee_records:
            dept = record['department']
            dept_stats[dept] = dept_stats.get(dept, 0) + 1
        
        # Employment status breakdown
        status_stats = {}
        for record in employee_records:
            status = record['employment_status']
            status_stats[status] = status_stats.get(status, 0) + 1
        
        # Age statistics
        ages = [record['age'] for record in employee_records if record['age'] is not None]
        avg_age = sum(ages) / len(ages) if ages else 0
        
        logger.info(f"Successfully processed {total_employees} employee records")
        
        return {
            'type': 'personnel-list',
            'title': component.title,
            'data': {
                'report_type': report_type,
                'employees': employee_records,
                'total_count': total_employees,
                'summary_stats': {
                    'total_employees': total_employees,
                    'avg_age': round(avg_age, 1) if avg_age else 0,
                    'gender_breakdown': gender_stats,
                    'department_breakdown': dict(sorted(dept_stats.items())),
                    'status_breakdown': status_stats
                },
                'criteria_applied': personnel_criteria,
                'hire_date_range': hire_date_range
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing personnel list component: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'type': 'personnel-list',
            'title': component.title,
            'error': str(e),
            'data': {
                'employees': [],
                'total_count': 0,
                'summary_stats': {},
                'error_details': f"Failed to process personnel data: {str(e)}"
            }
        }

# Continue with attendance component processing functions...
async def process_attendance_overview_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process attendance overview component"""
    
    try:
        logger.info("Processing attendance overview component")
        
        query = db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date
            )
        )
        
        # Apply department filter
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = [int(d) for d in dept_filter if str(d).isdigit()]
                if dept_ids:
                    query = query.join(Employee).filter(Employee.department_id.in_(dept_ids))
            except (ValueError, TypeError):
                logger.warning(f"Invalid department filter in attendance overview: {dept_filter}")
        
        records = query.all()
        logger.info(f"Found {len(records)} records for attendance overview")
        
        # Calculate daily attendance
        daily_stats = {}
        for record in records:
            date_str = record.date.strftime('%Y-%m-%d')
            if date_str not in daily_stats:
                daily_stats[date_str] = {
                    'date': date_str,
                    'day_name': record.date.strftime('%A'),
                    'present': 0,
                    'absent': 0,
                    'late': 0,
                    'total': 0
                }
            
            daily_stats[date_str]['total'] += 1
            if record.is_present:
                daily_stats[date_str]['present'] += 1
                if getattr(record, 'is_late', False):
                    daily_stats[date_str]['late'] += 1
            else:
                daily_stats[date_str]['absent'] += 1
        
        # Calculate percentages
        for day_data in daily_stats.values():
            if day_data['total'] > 0:
                day_data['attendance_rate'] = round((day_data['present'] / day_data['total']) * 100, 2)
                day_data['punctuality_rate'] = round(((day_data['present'] - day_data['late']) / day_data['present']) * 100, 2) if day_data['present'] > 0 else 0
            else:
                day_data['attendance_rate'] = 0
                day_data['punctuality_rate'] = 0
        
        sorted_daily = sorted(daily_stats.values(), key=lambda x: x['date'])
        
        return {
            'type': 'attendance-overview',
            'title': component.title,
            'data': {
                'daily_stats': sorted_daily,
                'total_days': len(daily_stats),
                'show_trends': component.config.get('showTrends', True),
                'chart_type': component.config.get('chartType', 'line')
            }
        }
    except Exception as e:
        logger.error(f"Error processing attendance overview: {str(e)}")
        return {
            'type': 'attendance-overview',
            'title': component.title,
            'error': str(e),
            'data': {}
        }

async def process_department_summary_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process department summary component"""
    
    try:
        sort_by = component.config.get('sortBy', 'name')
        
        # Get all departments
        dept_query = db.query(Department).filter(Department.is_active == True)
        
        # Apply department filter if specified
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = [int(d) for d in dept_filter if str(d).isdigit()]
                if dept_ids:
                    dept_query = dept_query.filter(Department.id.in_(dept_ids))
            except (ValueError, TypeError):
                logger.warning(f"Invalid department filter in department summary: {dept_filter}")
        
        departments = dept_query.all()
        
        dept_summaries = []
        for dept in departments:
            try:
                # Get attendance records for this department
                attendance_query = db.query(ProcessedAttendance).join(Employee).filter(
                    and_(
                        ProcessedAttendance.date >= start_date,
                        ProcessedAttendance.date <= end_date,
                        Employee.department_id == dept.id
                    )
                )
                
                records = attendance_query.all()
                
                if records:
                    total_records = len(records)
                    present_count = sum(1 for r in records if r.is_present)
                    late_count = sum(1 for r in records if getattr(r, 'is_late', False))
                    absent_count = total_records - present_count
                    total_hours = sum(r.total_working_hours or 0 for r in records)
                    
                    attendance_rate = (present_count / total_records * 100) if total_records > 0 else 0
                    punctuality_rate = ((present_count - late_count) / present_count * 100) if present_count > 0 else 0
                    
                    # Get unique employees in this department
                    unique_employees = len(set(r.employee_id for r in records))
                    
                    dept_summaries.append({
                        'department_name': dept.name,
                        'department_code': dept.code or dept.name[:3].upper(),
                        'total_employees': unique_employees,
                        'total_records': total_records,
                        'present_count': present_count,
                        'absent_count': absent_count,
                        'late_count': late_count,
                        'total_hours': round(total_hours, 2),
                        'avg_hours': round(total_hours / present_count, 2) if present_count > 0 else 0,
                        'attendance_rate': round(attendance_rate, 2),
                        'punctuality_rate': round(punctuality_rate, 2)
                    })
                
            except Exception as e:
                logger.warning(f"Error processing department {dept.name}: {e}")
                continue
        
        # Sort departments
        if sort_by == 'attendance_rate':
            dept_summaries.sort(key=lambda x: x['attendance_rate'], reverse=True)
        elif sort_by == 'punctuality_rate':
            dept_summaries.sort(key=lambda x: x['punctuality_rate'], reverse=True)
        else:  # default to name
            dept_summaries.sort(key=lambda x: x['department_name'])
        
        return {
            'type': 'department-summary',
            'title': component.title,
            'data': {
                'departments': dept_summaries,
                'total_departments': len(dept_summaries),
                'sort_by': sort_by
            }
        }
    except Exception as e:
        logger.error(f"Error processing department summary: {str(e)}")
        return {
            'type': 'department-summary',
            'title': component.title,
            'error': str(e),
            'data': {}
        }

async def process_department_comparison_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process department comparison component"""
    
    try:
        metrics = component.config.get('metrics', ['attendance_rate', 'punctuality_rate'])
        
        # Reuse the department summary logic
        dept_summary_component = ReportComponent(
            id='temp',
            type='department-summary',
            title='Temp',
            config={'sortBy': 'name'}
        )
        
        summary_result = await process_department_summary_component(dept_summary_component, db, start_date, end_date, filters)
        departments = summary_result['data'].get('departments', [])
        
        # Calculate averages for comparison
        if departments:
            avg_attendance = sum(d['attendance_rate'] for d in departments) / len(departments)
            avg_punctuality = sum(d['punctuality_rate'] for d in departments) / len(departments)
            avg_hours = sum(d['avg_hours'] for d in departments) / len(departments)
            
            # Add comparison indicators
            for dept in departments:
                dept['attendance_vs_avg'] = dept['attendance_rate'] - avg_attendance
                dept['punctuality_vs_avg'] = dept['punctuality_rate'] - avg_punctuality
                dept['hours_vs_avg'] = dept['avg_hours'] - avg_hours
        else:
            avg_attendance = avg_punctuality = avg_hours = 0
        
        return {
            'type': 'department-comparison',
            'title': component.title,
            'data': {
                'departments': departments,
                'averages': {
                    'attendance_rate': round(avg_attendance, 2),
                    'punctuality_rate': round(avg_punctuality, 2),
                    'avg_hours': round(avg_hours, 2)
                },
                'metrics': metrics,
                'chart_type': component.config.get('chartType', 'bar')
            }
        }
    except Exception as e:
        logger.error(f"Error processing department comparison: {str(e)}")
        return {
            'type': 'department-comparison',
            'title': component.title,
            'error': str(e),
            'data': {}
        }

async def process_trends_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process trends component"""
    
    try:
        trend_type = component.config.get('trendType', 'daily')  # daily, weekly, monthly
        
        query = db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date
            )
        )
        
        # Apply department filter
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = [int(d) for d in dept_filter if str(d).isdigit()]
                if dept_ids:
                    query = query.join(Employee).filter(Employee.department_id.in_(dept_ids))
            except (ValueError, TypeError):
                logger.warning(f"Invalid department filter in trends: {dept_filter}")
        
        records = query.all()
        
        # Group data by trend type
        trend_data = {}
        
        for record in records:
            if trend_type == 'daily':
                key = record.date.strftime('%Y-%m-%d')
                display_key = record.date.strftime('%m/%d')
            elif trend_type == 'weekly':
                # Get week number
                week_start = record.date - timedelta(days=record.date.weekday())
                key = week_start.strftime('%Y-%m-%d')
                display_key = f"Week of {week_start.strftime('%m/%d')}"
            else:  # monthly
                key = record.date.strftime('%Y-%m')
                display_key = record.date.strftime('%B %Y')
            
            if key not in trend_data:
                trend_data[key] = {
                    'period': display_key,
                    'total': 0,
                    'present': 0,
                    'late': 0,
                    'absent': 0
                }
            
            trend_data[key]['total'] += 1
            if record.is_present:
                trend_data[key]['present'] += 1
                if getattr(record, 'is_late', False):
                    trend_data[key]['late'] += 1
            else:
                trend_data[key]['absent'] += 1
        
        # Calculate rates and sort
        trends = []
        for data in trend_data.values():
            attendance_rate = (data['present'] / data['total'] * 100) if data['total'] > 0 else 0
            punctuality_rate = ((data['present'] - data['late']) / data['present'] * 100) if data['present'] > 0 else 0
            
            trends.append({
                'period': data['period'],
                'total_records': data['total'],
                'present': data['present'],
                'absent': data['absent'],
                'late': data['late'],
                'attendance_rate': round(attendance_rate, 2),
                'punctuality_rate': round(punctuality_rate, 2)
            })
        
        # Sort by period
        trends.sort(key=lambda x: x['period'])
        
        return {
            'type': 'trends',
            'title': component.title,
            'data': {
                'trends': trends,
                'trend_type': trend_type,
                'total_periods': len(trends),
                'chart_type': component.config.get('chartType', 'line')
            }
        }
    except Exception as e:
        logger.error(f"Error processing trends: {str(e)}")
        return {
            'type': 'trends',
            'title': component.title,
            'error': str(e),
            'data': {}
        }

async def process_performance_metrics_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process performance metrics component"""
    
    try:
        metrics = component.config.get('metrics', ['attendance', 'punctuality', 'productivity'])
        
        query = db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date
            )
        )
        
        # Apply department filter
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = [int(d) for d in dept_filter if str(d).isdigit()]
                if dept_ids:
                    query = query.join(Employee).filter(Employee.department_id.in_(dept_ids))
            except (ValueError, TypeError):
                logger.warning(f"Invalid department filter in performance metrics: {dept_filter}")
        
        records = query.all()
        
        if not records:
            return {
                'type': 'performance-metrics',
                'title': component.title,
                'data': {
                    'metrics': {},
                    'message': 'No data available for the selected criteria'
                }
            }
        
        # Calculate performance metrics
        total_records = len(records)
        present_records = [r for r in records if r.is_present]
        present_count = len(present_records)
        late_count = sum(1 for r in records if getattr(r, 'is_late', False))
        
        # Basic metrics
        performance_metrics = {}
        
        if 'attendance' in metrics:
            attendance_rate = (present_count / total_records * 100) if total_records > 0 else 0
            performance_metrics['attendance'] = {
                'name': 'Attendance Rate',
                'value': round(attendance_rate, 2),
                'unit': '%',
                'description': 'Percentage of expected attendance',
                'benchmark': 95.0,
                'status': 'good' if attendance_rate >= 95 else 'warning' if attendance_rate >= 85 else 'poor'
            }
        
        if 'punctuality' in metrics:
            punctuality_rate = ((present_count - late_count) / present_count * 100) if present_count > 0 else 0
            performance_metrics['punctuality'] = {
                'name': 'Punctuality Rate',
                'value': round(punctuality_rate, 2),
                'unit': '%',
                'description': 'Percentage of on-time arrivals',
                'benchmark': 90.0,
                'status': 'good' if punctuality_rate >= 90 else 'warning' if punctuality_rate >= 80 else 'poor'
            }
        
        if 'productivity' in metrics:
            total_hours = sum(r.total_working_hours or 0 for r in present_records)
            expected_hours = sum(r.expected_working_hours or 0 for r in present_records)
            productivity_rate = (total_hours / expected_hours * 100) if expected_hours > 0 else 0
            
            performance_metrics['productivity'] = {
                'name': 'Productivity Rate',
                'value': round(productivity_rate, 2),
                'unit': '%',
                'description': 'Actual vs expected working hours',
                'benchmark': 100.0,
                'status': 'good' if productivity_rate >= 95 else 'warning' if productivity_rate >= 85 else 'poor'
            }
        
        # Additional insights
        insights = []
        if 'attendance' in performance_metrics and performance_metrics['attendance']['value'] < 90:
            insights.append("Attendance rate is below recommended threshold")
        if 'punctuality' in performance_metrics and performance_metrics['punctuality']['value'] < 85:
            insights.append("Punctuality needs improvement")
        if 'productivity' in performance_metrics and performance_metrics['productivity']['value'] > 105:
            insights.append("High productivity - consider workload balance")
        
        return {
            'type': 'performance-metrics',
            'title': component.title,
            'data': {
                'metrics': performance_metrics,
                'insights': insights,
                'period_summary': {
                    'total_records': total_records,
                    'present_count': present_count,
                    'absent_count': total_records - present_count,
                    'late_count': late_count
                }
            }
        }
    except Exception as e:
        logger.error(f"Error processing performance metrics: {str(e)}")
        return {
            'type': 'performance-metrics',
            'title': component.title,
            'error': str(e),
            'data': {}
        }

async def generate_pdf_report(report_data: Dict[str, Any], current_user: User) -> bytes:
    """Generate PDF report from processed data - LANDSCAPE OPTIMIZED"""
    
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        html_content = generate_report_html(report_data, current_user)
        
        # Font configuration for better PDF rendering
        font_config = FontConfiguration()
        
        # CSS for LANDSCAPE PDF styling
        css = CSS(string='''
            @page {
                margin: 1.5cm;
                size: A4 landscape;  /* LANDSCAPE ORIENTATION */
                @bottom-center {
                    content: "Page " counter(page) " of " counter(pages) " | IUEA Attendance Report | " attr(title);
                    font-size: 9px;
                    color: #666;
                }
            }
            body {
                font-family: 'Arial', sans-serif;
                font-size: 10px;
                line-height: 1.3;
                margin: 0;
                padding: 0;
            }
            
            /* Header optimized for landscape */
            .header {
                text-align: center;
                border-bottom: 2px solid #6b1e1e;
                padding-bottom: 15px;
                margin-bottom: 20px;
            }
            .logo {
                max-height: 60px;
                margin-bottom: 10px;
            }
            .report-title {
                color: #6b1e1e;
                font-size: 22px;
                font-weight: bold;
                margin: 10px 0 5px 0;
            }
            .report-meta {
                color: #666;
                font-size: 11px;
                margin: 5px 0;
            }
            
            /* Compact stats grid for landscape */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(6, 1fr);
                gap: 10px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 12px 8px;
                border-radius: 6px;
                text-align: center;
                border-left: 3px solid #6b1e1e;
            }
            .stat-value {
                font-size: 18px;
                font-weight: bold;
                color: #6b1e1e;
                margin-bottom: 3px;
            }
            .stat-label {
                font-size: 9px;
                color: #666;
                text-transform: uppercase;
                font-weight: 500;
            }
            
            /* Landscape-optimized tables */
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 9px;
                background: white;
            }
            th {
                background-color: #6b1e1e;
                color: white;
                font-weight: bold;
                text-transform: uppercase;
                font-size: 8px;
                padding: 8px 4px;
                text-align: left;
                border: 1px solid #ddd;
            }
            td {
                padding: 6px 4px;
                text-align: left;
                border: 1px solid #ddd;
                vertical-align: top;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            
            /* Compact column widths for landscape */
            .col-date { width: 60px; }
            .col-name { width: 120px; font-weight: 500; }
            .col-dept { width: 80px; }
            .col-time { width: 50px; }
            .col-hours { width: 50px; }
            .col-status { width: 80px; }
            .col-comments { width: 100px; }
            .col-late { width: 60px; color: #dc3545; font-weight: bold; }
            
            /* Status styling */
            .status-present { color: #28a745; font-weight: bold; }
            .status-late { color: #ffc107; font-weight: bold; }
            .status-absent { color: #dc3545; font-weight: bold; }
            .status-early { color: #fd7e14; font-weight: bold; }
            
            /* Section titles */
            .section-title {
                color: #6b1e1e;
                font-size: 16px;
                font-weight: bold;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
                margin: 20px 0 15px 0;
                page-break-after: avoid;
            }
            
            /* Summary boxes */
            .summary-box {
                background: #e8f4f8;
                border: 1px solid #bee5eb;
                border-radius: 6px;
                padding: 12px;
                margin: 15px 0;
                font-size: 10px;
            }
            .summary-title {
                color: #0c5460;
                font-weight: bold;
                margin-bottom: 8px;
                font-size: 11px;
            }
            
            /* Footer */
            .footer {
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 9px;
            }
            
            /* Page breaks */
            .page-break { page-break-before: always; }
            .no-break { page-break-inside: avoid; }
            
        ''', font_config=font_config)
        
        # Generate PDF
        pdf_buffer = io.BytesIO()
        HTML(string=html_content, base_url=".").write_pdf(
            pdf_buffer,
            stylesheets=[css],
            font_config=font_config
        )
        pdf_buffer.seek(0)
        
        return pdf_buffer.read()
        
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


async def generate_excel_report(report_data: Dict[str, Any], current_user: User) -> bytes:
    """Generate Excel report from processed data - FIXED VERSION"""
    
    try:
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'font_color': 'white',
            'bg_color': '#6b1e1e',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'font_color': '#6b1e1e'
        })
        
        subtitle_format = workbook.add_format({
            'bold': True,
            'font_size': 12,
            'font_color': '#2c3e50'
        })
        
        data_format = workbook.add_format({
            'border': 1,
            'align': 'left',
            'valign': 'top'
        })
        
        number_format = workbook.add_format({
            'border': 1,
            'num_format': '0.00',
            'align': 'right'
        })
        
        date_format = workbook.add_format({
            'border': 1,
            'num_format': 'yyyy-mm-dd',
            'align': 'center'
        })
        
        # Create summary worksheet
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.write('A1', report_data['title'], title_format)
        summary_sheet.write('A2', f"Generated on: {report_data['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        summary_sheet.write('A3', f"Generated by: {current_user.username}")
        summary_sheet.write('A4', f"Date Range: {report_data['date_range']['formatted']}")
        
        summary_row = 6
        
        # Process each section and create worksheets
        for section in report_data['sections']:
            section_type = section.get('type')
            section_title = section.get('title', 'Unknown Section')
            section_data = section.get('data', {})
            
            logger.info(f"Processing Excel section: {section_type} - {section_title}")
            
            if section.get('error'):
                summary_sheet.write(f'A{summary_row}', f"Error in {section_title}: {section['error']}")
                summary_row += 2
                continue
            
            # Skip header sections
            if section_type == 'header':
                continue
            
            # Process summary statistics
            if section_type == 'summary-stats':
                summary_sheet.write(f'A{summary_row}', 'SUMMARY STATISTICS', subtitle_format)
                summary_row += 1
                
                stats_to_write = [
                    ('Total Records', section_data.get('total_records', 0)),
                    ('Unique Employees', section_data.get('unique_employees', 0)),
                    ('Present Count', section_data.get('present_count', 0)),
                    ('Absent Count', section_data.get('absent_count', 0)),
                    ('Late Count', section_data.get('late_count', 0)),
                    ('Attendance Rate', f"{section_data.get('attendance_rate', 0)}%"),
                    ('Punctuality Rate', f"{section_data.get('punctuality_rate', 0)}%"),
                    ('Average Hours/Day', f"{section_data.get('avg_hours', 0)}h")
                ]
                
                for stat_name, stat_value in stats_to_write:
                    summary_sheet.write(f'A{summary_row}', stat_name)
                    summary_sheet.write(f'B{summary_row}', stat_value)
                    summary_row += 1
                
                summary_row += 2
                continue
            
            # Create separate worksheets for data sections
            if section_type in ['individual-records', 'late-arrivals', 'absent-employees', 'early-departures', 'overtime', 'staff-wise-summary']:
                records = section_data.get('records', []) if section_type != 'staff-wise-summary' else section_data.get('employees', [])
            
            if records:
                # Create safe worksheet name
                safe_title = "".join(c for c in section_title if c.isalnum() or c in (' ', '-', '_'))[:30]
                sheet = workbook.add_worksheet(safe_title)
                
                # Write section header
                sheet.write('A1', section_title, title_format)
                sheet.write('A2', f"Total Records: {section_data.get('total_count', len(records)) if section_type != 'staff-wise-summary' else section_data.get('total_employees', len(records))}")
                
                # Write data based on section type
                if section_type == 'individual-records':
                    write_individual_records_to_excel(sheet, records, header_format, data_format, number_format, date_format)
                elif section_type == 'late-arrivals':
                    write_late_arrivals_to_excel(sheet, records, header_format, data_format, number_format, date_format)
                elif section_type == 'absent-employees':
                    write_absent_employees_to_excel(sheet, records, header_format, data_format, date_format)
                elif section_type == 'early-departures':
                    write_early_departures_to_excel(sheet, records, header_format, data_format, number_format, date_format)
                elif section_type == 'overtime':
                    write_overtime_to_excel(sheet, records, header_format, data_format, number_format, date_format)
                elif section_type == 'staff-wise-summary':
                    write_staff_wise_summary_to_excel(sheet, records, header_format, data_format, number_format)
                
                logger.info(f"Created Excel sheet '{safe_title}' with {len(records)} records")
                
                # Add summary to main sheet
                summary_sheet.write(f'A{summary_row}', f"{section_title}: {len(records)} {'employees' if section_type == 'staff-wise-summary' else 'records'} (see '{safe_title}' sheet)")
                summary_row += 1
            
            # Handle personnel lists
            elif section_type == 'personnel-list':
                employees = section_data.get('employees', [])
                
                if employees:
                    safe_title = "Personnel_List"
                    sheet = workbook.add_worksheet(safe_title)
                    
                    sheet.write('A1', section_title, title_format)
                    sheet.write('A2', f"Total Employees: {section_data.get('total_count', len(employees))}")
                    
                    write_personnel_to_excel(sheet, employees, header_format, data_format)
                    
                    logger.info(f"Created Excel sheet '{safe_title}' with {len(employees)} employees")
                    
                    summary_sheet.write(f'A{summary_row}', f"{section_title}: {len(employees)} employees (see '{safe_title}' sheet)")
                    summary_row += 1
        
        # Auto-adjust column widths for summary sheet
        summary_sheet.set_column('A:A', 25)
        summary_sheet.set_column('B:B', 20)
        
        workbook.close()
        output.seek(0)
        
        logger.info("Excel report generated successfully")
        return output.read()
        
    except Exception as e:
        logger.error(f"Error generating Excel: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def calculate_leave_summary(leave_balances: List[Dict]) -> Dict:
    """Calculate leave summary statistics"""
    if not leave_balances:
        return {}
    
    total_annual = sum(emp.get('annual_leave_balance', 0) for emp in leave_balances)
    total_sick = sum(emp.get('sick_leave_balance', 0) for emp in leave_balances)
    total_taken = sum(emp.get('total_leave_taken', 0) for emp in leave_balances)
    
    return {
        'total_annual_balance': total_annual,
        'total_sick_balance': total_sick,
        'total_leave_taken': total_taken,
        'average_annual_balance': total_annual / len(leave_balances),
        'average_sick_balance': total_sick / len(leave_balances)
    }

def write_individual_records_to_excel(sheet, records, header_format, data_format, number_format, date_format):
    """Write individual attendance records to Excel"""
    
    row = 4  # Start after title and summary
    
    # Headers
    headers = ['Date', 'Employee Name', 'Department', 'Position', 'Check In', 'Check Out', 'Total Hours', 'Expected Hours', 'Status', 'Comments']
    
    for col, header in enumerate(headers):
        sheet.write(row, col, header, header_format)
    
    # Data
    for record in records:
        row += 1
        sheet.write(row, 0, record.get('date', 'N/A'), data_format)
        sheet.write(row, 1, record.get('employee_name', 'N/A'), data_format)
        sheet.write(row, 2, record.get('department', 'N/A'), data_format)
        sheet.write(row, 3, record.get('position', 'N/A'), data_format)
        sheet.write(row, 4, record.get('check_in_time', 'N/A'), data_format)
        sheet.write(row, 5, record.get('check_out_time', 'N/A'), data_format)
        sheet.write(row, 6, record.get('total_hours', 0), number_format)
        sheet.write(row, 7, record.get('expected_hours', 0), number_format)
        sheet.write(row, 8, record.get('status', 'N/A'), data_format)
        sheet.write(row, 9, record.get('comments', 'N/A'), data_format)
    
    # Auto-adjust column widths
    column_widths = [12, 20, 15, 15, 12, 12, 12, 12, 15, 20]
    for i, width in enumerate(column_widths):
        sheet.set_column(i, i, width)

def write_late_arrivals_to_excel(sheet, records, header_format, data_format, number_format, date_format):
    """Write late arrivals to Excel"""
    
    row = 4
    
    # Headers
    headers = ['Date', 'Employee Name', 'Department', 'Position', 'Check In Time', 'Expected Time', 'Late Minutes']
    
    for col, header in enumerate(headers):
        sheet.write(row, col, header, header_format)
    
    # Data
    for record in records:
        row += 1
        sheet.write(row, 0, record.get('date', 'N/A'), data_format)
        sheet.write(row, 1, record.get('employee_name', 'N/A'), data_format)
        sheet.write(row, 2, record.get('department', 'N/A'), data_format)
        sheet.write(row, 3, record.get('position', 'N/A'), data_format)
        sheet.write(row, 4, record.get('check_in_time', 'N/A'), data_format)
        sheet.write(row, 5, record.get('expected_time', 'N/A'), data_format)
        sheet.write(row, 6, record.get('late_minutes', 0), number_format)
    
    # Auto-adjust column widths
    column_widths = [12, 20, 15, 15, 12, 12, 12]
    for i, width in enumerate(column_widths):
        sheet.set_column(i, i, width)

def write_staff_wise_summary_to_excel(sheet, employees, header_format, data_format, number_format):
    """Write staff-wise summary to Excel - WITH EXPLANATION"""
    
    row = 4  # Start after title and summary
    
    # Headers
    headers = ['Employee ID', 'Employee Name', 'Department', 'Position', 'Expected Days', 'Present Days', 
               'Absent Days', 'Holiday Days', 'Leave Days', 'Late Days', 'Attendance %', 'Punctuality %', 'Performance Rating']
    
    for col, header in enumerate(headers):
        sheet.write(row, col, header, header_format)
    
    # Data
    for emp in employees:
        row += 1
        sheet.write(row, 0, emp.get('employee_id', 'N/A'), data_format)
        sheet.write(row, 1, emp.get('employee_name', 'N/A'), data_format)
        sheet.write(row, 2, emp.get('department', 'N/A'), data_format)
        sheet.write(row, 3, emp.get('position', 'N/A'), data_format)
        sheet.write(row, 4, emp.get('total_expected_days', 0), data_format)
        sheet.write(row, 5, emp.get('present_days', 0), data_format)
        sheet.write(row, 6, emp.get('absent_days', 0), data_format)
        sheet.write(row, 7, emp.get('holiday_days', 0), data_format)
        sheet.write(row, 8, emp.get('leave_days', 0), data_format)
        sheet.write(row, 9, emp.get('late_days', 0), data_format)
        sheet.write(row, 10, emp.get('attendance_rate', 0), number_format)
        sheet.write(row, 11, emp.get('punctuality_rate', 0), number_format)
        sheet.write(row, 12, emp.get('performance_rating', 'N/A'), data_format)
    
    # Add explanation below the data
    explanation_row = row + 3
    sheet.write(explanation_row, 0, "HOW THESE CALCULATIONS WORK:", header_format)
    explanation_row += 1
    
    explanations = [
        "Attendance Rate = (Present Days + Holiday Days + Leave Days) ÷ Total Days × 100",
        "Punctuality Rate = (Present Days - Late Days) ÷ Present Days × 100",
        "Performance combines Attendance (50%) + Punctuality (30%) + Hours Worked (20%)",
        "",
        "IMPORTANT NOTES:",
        "• Public holidays count as 100% attendance (no penalty)",
        "• Approved leave counts as 100% attendance (no penalty)", 
        "• Only unauthorized absences count against you",
        "• Late arrivals only affect punctuality, not attendance",
        "",
        "PERFORMANCE RATINGS:",
        "• 90%+ = Excellent",
        "• 80-89% = Good", 
        "• 70-79% = Satisfactory",
        "• 60-69% = Needs Improvement",
        "• Below 60% = Poor"
    ]
    
    for explanation in explanations:
        sheet.write(explanation_row, 0, explanation, data_format)
        explanation_row += 1
    
    # Auto-adjust column widths
    column_widths = [12, 20, 15, 15, 12, 12, 12, 12, 12, 12, 12, 12, 15]
    for i, width in enumerate(column_widths):
        sheet.set_column(i, i, width)

def write_absent_employees_to_excel(sheet, records, header_format, data_format, date_format):
    """Write absent employees to Excel"""
    
    row = 4
    
    # Headers
    headers = ['Date', 'Day', 'Employee Name', 'Department', 'Position', 'Absence Type']
    
    for col, header in enumerate(headers):
        sheet.write(row, col, header, header_format)
    
    # Data
    for record in records:
        row += 1
        sheet.write(row, 0, record.get('date', 'N/A'), data_format)
        sheet.write(row, 1, record.get('day_name', 'N/A'), data_format)
        sheet.write(row, 2, record.get('employee_name', 'N/A'), data_format)
        sheet.write(row, 3, record.get('department', 'N/A'), data_format)
        sheet.write(row, 4, record.get('position', 'N/A'), data_format)
        sheet.write(row, 5, record.get('absence_type', 'Unspecified'), data_format)
    
    # Auto-adjust column widths
    column_widths = [12, 12, 20, 15, 15, 15]
    for i, width in enumerate(column_widths):
        sheet.set_column(i, i, width)

def write_early_departures_to_excel(sheet, records, header_format, data_format, number_format, date_format):
    """Write early departures to Excel"""
    
    row = 4
    
    # Headers
    headers = ['Date', 'Employee Name', 'Department', 'Check Out Time', 'Expected Time', 'Early Minutes']
    
    for col, header in enumerate(headers):
        sheet.write(row, col, header, header_format)
    
    # Data
    for record in records:
        row += 1
        sheet.write(row, 0, record.get('date', 'N/A'), data_format)
        sheet.write(row, 1, record.get('employee_name', 'N/A'), data_format)
        sheet.write(row, 2, record.get('department', 'N/A'), data_format)
        sheet.write(row, 3, record.get('check_out_time', 'N/A'), data_format)
        sheet.write(row, 4, record.get('expected_time', 'N/A'), data_format)
        sheet.write(row, 5, record.get('early_minutes', 0), number_format)
    
    # Auto-adjust column widths
    column_widths = [12, 20, 15, 12, 12, 12]
    for i, width in enumerate(column_widths):
        sheet.set_column(i, i, width)

def write_overtime_to_excel(sheet, records, header_format, data_format, number_format, date_format):
    """Write overtime records to Excel"""
    
    row = 4
    
    # Headers
    headers = ['Date', 'Employee Name', 'Department', 'Regular Hours', 'Total Hours', 'Overtime Hours']
    
    for col, header in enumerate(headers):
        sheet.write(row, col, header, header_format)
    
    # Data
    for record in records:
        row += 1
        sheet.write(row, 0, record.get('date', 'N/A'), data_format)
        sheet.write(row, 1, record.get('employee_name', 'N/A'), data_format)
        sheet.write(row, 2, record.get('department', 'N/A'), data_format)
        sheet.write(row, 3, record.get('regular_hours', 0), number_format)
        sheet.write(row, 4, record.get('total_hours', 0), number_format)
        sheet.write(row, 5, record.get('overtime_hours', 0), number_format)
    
    # Auto-adjust column widths
    column_widths = [12, 20, 15, 12, 12, 12]
    for i, width in enumerate(column_widths):
        sheet.set_column(i, i, width)

def write_personnel_to_excel(sheet, employees, header_format, data_format):
    """Write personnel data to Excel"""
    
    row = 4
    
    # Headers
    headers = ['Employee ID', 'Full Name', 'Email', 'Phone', 'Department', 'Position', 'Employment Status', 'Gender', 'Age', 'Hire Date']
    
    for col, header in enumerate(headers):
        sheet.write(row, col, header, header_format)
    
    # Data
    for emp in employees:
        row += 1
        sheet.write(row, 0, emp.get('employee_id', 'N/A'), data_format)
        sheet.write(row, 1, emp.get('full_name', 'N/A'), data_format)
        sheet.write(row, 2, emp.get('email', 'N/A'), data_format)
        sheet.write(row, 3, emp.get('phone', 'N/A'), data_format)
        sheet.write(row, 4, emp.get('department', 'N/A'), data_format)
        sheet.write(row, 5, emp.get('position', 'N/A'), data_format)
        sheet.write(row, 6, emp.get('employment_status', 'N/A'), data_format)
        sheet.write(row, 7, emp.get('gender', 'N/A'), data_format)
        sheet.write(row, 8, emp.get('age', 'N/A'), data_format)
        sheet.write(row, 9, emp.get('hire_date', 'N/A'), data_format)
    
    # Auto-adjust column widths
    column_widths = [12, 20, 25, 15, 15, 15, 15, 10, 8, 12]
    for i, width in enumerate(column_widths):
        sheet.set_column(i, i, width)

def calculate_expected_hours(attendance_record, db: Session = None) -> float:
    """Calculate expected working hours from shift and attendance data - NO BREAKS VERSION"""
    try:
        # 1. Try to get from shift information (best source)
        if hasattr(attendance_record, 'shift') and attendance_record.shift:
            shift = attendance_record.shift
            
            # Determine if it's a weekday or weekend
            weekday = attendance_record.date.weekday()  # 0=Monday, 6=Sunday
            is_weekend = weekday >= 5  # Saturday=5, Sunday=6
            
            # Get appropriate shift times
            if is_weekend:
                start_time = getattr(shift, 'weekend_start', None)
                end_time = getattr(shift, 'weekend_end', None)
            else:
                start_time = getattr(shift, 'weekday_start', None)
                end_time = getattr(shift, 'weekday_end', None)
            
            if start_time and end_time:
                # Calculate hours from shift times (NO BREAK DEDUCTION)
                from datetime import datetime, timedelta
                
                start_dt = datetime.combine(attendance_record.date, start_time)
                end_dt = datetime.combine(attendance_record.date, end_time)
                
                # Handle overnight shifts
                if end_dt <= start_dt:
                    end_dt += timedelta(days=1)
                
                duration = end_dt - start_dt
                hours = duration.total_seconds() / 3600
                
                return round(hours, 2)
        
        # 2. Try from captured expected times in attendance record
        if (hasattr(attendance_record, 'expected_start_time') and 
            hasattr(attendance_record, 'expected_end_time') and
            attendance_record.expected_start_time and 
            attendance_record.expected_end_time):
            
            from datetime import datetime, timedelta
            
            start_dt = datetime.combine(attendance_record.date, attendance_record.expected_start_time)
            end_dt = datetime.combine(attendance_record.date, attendance_record.expected_end_time)
            
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)
            
            duration = end_dt - start_dt
            hours = duration.total_seconds() / 3600
                
            return round(hours, 2)
        
        # 3. Weekend/holiday logic
        weekday = attendance_record.date.weekday()
        if weekday >= 5:  # Weekend
            return 4.0  # Shorter weekend hours
        
        # 4. Check if it's a public holiday
        if hasattr(attendance_record, 'holiday_name') and attendance_record.holiday_name:
            return 4.0  # Shorter holiday hours if someone works
        
        # 5. Default standard workday
        return 8.0
        
    except Exception as e:
        logger.warning(f"Error calculating expected hours for {attendance_record.date}: {e}")
        return 8.0


def generate_report_html(report_data: Dict[str, Any], current_user: User) -> str:
    """Generate HTML content for PDF report - LANDSCAPE OPTIMIZED"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{report_data['title']}</title>
    </head>
    <body>
        <div class="header no-break">
            {f'<img src="{get_logo_url()}" class="logo" alt="IUEA Logo">' if get_logo_url() else ''}
            <div class="report-title">{report_data['title']}</div>
            <div class="report-meta">
                <strong>Period:</strong> {report_data['date_range']['formatted']} | 
                <strong>Generated:</strong> {report_data['generated_at'].strftime('%B %d, %Y at %I:%M %p')} | 
                <strong>By:</strong> {current_user.username}
            </div>
        </div>
    """
    
    # Add sections
    for section in report_data['sections']:
        html_content += generate_section_html_landscape(section)
    
    html_content += f"""
        <div class="footer">
            <p><strong>International University of East Africa - Attendance Management System</strong></p>
            <p>This report was automatically generated on {report_data['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}. Please verify data accuracy.</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def _generate_summary_stats_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate summary statistics - LANDSCAPE OPTIMIZED"""
    if section_data.get('message'):
        return f'<div style="background: #fff3cd; padding: 15px; border-radius: 6px; text-align: center;">{section_data["message"]}</div>'
    
    html = '<div class="stats-grid">'
    
    key_stats = [
        ('total_records', 'Total Records'),
        ('unique_employees', 'Employees'),
        ('present_count', 'Present'),
        ('absent_count', 'Absent'),
        ('late_count', 'Late'),
        ('attendance_rate', 'Attendance %'),
    ]
    
    for key, label in key_stats:
        if key in section_data:
            value = section_data[key]
            if key.endswith('_rate'):
                display_value = f"{value}%"
            else:
                display_value = str(value)
            
            html += f'''
                <div class="stat-card">
                    <div class="stat-value">{display_value}</div>
                    <div class="stat-label">{label}</div>
                </div>
            '''
    
    html += '</div>'
    return html

async def process_staff_wise_summary_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process staff-wise attendance summary - NO SHIFT IMPORT NEEDED"""
    
    try:
        logger.info("=== STAFF-WISE SUMMARY WITH WEEKEND SHIFT LOGIC ===")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Filters: {filters}")
        
        # First, let's replicate what the working summary stats does
        summary_query = db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date
            )
        )
        
        # Apply the SAME department filter as summary stats
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            try:
                dept_ids = []
                for d in dept_filter:
                    if isinstance(d, str) and d.isdigit():
                        dept_ids.append(int(d))
                    elif isinstance(d, int):
                        dept_ids.append(d)
                
                if dept_ids:
                    summary_query = summary_query.join(Employee).filter(Employee.department_id.in_(dept_ids))
                    logger.info(f"Applied department filter: {dept_ids}")
            except Exception as e:
                logger.warning(f"Department filter error: {e}")
        
        # Get the attendance records (same as summary stats)
        attendance_records = summary_query.all()
        logger.info(f"Found {len(attendance_records)} attendance records")
        
        # Get unique employee IDs from attendance records
        employee_ids_with_attendance = list(set(record.employee_id for record in attendance_records if record.employee_id))
        logger.info(f"Unique employee IDs with attendance: {employee_ids_with_attendance}")
        
        # Get employees - use the relationship to get shift data
        employees = db.query(Employee).filter(Employee.id.in_(employee_ids_with_attendance)).all()
        logger.info(f"Found {len(employees)} employees with attendance data")
        
        staff_summaries = []
        
        for emp in employees:
            try:
                logger.info(f"Processing employee: {emp.first_name} {emp.last_name} (ID: {emp.id})")
                
                # Get employee's shift information through relationship
                shift = emp.shift  # This uses the relationship instead of a separate query
                
                # Get attendance records for this specific employee
                emp_records = [r for r in attendance_records if r.employee_id == emp.id]
                logger.info(f"  - Found {len(emp_records)} records for this employee")
                
                if emp_records:
                    # Process each day with shift-aware logic
                    total_expected_days = 0  # Only count days that should be worked
                    present_days = 0
                    late_days = 0
                    holiday_days = 0
                    leave_days = 0
                    weekend_work_days = 0
                    weekend_off_days = 0
                    total_hours_worked = 0
                    total_expected_hours = 0
                    
                    for r in emp_records:
                        weekday = r.date.weekday()  # 0=Monday, 6=Sunday
                        is_weekend = weekday >= 5   # Saturday=5, Sunday=6
                        
                        # Determine if this day should be worked based on shift
                        should_work_today = True
                        expected_hours_today = 8.0  # Default
                        
                        if shift and is_weekend:
                            # Check if employee has weekend working hours
                            if (hasattr(shift, 'weekend_start') and hasattr(shift, 'weekend_end') and 
                                shift.weekend_start and shift.weekend_end):
                                # Employee works weekends
                                should_work_today = True
                                weekend_work_days += 1
                                # Calculate weekend hours
                                expected_hours_today = calculate_shift_hours(shift.weekend_start, shift.weekend_end)
                                logger.info(f"    Weekend work day: {r.date} ({weekday_name(weekday)}) - Expected: {expected_hours_today}h")
                            else:
                                # Employee doesn't work weekends
                                should_work_today = False
                                weekend_off_days += 1
                                logger.info(f"    Weekend off day: {r.date} ({weekday_name(weekday)}) - Not counted")
                                continue  # Skip this day entirely
                        elif shift and not is_weekend:
                            # Weekday with shift
                            if (hasattr(shift, 'weekday_start') and hasattr(shift, 'weekday_end') and
                                shift.weekday_start and shift.weekday_end):
                                expected_hours_today = calculate_shift_hours(shift.weekday_start, shift.weekday_end)
                            else:
                                expected_hours_today = 8.0  # Default weekday hours
                        
                        # Only count days that should be worked
                        if should_work_today:
                            total_expected_days += 1
                            total_expected_hours += expected_hours_today
                            
                            # Categorize the day
                            if getattr(r, 'holiday_name', None):  # PUBLIC HOLIDAY
                                holiday_days += 1
                                present_days += 1  # COUNT AS PRESENT (100%)
                                total_hours_worked += expected_hours_today  # Add expected hours for holidays
                                logger.info(f"    Holiday: {r.date} - {r.holiday_name}")
                                
                            elif getattr(r, 'leave_id', None) or getattr(r, 'leave_type', None):  # ON LEAVE
                                leave_days += 1
                                present_days += 1  # COUNT AS PRESENT (100%)
                                total_hours_worked += expected_hours_today  # Add expected hours for leave
                                logger.info(f"    Leave: {r.date} - {getattr(r, 'leave_type', 'Leave')}")
                                
                            elif getattr(r, 'is_present', False):  # ACTUALLY PRESENT
                                present_days += 1
                                actual_hours = getattr(r, 'total_working_hours', 0) or 0
                                total_hours_worked += actual_hours
                                logger.info(f"    Present: {r.date} - {actual_hours}h worked")
                                
                                # Check for lateness
                                if getattr(r, 'is_late', False):
                                    late_days += 1
                                    logger.info(f"      Late arrival on {r.date}")
                            else:
                                # Truly absent
                                logger.info(f"    Absent: {r.date}")
                    
                    absent_days = total_expected_days - present_days
                    
                    logger.info(f"  - Summary: {total_expected_days} expected days, {present_days} present, {absent_days} absent")
                    logger.info(f"  - Weekend: {weekend_work_days} work days, {weekend_off_days} off days")
                    
                    if total_expected_days > 0:  # Only process if there are working days
                        # Calculate rates
                        attendance_rate = (present_days / total_expected_days * 100)
                        
                        # Punctuality rate: only count actual working days (exclude holidays/leave)
                        actual_work_days = present_days - holiday_days - leave_days
                        punctuality_rate = ((actual_work_days - late_days) / actual_work_days * 100) if actual_work_days > 0 else 100
                        
                        # Productivity rate
                        productivity_rate = (total_hours_worked / total_expected_hours * 100) if total_expected_hours > 0 else 0
                        
                        # Performance score
                        performance_score = (attendance_rate * 0.5 + punctuality_rate * 0.3 + min(productivity_rate, 100) * 0.2)
                        
                        if performance_score >= 90:
                            performance_rating, rating_color = 'Excellent', '#28a745'
                        elif performance_score >= 80:
                            performance_rating, rating_color = 'Good', '#28a745'
                        elif performance_score >= 70:
                            performance_rating, rating_color = 'Satisfactory', '#ffc107'
                        elif performance_score >= 60:
                            performance_rating, rating_color = 'Needs Improvement', '#fd7e14'
                        else:
                            performance_rating, rating_color = 'Poor', '#dc3545'
                        
                        logger.info(f"  - Final rates: Attendance {attendance_rate:.1f}%, Punctuality {punctuality_rate:.1f}%, Performance {performance_rating}")
                        
                        # Check if works weekends
                        works_weekends = False
                        if shift:
                            works_weekends = bool(
                                hasattr(shift, 'weekend_start') and hasattr(shift, 'weekend_end') and 
                                shift.weekend_start and shift.weekend_end
                            )
                        
                        staff_summaries.append({
                            'employee_id': emp.employee_id or f'EMP_{emp.id}',
                            'employee_name': f"{emp.first_name} {emp.last_name}",
                            'department': emp.department.name if emp.department else 'No Department',
                            'position': emp.position or 'No Position',
                            'shift_name': shift.name if shift else 'No Shift Assigned',
                            'works_weekends': works_weekends,
                            'total_expected_days': total_expected_days,
                            'present_days': present_days,
                            'absent_days': absent_days,
                            'late_days': late_days,
                            'early_departure_days': 0,
                            'holiday_days': holiday_days,
                            'leave_days': leave_days,
                            'weekend_work_days': weekend_work_days,
                            'weekend_off_days': weekend_off_days,
                            'actual_work_days': actual_work_days,
                            'total_hours_worked': round(total_hours_worked, 2),
                            'total_expected_hours': round(total_expected_hours, 2),
                            'avg_hours_per_day': round(total_hours_worked / present_days, 2) if present_days > 0 else 0,
                            'avg_expected_hours_per_day': round(total_expected_hours / total_expected_days, 2) if total_expected_days > 0 else 0,
                            'attendance_rate': round(attendance_rate, 2),
                            'punctuality_rate': round(punctuality_rate, 2),
                            'productivity_rate': round(productivity_rate, 2),
                            'performance_score': round(performance_score, 2),
                            'performance_rating': performance_rating,
                            'rating_color': rating_color,
                            'issues': [],
                            'shift_type': shift.shift_type if shift and hasattr(shift, 'shift_type') and shift.shift_type else 'standard'
                        })
                        
                        logger.info(f"  - ✅ Successfully added employee to summary")
                    else:
                        logger.info(f"  - ⚠️ No working days found for employee (all weekends/holidays)")
                
            except Exception as e:
                logger.error(f"Error processing employee {emp.id}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        logger.info(f"=== FINAL RESULT: {len(staff_summaries)} employees processed ===")
        
        # Generate insights with weekend information
        insights = []
        if len(staff_summaries) > 0:
            avg_attendance = sum(s['attendance_rate'] for s in staff_summaries) / len(staff_summaries)
            weekend_workers = sum(1 for s in staff_summaries if s['works_weekends'])
            
            insights.append(f"Average attendance rate: {avg_attendance:.1f}%")
            insights.append(f"Successfully processed {len(staff_summaries)} employees")
            if weekend_workers > 0:
                insights.append(f"{weekend_workers} employee(s) work weekends according to their shift schedule")
            insights.append("Weekend days are only counted if the employee's shift includes weekend work")
        else:
            insights.append("No employee data available for analysis")
        
        return {
            'type': 'staff-wise-summary',
            'title': component.title,
            'data': {
                'employees': staff_summaries,
                'total_employees': len(staff_summaries),
                'organization_totals': {},
                'organization_averages': {},
                'insights': insights,
                'sort_by': 'name',
                'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ ERROR in staff-wise summary: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'type': 'staff-wise-summary',
            'title': component.title,
            'error': str(e),
            'data': {
                'employees': [],
                'total_employees': 0,
                'message': f'Error processing staff-wise summary: {str(e)}'
            }
        }

def generate_employee_issues_v2(attendance_rate: float, punctuality_rate: float, productivity_rate: float, actual_work_days: int, total_expected_days: int) -> List[str]:
    """Generate list of issues for an employee - VERSION 2 with holiday/leave logic"""
    issues = []
    
    if attendance_rate < 85:
        issues.append('Low attendance (excluding holidays/leave)')
    if punctuality_rate < 80 and actual_work_days > 0:
        issues.append('Frequent lateness on working days')
    if productivity_rate < 85:
        issues.append('Low productivity')
    if attendance_rate < 70:
        issues.append('Critical attendance issue')
    
    # Add positive notes
    if actual_work_days == 0 and total_expected_days > 0:
        issues.append('All days were holidays/leave - no work days to evaluate')
    elif attendance_rate >= 95:
        issues.append('Excellent attendance')
    
    return issues

def generate_staff_wise_insights_v2(staff_summaries: List[Dict], org_averages: Dict, org_totals: Dict) -> List[str]:
    """Generate insights from staff-wise data - VERSION 2"""
    insights = []
    
    if not staff_summaries:
        return ["No employee data available for analysis"]
    
    total_employees = len(staff_summaries)
    
    # Attendance insights (holidays and leave count as 100%)
    excellent_attendance = [s for s in staff_summaries if s['attendance_rate'] >= 95]
    if excellent_attendance:
        insights.append(f"{len(excellent_attendance)} employee(s) have excellent attendance (≥95%) including holidays/leave")
    
    # Poor attendance (excluding holidays/leave)
    poor_attendance = [s for s in staff_summaries if s['attendance_rate'] < 85]
    if poor_attendance:
        insights.append(f"{len(poor_attendance)} employee(s) have attendance below 85%")
    
    # Punctuality insights (only on actual working days)
    punctuality_issues = [s for s in staff_summaries if s['punctuality_rate'] < 80 and s['actual_work_days'] > 0]
    if punctuality_issues:
        insights.append(f"{len(punctuality_issues)} employee(s) have punctuality issues on working days")
    
    # Holiday and leave utilization
    total_holiday_days = org_totals.get('total_holiday_days', 0)
    total_leave_days = org_totals.get('total_leave_days', 0)
    if total_holiday_days > 0:
        insights.append(f"Organization observed {total_holiday_days} total holiday days (counted as 100% attendance)")
    if total_leave_days > 0:
        insights.append(f"Employees took {total_leave_days} total leave days (counted as 100% attendance)")
    
    # Performance distribution
    excellent_performers = [s for s in staff_summaries if s['performance_rating'] == 'Excellent']
    if excellent_performers:
        insights.append(f"{len(excellent_performers)} employee(s) have excellent overall performance")
    
    return insights

def generate_employee_issues(attendance_rate: float, punctuality_rate: float, productivity_rate: float) -> List[str]:
    """Generate list of issues for an employee"""
    issues = []
    
    if attendance_rate < 85:
        issues.append('Low attendance')
    if punctuality_rate < 80:
        issues.append('Frequent lateness')
    if productivity_rate < 85:
        issues.append('Low productivity')
    if attendance_rate < 70:
        issues.append('Critical attendance issue')
    
    return issues

def generate_staff_wise_insights(staff_summaries: List[Dict], org_averages: Dict) -> List[str]:
    """Generate insights from staff-wise data"""
    insights = []
    
    if not staff_summaries:
        return ["No employee data available for analysis"]
    
    total_employees = len(staff_summaries)
    
    # Top performers
    excellent_performers = [s for s in staff_summaries if s['performance_rating'] == 'Excellent']
    if excellent_performers:
        insights.append(f"{len(excellent_performers)} employee(s) have excellent performance ratings")
    
    # Poor performers
    poor_performers = [s for s in staff_summaries if s['performance_rating'] in ['Poor', 'Needs Improvement']]
    if poor_performers:
        insights.append(f"{len(poor_performers)} employee(s) need performance improvement")
    
    # Attendance issues
    low_attendance = [s for s in staff_summaries if s['attendance_rate'] < 85]
    if low_attendance:
        insights.append(f"{len(low_attendance)} employee(s) have attendance below 85%")
    
    # Punctuality issues  
    punctuality_issues = [s for s in staff_summaries if s['punctuality_rate'] < 80]
    if punctuality_issues:
        insights.append(f"{len(punctuality_issues)} employee(s) have punctuality issues")
    
    # Department comparison
    dept_stats = {}
    for emp in staff_summaries:
        dept = emp['department']
        if dept not in dept_stats:
            dept_stats[dept] = {'count': 0, 'total_attendance': 0}
        dept_stats[dept]['count'] += 1
        dept_stats[dept]['total_attendance'] += emp['attendance_rate']
    
    # Find best and worst departments
    dept_averages = {dept: stats['total_attendance'] / stats['count'] 
                    for dept, stats in dept_stats.items() if stats['count'] > 0}
    
    if len(dept_averages) > 1:
        best_dept = max(dept_averages, key=dept_averages.get)
        worst_dept = min(dept_averages, key=dept_averages.get)
        insights.append(f"Best performing department: {best_dept} ({dept_averages[best_dept]:.1f}% attendance)")
        insights.append(f"Needs attention: {worst_dept} ({dept_averages[worst_dept]:.1f}% attendance)")
    
    return insights

def _generate_staff_wise_summary_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate staff-wise summary table - WITH WEEKEND SHIFT INFO"""
    employees = section_data.get('employees', [])
    total_employees = section_data.get('total_employees', 0)
    insights = section_data.get('insights', [])
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Staff-Wise Attendance Summary</div>
            <strong>Total Employees:</strong> {total_employees} | 
            <strong>Note:</strong> Only counts working days based on employee shift schedules
        </div>
    '''
    
    if insights:
        html += '<div class="summary-box"><div class="summary-title">Key Insights</div>'
        for insight in insights:
            html += f'<p style="margin: 5px 0;">• {insight}</p>'
        html += '</div>'
    
    if not employees:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No employee attendance data found.</div>'
    
    # Updated table with weekend information
    html += '''
        <table>
            <thead>
                <tr>
                    <th class="col-name">Employee</th>
                    <th class="col-dept">Department</th>
                    <th style="width: 40px;">Shift</th>
                    <th style="width: 50px;">Work Days</th>
                    <th style="width: 50px;">Present</th>
                    <th style="width: 40px;">Holidays</th>
                    <th style="width: 40px;">Leave</th>
                    <th style="width: 40px;">Absent</th>
                    <th style="width: 40px;">Late</th>
                    <th style="width: 80px;">Attendance %</th>
                    <th style="width: 80px;">Punctuality %</th>
                    <th style="width: 90px;">Performance</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for emp in employees[:50]:
        employee_name = emp.get('employee_name', 'N/A')
        if len(employee_name) > 18:
            employee_name = employee_name[:15] + "..."
        
        dept_name = emp.get('department', 'N/A')
        if len(dept_name) > 12:
            dept_name = dept_name[:9] + "..."
        
        shift_name = emp.get('shift_name', 'No Shift')
        if len(shift_name) > 8:
            shift_name = shift_name[:5] + "..."
        
        # Show if works weekends
        works_weekends = emp.get('works_weekends', False)
        shift_display = f"{shift_name}" + ("📅" if works_weekends else "")
        
        attendance_rate = emp.get('attendance_rate', 0)
        punctuality_rate = emp.get('punctuality_rate', 0)
        performance_rating = emp.get('performance_rating', 'N/A')
        rating_color = emp.get('rating_color', '#6c757d')
        
        # Color coding
        att_color = 'color: #28a745;' if attendance_rate >= 95 else 'color: #dc3545;' if attendance_rate < 85 else 'color: #ffc107;'
        punc_color = 'color: #28a745;' if punctuality_rate >= 90 else 'color: #dc3545;' if punctuality_rate < 80 else 'color: #ffc107;'
        
        html += f'''
            <tr>
                <td class="col-name"><strong>{employee_name}</strong></td>
                <td class="col-dept">{dept_name}</td>
                <td style="font-size: 0.8em;" title="📅 = Works weekends">{shift_display}</td>
                <td>{emp.get('total_expected_days', 0)}</td>
                <td>{emp.get('present_days', 0)}</td>
                <td style="color: #28a745;">{emp.get('holiday_days', 0)}</td>
                <td style="color: #17a2b8;">{emp.get('leave_days', 0)}</td>
                <td style="color: #dc3545;">{emp.get('absent_days', 0)}</td>
                <td style="color: #ffc107;">{emp.get('late_days', 0)}</td>
                <td style="{att_color} font-weight: bold;">{attendance_rate}%</td>
                <td style="{punc_color} font-weight: bold;">{punctuality_rate}%</td>
                <td style="color: {rating_color}; font-weight: bold; font-size: 0.85em;">{performance_rating}</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    
    if len(employees) > 50:
        html += f'<p style="text-align: center; font-style: italic; color: #666;">Showing first 50 of {len(employees)} employees.</p>'
    
    # Updated explanation
    html += '''
        <div style="background: #f8f9fa; border: 2px solid #e9ecef; border-radius: 10px; padding: 20px; margin-top: 25px;">
            <h3 style="color: #6b1e1e; margin-top: 0;">
                <i class="fas fa-info-circle"></i> How We Calculate These Numbers (Updated with Shift Logic)
            </h3>
            
            <div style="line-height: 1.6; font-size: 14px;">
                <div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 12px; margin: 15px 0;">
                    <strong style="color: #1565c0;">🚨 NEW: Weekend Logic</strong><br>
                    • If your shift includes <strong>weekend hours</strong> (Saturday/Sunday times set), weekend days count as work days<br>
                    • If your shift has <strong>NO weekend hours</strong>, weekends are automatically excluded from calculations<br>
                    • Look for the 📅 icon next to shift names - this means the employee works weekends
                </div>
                
                <p><strong>Here's how this report works:</strong></p>
                
                <div style="margin: 15px 0;">
                    <strong style="color: #28a745;">✅ Attendance Rate:</strong><br>
                    We count <em>Present + Holidays + Leave days</em> and divide by <em>Working days only</em>.<br>
                    <strong>Example:</strong> Someone with Monday-Friday shift: 30 calendar days = 22 weekdays + 8 weekends.<br>
                    We only count the 22 weekdays for attendance calculation (weekends are ignored).
                </div>
                
                <div style="margin: 15px 0;">
                    <strong style="color: #2196f3;">📅 Weekend Workers:</strong><br>
                    If someone works Saturday 9am-5pm and Sunday 10am-6pm (weekend shift), then:<br>
                    30 calendar days = 22 weekdays + 8 weekend days = <strong>30 working days total</strong><br>
                    All days count because their shift includes weekends.
                </div>
                
                <div style="margin: 15px 0;">
                    <strong style="color: #ffc107;">⏰ Punctuality Rate:</strong><br>
                    Only looks at days they actually worked (not holidays/leave) and counts on-time arrivals.
                </div>
                
                <div style="background: #e8f5e8; border-left: 4px solid #28a745; padding: 12px; margin: 15px 0;">
                    <strong style="color: #155724;">💡 Fair System:</strong><br>
                    • <strong>Monday-Friday workers:</strong> Weekends don't count against them<br>
                    • <strong>Weekend workers:</strong> All days count as they're scheduled to work<br>
                    • <strong>Public holidays and approved leave:</strong> Always count as 100% attendance<br>
                    • <strong>Only unauthorized absences</strong> hurt your score
                </div>
                
                <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px;">
                    <strong style="color: #856404;">📝 Example:</strong><br>
                    <strong>Mary (Monday-Friday shift):</strong> 30 calendar days = 22 weekdays counted. Present 18 days, 2 holidays, 1 sick day, absent 1 day.<br>
                    Attendance = (18 + 2 + 1) ÷ 22 = 95% ✅<br><br>
                    <strong>John (7-day shift):</strong> 30 calendar days = 30 working days counted. Present 25 days, 2 holidays, 1 leave, absent 2 days.<br>
                    Attendance = (25 + 2 + 1) ÷ 30 = 93% ✅
                </div>
            </div>
        </div>
    '''
    
    return html

def _generate_individual_records_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate individual records table - LANDSCAPE OPTIMIZED"""
    records = section_data.get('records', [])
    total_count = section_data.get('total_count', 0)
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Individual Attendance Records</div>
            <strong>Records Found:</strong> {total_count}
        </div>
    '''
    
    if not records:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No attendance records found.</div>'
    
    # Limit to 50 records per page for better readability
    display_records = records[:50]
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th class="col-date">Date</th>
                    <th class="col-name">Employee</th>
                    <th class="col-dept">Department</th>
                    <th class="col-time">Check In</th>
                    <th class="col-time">Check Out</th>
                    <th class="col-hours">Hours</th>
                    <th class="col-status">Status</th>
                    <th class="col-comments">Comments</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for record in display_records:
        # Format date compactly
        try:
            date_obj = datetime.strptime(record.get('date', ''), '%Y-%m-%d')
            formatted_date = date_obj.strftime('%m/%d')
        except:
            formatted_date = record.get('date', 'N/A')[-5:]
        
        # Truncate long names and departments
        employee_name = record.get('employee_name', 'N/A')
        if len(employee_name) > 18:
            employee_name = employee_name[:15] + "..."
        
        dept_name = record.get('department', 'N/A')
        if len(dept_name) > 12:
            dept_name = dept_name[:9] + "..."
        
        # Determine status class
        status = record.get('status', 'N/A')
        status_class = 'status-present'
        if 'Late' in status:
            status_class = 'status-late'
        elif 'Absent' in status:
            status_class = 'status-absent'
        elif 'Early' in status:
            status_class = 'status-early'
        
        # Compact comments
        comments = record.get('comments', 'N/A')
        if len(comments) > 20:
            comments = comments[:17] + "..."
        
        html += f'''
            <tr>
                <td class="col-date">{formatted_date}</td>
                <td class="col-name">{employee_name}</td>
                <td class="col-dept">{dept_name}</td>
                <td class="col-time">{record.get('check_in_time', 'N/A')}</td>
                <td class="col-time">{record.get('check_out_time', 'N/A')}</td>
                <td class="col-hours">{record.get('total_hours', 0)}h</td>
                <td class="col-status"><span class="{status_class}">{status}</span></td>
                <td class="col-comments">{comments}</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    
    if len(records) > 50:
        html += f'<p style="text-align: center; font-style: italic; color: #666;">Showing first 50 of {len(records)} records. Full data available in Excel export.</p>'
    
    return html

def _generate_department_summary_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate department summary table - LANDSCAPE OPTIMIZED"""
    departments = section_data.get('departments', [])
    sort_by = section_data.get('sort_by', 'name')
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Department Summary</div>
            <strong>Total Departments:</strong> {len(departments)} | 
            <strong>Sorted by:</strong> {sort_by.replace('_', ' ').title()}
        </div>
    '''
    
    if not departments:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No department data available.</div>'
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th class="col-dept">Department</th>
                    <th style="width: 80px;">Code</th>
                    <th style="width: 80px;">Employees</th>
                    <th style="width: 80px;">Present</th>
                    <th style="width: 80px;">Absent</th>
                    <th style="width: 80px;">Late</th>
                    <th style="width: 100px;">Attendance %</th>
                    <th style="width: 100px;">Punctuality %</th>
                    <th style="width: 80px;">Avg Hours</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for dept in departments:
        attendance_color = 'color: #28a745;' if dept.get('attendance_rate', 0) >= 95 else 'color: #dc3545;' if dept.get('attendance_rate', 0) < 85 else 'color: #ffc107;'
        punctuality_color = 'color: #28a745;' if dept.get('punctuality_rate', 0) >= 90 else 'color: #dc3545;' if dept.get('punctuality_rate', 0) < 80 else 'color: #ffc107;'
        
        html += f'''
            <tr>
                <td><strong>{dept.get('department_name', 'N/A')}</strong></td>
                <td>{dept.get('department_code', 'N/A')}</td>
                <td>{dept.get('total_employees', 0)}</td>
                <td>{dept.get('present_count', 0)}</td>
                <td>{dept.get('absent_count', 0)}</td>
                <td>{dept.get('late_count', 0)}</td>
                <td style="{attendance_color} font-weight: bold;">{dept.get('attendance_rate', 0)}%</td>
                <td style="{punctuality_color} font-weight: bold;">{dept.get('punctuality_rate', 0)}%</td>
                <td>{dept.get('avg_hours', 0)}h</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    return html

def _generate_department_comparison_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate department comparison table - LANDSCAPE OPTIMIZED"""
    departments = section_data.get('departments', [])
    averages = section_data.get('averages', {})
    metrics = section_data.get('metrics', [])
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Department Comparison</div>
            <strong>Metrics:</strong> {', '.join(metrics)}
        </div>
    '''
    
    if averages:
        html += f'''
            <div class="summary-box">
                <div class="summary-title">Organization Averages</div>
                <strong>Attendance:</strong> {averages.get('attendance_rate', 0)}% | 
                <strong>Punctuality:</strong> {averages.get('punctuality_rate', 0)}% | 
                <strong>Working Hours:</strong> {averages.get('avg_hours', 0)}h
            </div>
        '''
    
    if not departments:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No department comparison data available.</div>'
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th class="col-dept">Department</th>
                    <th style="width: 100px;">Attendance %</th>
                    <th style="width: 80px;">vs Avg</th>
                    <th style="width: 100px;">Punctuality %</th>
                    <th style="width: 80px;">vs Avg</th>
                    <th style="width: 80px;">Avg Hours</th>
                    <th style="width: 80px;">vs Avg</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for dept in departments:
        att_vs_avg = dept.get('attendance_vs_avg', 0)
        punc_vs_avg = dept.get('punctuality_vs_avg', 0)
        hours_vs_avg = dept.get('hours_vs_avg', 0)
        
        att_color = 'color: #28a745;' if att_vs_avg >= 0 else 'color: #dc3545;'
        punc_color = 'color: #28a745;' if punc_vs_avg >= 0 else 'color: #dc3545;'
        hours_color = 'color: #28a745;' if hours_vs_avg >= 0 else 'color: #dc3545;'
        
        html += f'''
            <tr>
                <td><strong>{dept.get('department_name', 'N/A')}</strong></td>
                <td>{dept.get('attendance_rate', 0)}%</td>
                <td style="{att_color} font-weight: bold;">{att_vs_avg:+.1f}%</td>
                <td>{dept.get('punctuality_rate', 0)}%</td>
                <td style="{punc_color} font-weight: bold;">{punc_vs_avg:+.1f}%</td>
                <td>{dept.get('avg_hours', 0)}h</td>
                <td style="{hours_color} font-weight: bold;">{hours_vs_avg:+.1f}h</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    return html

def _generate_attendance_overview_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate attendance overview table - LANDSCAPE OPTIMIZED"""
    daily_stats = section_data.get('daily_stats', [])
    total_days = section_data.get('total_days', 0)
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Daily Attendance Overview</div>
            <strong>Total Days:</strong> {total_days}
        </div>
    '''
    
    if not daily_stats:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No daily attendance data available.</div>'
    
    # Limit to recent 30 days for landscape view
    display_stats = daily_stats[-30:] if len(daily_stats) > 30 else daily_stats
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th style="width: 80px;">Date</th>
                    <th style="width: 80px;">Day</th>
                    <th style="width: 60px;">Present</th>
                    <th style="width: 60px;">Absent</th>
                    <th style="width: 60px;">Late</th>
                    <th style="width: 100px;">Attendance %</th>
                    <th style="width: 100px;">Punctuality %</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for day in display_stats:
        attendance_rate = day.get('attendance_rate', 0)
        punctuality_rate = day.get('punctuality_rate', 0)
        
        att_color = 'color: #28a745;' if attendance_rate >= 95 else 'color: #dc3545;' if attendance_rate < 85 else 'color: #ffc107;'
        punc_color = 'color: #28a745;' if punctuality_rate >= 90 else 'color: #dc3545;' if punctuality_rate < 80 else 'color: #ffc107;'
        
        # Format date compactly
        date_str = day.get('date', 'N/A')
        if len(date_str) == 10:  # YYYY-MM-DD format
            date_str = date_str[5:]  # Show MM-DD only
        
        html += f'''
            <tr>
                <td>{date_str}</td>
                <td>{day.get('day_name', 'N/A')[:3]}</td>
                <td>{day.get('present', 0)}</td>
                <td>{day.get('absent', 0)}</td>
                <td>{day.get('late', 0)}</td>
                <td style="{att_color} font-weight: bold;">{attendance_rate}%</td>
                <td style="{punc_color} font-weight: bold;">{punctuality_rate}%</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    
    if len(daily_stats) > 30:
        html += f'<p style="text-align: center; font-style: italic; color: #666;">Showing recent 30 days of {len(daily_stats)} total days. Full data available in Excel export.</p>'
    
    return html

def _generate_trends_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate trends table - LANDSCAPE OPTIMIZED"""
    trends = section_data.get('trends', [])
    trend_type = section_data.get('trend_type', 'daily')
    total_periods = section_data.get('total_periods', 0)
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Attendance Trends ({trend_type.title()})</div>
            <strong>Total Periods:</strong> {total_periods}
        </div>
    '''
    
    if not trends:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No trend data available.</div>'
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th style="width: 120px;">Period</th>
                    <th style="width: 80px;">Total</th>
                    <th style="width: 80px;">Present</th>
                    <th style="width: 80px;">Absent</th>
                    <th style="width: 80px;">Late</th>
                    <th style="width: 100px;">Attendance %</th>
                    <th style="width: 100px;">Punctuality %</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for trend in trends:
        attendance_rate = trend.get('attendance_rate', 0)
        punctuality_rate = trend.get('punctuality_rate', 0)
        
        att_color = 'color: #28a745;' if attendance_rate >= 95 else 'color: #dc3545;' if attendance_rate < 85 else 'color: #ffc107;'
        punc_color = 'color: #28a745;' if punctuality_rate >= 90 else 'color: #dc3545;' if punctuality_rate < 80 else 'color: #ffc107;'
        
        html += f'''
            <tr>
                <td>{trend.get('period', 'N/A')}</td>
                <td>{trend.get('total_records', 0)}</td>
                <td>{trend.get('present', 0)}</td>
                <td>{trend.get('absent', 0)}</td>
                <td>{trend.get('late', 0)}</td>
                <td style="{att_color} font-weight: bold;">{attendance_rate}%</td>
                <td style="{punc_color} font-weight: bold;">{punctuality_rate}%</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    return html

def _generate_late_arrivals_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate late arrivals table - LANDSCAPE OPTIMIZED"""
    records = section_data.get('records', [])
    total_count = section_data.get('total_count', 0)
    threshold_minutes = section_data.get('threshold_minutes', 0)
    avg_lateness = section_data.get('avg_lateness', 0)
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Late Arrivals Summary</div>
            <strong>Total:</strong> {total_count} | 
            <strong>Threshold:</strong> {threshold_minutes} mins | 
            <strong>Average:</strong> {avg_lateness} mins
        </div>
    '''
    
    if not records:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No late arrivals found.</div>'
    
    # Limit to 30 records for landscape view
    display_records = records[:30]
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th class="col-name">Employee</th>
                    <th class="col-dept">Department</th>
                    <th class="col-date">Date</th>
                    <th class="col-time">Check-in</th>
                    <th class="col-time">Expected</th>
                    <th class="col-late">Late By</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for record in display_records:
        # Truncate names for landscape view
        employee_name = record.get('employee_name', 'N/A')
        if len(employee_name) > 20:
            employee_name = employee_name[:17] + "..."
        
        dept_name = record.get('department', 'N/A')
        if len(dept_name) > 15:
            dept_name = dept_name[:12] + "..."
        
        html += f'''
            <tr>
                <td class="col-name">{employee_name}</td>
                <td class="col-dept">{dept_name}</td>
                <td class="col-date">{record.get('date', 'N/A')}</td>
                <td class="col-time">{record.get('check_in_time', 'N/A')}</td>
                <td class="col-time">{record.get('expected_time', 'N/A')}</td>
                <td class="col-late">{record.get('late_minutes', 0)} min</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    
    if len(records) > 30:
        html += f'<p style="text-align: center; font-style: italic; color: #666;">Showing first 30 of {len(records)} records. Full data available in Excel export.</p>'
    
    return html

def _generate_absent_employees_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate absent employees table - LANDSCAPE OPTIMIZED"""
    records = section_data.get('records', [])
    total_count = section_data.get('total_count', 0)
    unique_employees = section_data.get('unique_employees', 0)
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Absence Summary</div>
            <strong>Total Absences:</strong> {total_count} | 
            <strong>Unique Employees:</strong> {unique_employees}
        </div>
    '''
    
    if not records:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No absences found.</div>'
    
    # Limit to 40 records for landscape view
    display_records = records[:40]
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th class="col-name">Employee</th>
                    <th class="col-dept">Department</th>
                    <th class="col-date">Date</th>
                    <th class="col-date">Day</th>
                    <th class="col-comments">Type</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for record in display_records:
        # Truncate names for landscape view
        employee_name = record.get('employee_name', 'N/A')
        if len(employee_name) > 20:
            employee_name = employee_name[:17] + "..."
        
        dept_name = record.get('department', 'N/A')
        if len(dept_name) > 15:
            dept_name = dept_name[:12] + "..."
        
        html += f'''
            <tr>
                <td class="col-name">{employee_name}</td>
                <td class="col-dept">{dept_name}</td>
                <td class="col-date">{record.get('date', 'N/A')}</td>
                <td class="col-date">{record.get('day_name', 'N/A')}</td>
                <td class="col-comments">{record.get('absence_type', 'Unspecified')}</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    
    if len(records) > 40:
        html += f'<p style="text-align: center; font-style: italic; color: #666;">Showing first 40 of {len(records)} records. Full data available in Excel export.</p>'
    
    return html

def _generate_early_departures_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate early departures table - LANDSCAPE OPTIMIZED"""
    records = section_data.get('records', [])
    total_count = section_data.get('total_count', 0)
    threshold_minutes = section_data.get('threshold_minutes', 0)
    avg_early_minutes = section_data.get('avg_early_minutes', 0)
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Early Departures Summary</div>
            <strong>Total:</strong> {total_count} | 
            <strong>Threshold:</strong> {threshold_minutes} mins | 
            <strong>Average:</strong> {avg_early_minutes} mins
        </div>
    '''
    
    if not records:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No early departures found.</div>'
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th class="col-name">Employee</th>
                    <th class="col-dept">Department</th>
                    <th class="col-date">Date</th>
                    <th class="col-time">Check-out</th>
                    <th class="col-time">Expected</th>
                    <th class="col-late">Early By</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for record in records:
        # Truncate names for landscape view
        employee_name = record.get('employee_name', 'N/A')
        if len(employee_name) > 20:
            employee_name = employee_name[:17] + "..."
        
        dept_name = record.get('department', 'N/A')
        if len(dept_name) > 15:
            dept_name = dept_name[:12] + "..."
        
        html += f'''
            <tr>
                <td class="col-name">{employee_name}</td>
                <td class="col-dept">{dept_name}</td>
                <td class="col-date">{record.get('date', 'N/A')}</td>
                <td class="col-time">{record.get('check_out_time', 'N/A')}</td>
                <td class="col-time">{record.get('expected_time', 'N/A')}</td>
                <td class="col-late">{record.get('early_minutes', 0)} min</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    return html

def _generate_personnel_list_html(section_data: Dict[str, Any]) -> str:
    """Generate HTML for personnel list section WITH ACTUAL NAMES"""
    employees = section_data.get('employees', [])
    total_count = section_data.get('total_count', 0)
    summary_stats = section_data.get('summary_stats', {})
    
    # Summary section
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Summary</div>
            <p><strong>Total Employees:</strong> {total_count}</p>
    '''
    
    if summary_stats.get('avg_age'):
        html += f'<p><strong>Average Age:</strong> {summary_stats["avg_age"]} years</p>'
    
    gender_breakdown = summary_stats.get('gender_breakdown', {})
    if gender_breakdown:
        breakdown_text = ', '.join([f"{gender}: {count}" for gender, count in gender_breakdown.items() if gender != 'N/A'])
        html += f'<p><strong>Gender Distribution:</strong> {breakdown_text}</p>'
    
    html += '</div>'
    
    if not employees:
        return html + '<p><em>No employees found matching the selected criteria.</em></p>'
    
    display_employees = employees[:100]  # Limit for PDF
    report_type = section_data.get('report_type', 'directory')
    
    # Generate table based on report type - WITH ACTUAL NAMES
    if report_type == 'directory':
        html += _generate_employee_directory_table(display_employees)
    elif report_type == 'demographics':
        html += _generate_employee_demographics_table(display_employees)
    else:
        html += _generate_employee_organizational_table(display_employees)
    
    if len(employees) > 100:
        html += f'<p><em>Showing first 100 of {len(employees)} employees. Full data available in Excel export.</em></p>'
    
    return html

def _generate_overtime_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate overtime table - LANDSCAPE OPTIMIZED"""
    records = section_data.get('records', [])
    total_count = section_data.get('total_count', 0)
    threshold_hours = section_data.get('threshold_hours', 0)
    total_overtime_hours = section_data.get('total_overtime_hours', 0)
    avg_overtime_hours = section_data.get('avg_overtime_hours', 0)
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Overtime Summary</div>
            <strong>Total Records:</strong> {total_count} | 
            <strong>Threshold:</strong> {threshold_hours}h | 
            <strong>Total OT:</strong> {total_overtime_hours}h | 
            <strong>Average:</strong> {avg_overtime_hours}h
        </div>
    '''
    
    if not records:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No overtime records found.</div>'
    
    # Limit to 40 records for landscape view
    display_records = records[:40]
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th class="col-name">Employee</th>
                    <th class="col-dept">Department</th>
                    <th class="col-date">Date</th>
                    <th style="width: 80px;">Regular</th>
                    <th style="width: 80px;">Total</th>
                    <th style="width: 80px;">Overtime</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for record in display_records:
        # Truncate names for landscape view
        employee_name = record.get('employee_name', 'N/A')
        if len(employee_name) > 20:
            employee_name = employee_name[:17] + "..."
        
        dept_name = record.get('department', 'N/A')
        if len(dept_name) > 15:
            dept_name = dept_name[:12] + "..."
        
        html += f'''
            <tr>
                <td class="col-name">{employee_name}</td>
                <td class="col-dept">{dept_name}</td>
                <td class="col-date">{record.get('date', 'N/A')}</td>
                <td>{record.get('regular_hours', 0)}h</td>
                <td>{record.get('total_hours', 0)}h</td>
                <td style="color: #28a745; font-weight: bold;">{record.get('overtime_hours', 0)}h</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    
    if len(records) > 40:
        html += f'<p style="text-align: center; font-style: italic; color: #666;">Showing first 40 of {len(records)} records. Full data available in Excel export.</p>'
    
    return html

def _generate_department_html(section_data: Dict[str, Any], section_type: str) -> str:
    """Generate HTML for department summary and comparison sections"""
    departments = section_data.get('departments', [])
    
    if not departments:
        return '<p><em>No department data available.</em></p>'
    
    html = '''
        <table>
            <thead>
                <tr>
                    <th>Department</th>
                    <th>Employees</th>
                    <th>Attendance Rate</th>
                    <th>Punctuality Rate</th>
                    <th>Avg Hours</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for dept in departments:
        html += f'''
            <tr>
                <td><strong>{dept.get('department_name', 'N/A')}</strong><br><small>{dept.get('department_code', 'N/A')}</small></td>
                <td>{dept.get('total_employees', 0)}</td>
                <td>{dept.get('attendance_rate', 0)}%</td>
                <td>{dept.get('punctuality_rate', 0)}%</td>
                <td>{dept.get('avg_hours', 0)}h</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    
    # Add comparison metrics for department-comparison
    if section_type == 'department-comparison':
        averages = section_data.get('averages', {})
        if averages:
            html += f'''
                <div class="summary-box">
                    <div class="summary-title">Organization Averages</div>
                    <p><strong>Attendance:</strong> {averages.get('attendance_rate', 0)}%</p>
                    <p><strong>Punctuality:</strong> {averages.get('punctuality_rate', 0)}%</p>
                    <p><strong>Working Hours:</strong> {averages.get('avg_hours', 0)}h</p>
                </div>
            '''
    
    return html

def _generate_trends_html(section_data: Dict[str, Any]) -> str:
    """Generate HTML for trends section"""
    trends = section_data.get('trends', [])
    trend_type = section_data.get('trend_type', 'daily')
    total_periods = section_data.get('total_periods', 0)
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Trend Analysis ({trend_type.title()})</div>
            <p><strong>Total Periods:</strong> {total_periods}</p>
        </div>
    '''
    
    if not trends:
        return html + '<p><em>No trend data available.</em></p>'
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th>Period</th>
                    <th>Total Records</th>
                    <th>Present</th>
                    <th>Absent</th>
                    <th>Late</th>
                    <th>Attendance Rate</th>
                    <th>Punctuality Rate</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for trend in trends:
        html += f'''
            <tr>
                <td>{trend.get('period', 'N/A')}</td>
                <td>{trend.get('total_records', 0)}</td>
                <td>{trend.get('present', 0)}</td>
                <td>{trend.get('absent', 0)}</td>
                <td>{trend.get('late', 0)}</td>
                <td>{trend.get('attendance_rate', 0)}%</td>
                <td>{trend.get('punctuality_rate', 0)}%</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    return html

def _generate_performance_metrics_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate performance metrics - LANDSCAPE OPTIMIZED"""
    metrics = section_data.get('metrics', {})
    insights = section_data.get('insights', [])
    period_summary = section_data.get('period_summary', {})
    
    if not metrics:
        message = section_data.get('message', 'No performance data available')
        return f'<div style="background: #fff3cd; padding: 15px; border-radius: 6px; text-align: center;">{message}</div>'
    
    html = '<div class="stats-grid">'
    
    for metric_key, metric_data in metrics.items():
        status_color = _get_metric_status_color(metric_data.get('status', 'unknown'))
        benchmark = metric_data.get('benchmark', 0)
        value = metric_data.get('value', 0)
        
        html += f'''
            <div class="stat-card">
                <div class="stat-value" style="color: {status_color};">{value}{metric_data.get('unit', '')}</div>
                <div class="stat-label">{metric_data.get('name', metric_key)}</div>
                <div style="font-size: 8px; color: #666;">Target: {benchmark}{metric_data.get('unit', '')}</div>
            </div>
        '''
    
    html += '</div>'
    
    if insights:
        html += '<div class="summary-box"><div class="summary-title">Key Insights</div>'
        for insight in insights:
            html += f'<p style="margin: 5px 0;">• {insight}</p>'
        html += '</div>'
    
    return html

def _generate_attendance_overview_html(section_data: Dict[str, Any]) -> str:
    """Generate HTML for attendance overview section"""
    daily_stats = section_data.get('daily_stats', [])
    total_days = section_data.get('total_days', 0)
    
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Attendance Overview</div>
            <p><strong>Total Days:</strong> {total_days}</p>
        </div>
    '''
    
    if not daily_stats:
        return html + '<p><em>No attendance overview data available.</em></p>'
    
    html += '''
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Day</th>
                    <th>Present</th>
                    <th>Absent</th>
                    <th>Late</th>
                    <th>Attendance Rate</th>
                    <th>Punctuality Rate</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for day in daily_stats:
        html += f'''
            <tr>
                <td>{day.get('date', 'N/A')}</td>
                <td>{day.get('day_name', 'N/A')}</td>
                <td>{day.get('present', 0)}</td>
                <td>{day.get('absent', 0)}</td>
                <td>{day.get('late', 0)}</td>
                <td>{day.get('attendance_rate', 0)}%</td>
                <td>{day.get('punctuality_rate', 0)}%</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    return html

def _generate_employee_directory_table(employees: List[Dict[str, Any]]) -> str:
    """Generate employee directory table WITH ACTUAL NAMES"""
    html = '''
        <table>
            <thead>
                <tr>
                    <th>Employee ID</th>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Phone</th>
                    <th>Department</th>
                    <th>Position</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for emp in employees:
        html += f'''
            <tr>
                <td>{emp.get('employee_id', 'N/A')}</td>
                <td>{emp.get('full_name', 'N/A')}</td>           <!-- ACTUAL NAME -->
                <td>{emp.get('email', 'N/A')}</td>
                <td>{emp.get('phone', 'N/A')}</td>
                <td>{emp.get('department', 'N/A')}</td>          <!-- ACTUAL DEPARTMENT -->
                <td>{emp.get('position', 'N/A')}</td>           <!-- ACTUAL POSITION -->
                <td>{emp.get('employment_status', 'N/A')}</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    return html

def _generate_personnel_list_html_landscape(section_data: Dict[str, Any]) -> str:
    """Generate personnel list table - LANDSCAPE OPTIMIZED"""
    employees = section_data.get('employees', [])
    total_count = section_data.get('total_count', 0)
    summary_stats = section_data.get('summary_stats', {})
    report_type = section_data.get('report_type', 'directory')
    
    # Summary section
    html = f'''
        <div class="summary-box">
            <div class="summary-title">Employee Directory Summary</div>
            <strong>Total Employees:</strong> {total_count}
    '''
    
    if summary_stats.get('avg_age'):
        html += f' | <strong>Average Age:</strong> {summary_stats["avg_age"]} years'
    
    gender_breakdown = summary_stats.get('gender_breakdown', {})
    if gender_breakdown:
        breakdown_text = ', '.join([f"{gender}: {count}" for gender, count in gender_breakdown.items() if gender != 'N/A'])
        if breakdown_text:
            html += f' | <strong>Gender Distribution:</strong> {breakdown_text}'
    
    html += '</div>'
    
    if not employees:
        return html + '<div style="text-align: center; color: #666; padding: 20px;">No employees found matching the selected criteria.</div>'
    
    # Limit for PDF display
    display_employees = employees[:100]
    
    # Generate table based on report type
    if report_type == 'directory':
        html += _generate_employee_directory_table_landscape(display_employees)
    elif report_type == 'demographics':
        html += _generate_employee_demographics_table_landscape(display_employees)
    else:  # organizational or default
        html += _generate_employee_organizational_table_landscape(display_employees)
    
    if len(employees) > 100:
        html += f'<p style="text-align: center; font-style: italic; color: #666;">Showing first 100 of {len(employees)} employees. Full data available in Excel export.</p>'
    
    return html

def _generate_employee_directory_table_landscape(employees: List[Dict[str, Any]]) -> str:
    """Generate employee directory table - LANDSCAPE OPTIMIZED"""
    html = '''
        <table>
            <thead>
                <tr>
                    <th class="col-name">Employee ID</th>
                    <th class="col-name">Full Name</th>
                    <th style="width: 200px;">Email</th>
                    <th style="width: 120px;">Phone</th>
                    <th class="col-dept">Department</th>
                    <th style="width: 150px;">Position</th>
                    <th style="width: 100px;">Status</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for emp in employees:
        # Truncate long values for landscape view
        email = emp.get('email', 'N/A')
        if len(email) > 30:
            email = email[:27] + "..."
        
        position = emp.get('position', 'N/A')
        if len(position) > 20:
            position = position[:17] + "..."
        
        dept_name = emp.get('department', 'N/A')
        if len(dept_name) > 15:
            dept_name = dept_name[:12] + "..."
        
        html += f'''
            <tr>
                <td>{emp.get('employee_id', 'N/A')}</td>
                <td><strong>{emp.get('full_name', 'N/A')}</strong></td>
                <td>{email}</td>
                <td>{emp.get('phone', 'N/A')}</td>
                <td>{dept_name}</td>
                <td>{position}</td>
                <td>{emp.get('employment_status', 'N/A')}</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    return html

def _generate_employee_demographics_table_landscape(employees: List[Dict[str, Any]]) -> str:
    """Generate employee demographics table - LANDSCAPE OPTIMIZED"""
    html = '''
        <table>
            <thead>
                <tr>
                    <th class="col-name">Full Name</th>
                    <th style="width: 80px;">Gender</th>
                    <th style="width: 60px;">Age</th>
                    <th style="width: 100px;">Marital Status</th>
                    <th style="width: 120px;">Nationality</th>
                    <th class="col-dept">Department</th>
                    <th style="width: 100px;">Hire Date</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for emp in employees:
        age_display = str(emp.get('age', 'N/A')) if emp.get('age') else 'N/A'
        
        dept_name = emp.get('department', 'N/A')
        if len(dept_name) > 15:
            dept_name = dept_name[:12] + "..."
        
        nationality = emp.get('nationality', 'N/A')
        if len(nationality) > 15:
            nationality = nationality[:12] + "..."
        
        html += f'''
            <tr>
                <td><strong>{emp.get('full_name', 'N/A')}</strong></td>
                <td>{emp.get('gender', 'N/A')}</td>
                <td>{age_display}</td>
                <td>{emp.get('marital_status', 'N/A')}</td>
                <td>{nationality}</td>
                <td>{dept_name}</td>
                <td>{emp.get('hire_date', 'N/A')}</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    return html

def _generate_employee_organizational_table_landscape(employees: List[Dict[str, Any]]) -> str:
    """Generate employee organizational table - LANDSCAPE OPTIMIZED"""
    html = '''
        <table>
            <thead>
                <tr>
                    <th style="width: 100px;">Employee ID</th>
                    <th class="col-name">Full Name</th>
                    <th class="col-dept">Department</th>
                    <th style="width: 150px;">Position</th>
                    <th style="width: 200px;">Email</th>
                    <th style="width: 120px;">Phone</th>
                    <th style="width: 100px;">Status</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for emp in employees:
        # Truncate long values for landscape view
        email = emp.get('email', 'N/A')
        if len(email) > 30:
            email = email[:27] + "..."
        
        position = emp.get('position', 'N/A')
        if len(position) > 20:
            position = position[:17] + "..."
        
        dept_name = emp.get('department', 'N/A')
        if len(dept_name) > 15:
            dept_name = dept_name[:12] + "..."
        
        html += f'''
            <tr>
                <td>{emp.get('employee_id', 'N/A')}</td>
                <td><strong>{emp.get('full_name', 'N/A')}</strong></td>
                <td>{dept_name}</td>
                <td>{position}</td>
                <td>{email}</td>
                <td>{emp.get('phone', 'N/A')}</td>
                <td>{emp.get('employment_status', 'N/A')}</td>
            </tr>
        '''
    
    html += '</tbody></table>'
    return html

def _get_status_color(status: str) -> str:
    """Get color for attendance status"""
    status_colors = {
        'Present': '#28a745',
        'Late': '#ffc107', 
        'Absent': '#dc3545',
        'Early Departure': '#fd7e14',
        'Late & Early Departure': '#dc3545'
    }
    return status_colors.get(status, '#6c757d')

def _get_metric_status_color(status: str) -> str:
    """Get color for metric status"""
    status_colors = {
        'good': '#28a745',
        'warning': '#ffc107',
        'poor': '#dc3545'
    }
    return status_colors.get(status, '#6c757d')

def get_logo_url() -> str:
    """Get logo URL for reports"""
    logo_paths = [
        Path("static/images/logo.png"),
        Path("static/images/logo.jpg"),
        Path("static/images/logos/apple-touch-icon.png"),
        Path("static/assets/logo.png"),
        Path("static/img/logo.png")
    ]
    
    for logo_path in logo_paths:
        if logo_path.exists():
            return f"file://{logo_path.resolve()}"
    
    return ""

# Debug endpoints for testing
@router.get("/debug/data-check")
async def debug_data_check(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint to check data availability"""
    try:
        # Check attendance data
        attendance_count = db.query(ProcessedAttendance).count()
        
        # Check recent attendance data
        today = date.today()
        last_30_days = today - timedelta(days=30)
        recent_attendance = db.query(ProcessedAttendance).filter(
            ProcessedAttendance.date >= last_30_days
        ).count()
        
        # Check employees
        employee_count = db.query(Employee).count()
        active_employees = db.query(Employee).filter(
            Employee.status == StatusEnum.ACTIVE
        ).count()
        
        # Check departments
        department_count = db.query(Department).filter(
            Department.is_active == True
        ).count()
        
        # Sample attendance record
        sample_record = db.query(ProcessedAttendance).first()
        
        # Date range analysis
        if attendance_count > 0:
            earliest_record = db.query(ProcessedAttendance).order_by(ProcessedAttendance.date.asc()).first()
            latest_record = db.query(ProcessedAttendance).order_by(ProcessedAttendance.date.desc()).first()
            date_range = f"{earliest_record.date} to {latest_record.date}" if earliest_record and latest_record else "No data"
        else:
            date_range = "No attendance data"
        
        return {
            "status": "success",
            "data_summary": {
                "total_attendance_records": attendance_count,
                "recent_attendance_records": recent_attendance,
                "total_employees": employee_count,
                "active_employees": active_employees,
                "total_departments": department_count,
                "attendance_date_range": date_range,
                "sample_record_available": sample_record is not None,
                "sample_record_date": sample_record.date.isoformat() if sample_record else None
            },
            "recommendations": generate_data_recommendations(
                attendance_count, recent_attendance, employee_count, department_count
            )
        }
        
    except Exception as e:
        logger.error(f"Error in debug data check: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Error checking data availability"
        }

def generate_data_recommendations(attendance_count: int, recent_attendance: int, employee_count: int, department_count: int) -> List[str]:
    """Generate recommendations based on data availability"""
    recommendations = []
    
    if attendance_count == 0:
        recommendations.append("No attendance data found. Please ensure attendance data is being synced from your biometric devices.")
    elif recent_attendance == 0:
        recommendations.append("No recent attendance data found. Check if attendance sync is working for the last 30 days.")
    elif recent_attendance < employee_count * 10:  # Less than 10 days average
        recommendations.append("Limited recent attendance data. Consider syncing more data for better reports.")
    
    if employee_count == 0:
        recommendations.append("No employee records found. Please add employees to the system.")
    elif department_count == 0:
        recommendations.append("No departments found. Please create departments and assign employees.")
    
    if len(recommendations) == 0:
        recommendations.append("Data looks good! You should be able to generate comprehensive reports.")
    
    return recommendations

# Add to your reports router (paste-1.txt)
async def process_leave_balance_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Process leave balance component - MISSING FROM YOUR SYSTEM"""
    
    try:
        # Get employees based on filters
        query = db.query(Employee)
        
        # Apply department filter
        dept_filter = filters.get('departments', [])
        if dept_filter and dept_filter != ['all'] and 'all' not in dept_filter:
            dept_ids = [int(d) for d in dept_filter if str(d).isdigit()]
            if dept_ids:
                query = query.filter(Employee.department_id.in_(dept_ids))
        
        employees = query.all()
        
        leave_balances = []
        for emp in employees:
            # Calculate leave balances (YOU NEED TO IMPLEMENT THIS)
            annual_balance = calculate_annual_leave_balance(emp, db, end_date)
            sick_balance = calculate_sick_leave_balance(emp, db, end_date)
            leave_taken = get_leave_taken_in_period(emp.id, db, start_date, end_date)
            
            leave_balances.append({
                'employee_id': emp.employee_id,
                'employee_name': f"{emp.first_name} {emp.last_name}",
                'department': emp.department.name if emp.department else 'No Department',
                'annual_leave_balance': annual_balance,
                'sick_leave_balance': sick_balance,
                'total_leave_taken': leave_taken['total_days'],
                'leave_by_type': leave_taken['by_type'],
                'leave_remaining': annual_balance - leave_taken.get('annual', 0)
            })
        
        return {
            'type': 'leave-balance',
            'title': component.title,
            'data': {
                'employees': leave_balances,
                'total_employees': len(leave_balances),
                'summary': calculate_leave_summary(leave_balances)
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing leave balance component: {str(e)}")
        return {
            'type': 'leave-balance',
            'title': component.title,
            'error': str(e),
            'data': {}
        }

async def process_leave_utilization_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze leave usage patterns - MISSING FROM YOUR SYSTEM"""
    
    # Get all leave records in the period
    leave_query = db.query(Leave).filter(
        and_(
            or_(
                and_(Leave.start_date >= start_date, Leave.start_date <= end_date),
                and_(Leave.end_date >= start_date, Leave.end_date <= end_date),
                and_(Leave.start_date <= start_date, Leave.end_date >= end_date)
            ),
            Leave.status == LeaveStatusEnum.ACTIVE
        )
    )
    
    # Apply filters...
    
    leave_records = leave_query.all()
    
    # Analyze patterns
    utilization_data = {
        'leave_by_month': analyze_monthly_leave_patterns(leave_records),
        'leave_by_type': analyze_leave_by_type(leave_records),
        'leave_by_department': analyze_departmental_leave(leave_records),
        'peak_leave_periods': identify_peak_leave_periods(leave_records),
        'average_leave_duration': calculate_average_duration(leave_records),
        'leave_approval_rates': calculate_approval_rates(leave_records)
    }
    
    return {
        'type': 'leave-utilization',
        'title': component.title,
        'data': utilization_data
    }

async def process_leave_calendar_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Leave calendar view - MISSING FROM YOUR SYSTEM"""
    
    # Get upcoming and current leave
    upcoming_leave = db.query(Leave).filter(
        and_(
            Leave.start_date >= date.today(),
            Leave.start_date <= end_date,
            Leave.status == LeaveStatusEnum.ACTIVE
        )
    ).order_by(Leave.start_date).all()
    
    # Format for calendar display
    calendar_data = []
    for leave_record in upcoming_leave:
        calendar_data.append({
            'employee_name': f"{leave_record.employee.first_name} {leave_record.employee.last_name}",
            'department': leave_record.employee.department.name if leave_record.employee.department else 'No Department',
            'leave_type': leave_record.leave_type.name,
            'start_date': leave_record.start_date.isoformat(),
            'end_date': leave_record.end_date.isoformat(),
            'duration_days': (leave_record.end_date - leave_record.start_date).days + 1,
            'reason': leave_record.reason or 'Not specified',
            'status': leave_record.status.value
        })
    
    return {
        'type': 'leave-calendar',
        'title': component.title,
        'data': {
            'upcoming_leave': calendar_data,
            'total_upcoming': len(calendar_data),
            'impact_analysis': analyze_staffing_impact(calendar_data)
        }
    }

async def process_holiday_impact_component(component: ReportComponent, db: Session, start_date: date, end_date: date, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze public holiday impact - ENHANCE YOUR EXISTING CODE"""
    
    # Get public holidays in the period
    holidays = db.query(PublicHoliday).filter(
        and_(
            PublicHoliday.date >= start_date,
            PublicHoliday.date <= end_date,
            PublicHoliday.is_active == True
        )
    ).order_by(PublicHoliday.date).all()
    
    holiday_analysis = []
    for holiday in holidays:
        # Analyze attendance on holidays (who worked)
        holiday_attendance = db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.date == holiday.date,
                ProcessedAttendance.is_present == True,
                ProcessedAttendance.status != "public_holiday"  # Those who actually worked
            )
        ).all()
        
        holiday_analysis.append({
            'holiday_name': holiday.name,
            'holiday_date': holiday.date.isoformat(),
            'day_of_week': holiday.date.strftime('%A'),
            'employees_worked': len(holiday_attendance),
            'working_employees': [
                {
                    'employee_name': att.employee.first_name + ' ' + att.employee.last_name,
                    'department': att.employee.department.name if att.employee.department else 'No Department',
                    'hours_worked': att.total_working_hours or 0,
                    'overtime_eligible': att.total_working_hours > 0  # Holiday work = overtime
                }
                for att in holiday_attendance
            ],
            'total_holiday_hours': sum(att.total_working_hours or 0 for att in holiday_attendance)
        })
    
    return {
        'type': 'holiday-impact',
        'title': component.title,
        'data': {
            'holidays': holiday_analysis,
            'total_holidays': len(holidays),
            'total_employees_worked_holidays': sum(len(h['working_employees']) for h in holiday_analysis),
            'total_holiday_work_hours': sum(h['total_holiday_hours'] for h in holiday_analysis)
        }
    }

# Health check
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "reports",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

logger.info("Production-ready reports router initialized successfully")