"""
Staff Attendance Service
Handles all server-side logic for staff attendance management,
replacing JavaScript functionality with Python backend processing.
"""

import logging
import json
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from fastapi import HTTPException, status as http_status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from models.database import get_db
from models.employee import Employee, User
from models.attendance import ProcessedAttendance, AttendanceSyncLog
from models.attendance_exception import AttendanceException
from services.attendance_service import AttendanceService
from services.biometric_service import BiometricAPIService

# Configure logging
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")

class StaffAttendanceService:
    """Service class to handle all staff attendance operations"""
    
    def __init__(self, db: Session):
        self.db = db
        self.attendance_service = AttendanceService(db)
    
    def get_employee_by_id(self, employee_id: int) -> Employee:
        """Get employee by ID with error handling"""
        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Employee with ID {employee_id} not found"
            )
        return employee
    
    def get_current_month_dates(self) -> Tuple[date, date]:
        """Get first and last day of current month"""
        today = date.today()
        first_day = date(today.year, today.month, 1)
        
        # Get last day of month
        if today.month == 12:
            last_day = date(today.year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(today.year, today.month + 1, 1) - timedelta(days=1)
        
        return first_day, min(last_day, today)
    
    def parse_filter_dates(self, period: str) -> Tuple[date, date]:
        """Parse period string and return start and end dates"""
        today = date.today()
        
        if period == "current-month":
            return self.get_current_month_dates()
        
        elif period == "last-month":
            # Last month
            if today.month == 1:
                last_month = 12
                year = today.year - 1
            else:
                last_month = today.month - 1
                year = today.year
            
            start_date = date(year, last_month, 1)
            if last_month == 12:
                end_date = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(year, last_month + 1, 1) - timedelta(days=1)
            
            return start_date, end_date
        
        elif period == "last-3-months":
            # Last 3 months
            end_date = today
            start_date = today - timedelta(days=90)
            return start_date, end_date
        
        elif period == "current-year":
            # Current year
            start_date = date(today.year, 1, 1)
            end_date = today
            return start_date, end_date
        
        else:
            # Default to current month
            return self.get_current_month_dates()
    
    def get_attendance_data(self, employee_id: int, start_date: date, end_date: date) -> Dict[str, Any]:
        """Get formatted attendance data for the employee"""
        try:
            employee = self.get_employee_by_id(employee_id)
            
            # Check if employee has biometric ID
            if not employee.biometric_id:
                return {
                    'employee': employee,
                    'has_biometric_id': False,
                    'error': "No biometric ID assigned to this employee",
                    'records': [],
                    'summary': {
                        'total_days': 0,
                        'present_days': 0,
                        'absent_days': 0,
                        'late_days': 0,
                        'total_hours': 0,
                        'attendance_percentage': 0
                    },
                    'start_date': start_date,
                    'end_date': end_date,
                    'last_sync': None,
                    'exceptions': {}
                }
            
            # Get attendance data from service
            attendance_data = self.attendance_service.get_employee_attendance_summary(
                employee_id=employee_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get exceptions for the period
            exceptions = self.attendance_service.get_attendance_exceptions(
                employee_id, start_date, end_date
            )
            
            # Format records for template
            formatted_records = []
            for record in attendance_data['records']:
                exception_info = None
                record_date = record.date
                
                if record_date in exceptions:
                    exception_info = {
                        'id': exceptions[record_date].id,
                        'reason': exceptions[record_date].reason,
                        'created_at': exceptions[record_date].created_at,
                        'created_by': exceptions[record_date].created_by_user.username if hasattr(exceptions[record_date], 'created_by_user') and exceptions[record_date].created_by_user else 'System'
                    }
                
                shift_name = record.shift.name if record.shift else None
                
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
                    'has_exception': exception_info is not None,
                    'exception_info': exception_info
                }
                formatted_records.append(formatted_record)
            
            # Get last sync status
            last_sync = self.db.query(AttendanceSyncLog).filter(
                AttendanceSyncLog.employee_id == employee_id
            ).order_by(desc(AttendanceSyncLog.created_at)).first()
            
            return {
                'employee': employee,
                'has_biometric_id': True,
                'records': formatted_records,
                'summary': attendance_data['summary'],
                'start_date': start_date,
                'end_date': end_date,
                'last_sync': last_sync,
                'exceptions': exceptions,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error getting attendance data for employee {employee_id}: {str(e)}")
            return {
                'employee': None,
                'has_biometric_id': False,
                'error': f"Error loading attendance data: {str(e)}",
                'records': [],
                'summary': {
                    'total_days': 0,
                    'present_days': 0,
                    'absent_days': 0,
                    'late_days': 0,
                    'total_hours': 0,
                    'attendance_percentage': 0
                },
                'start_date': start_date,
                'end_date': end_date,
                'last_sync': None,
                'exceptions': {}
            }
    
    async def sync_employee_attendance(self, employee_id: int, sync_type: str = "incremental", 
                                     start_date: Optional[date] = None, 
                                     end_date: Optional[date] = None) -> Dict[str, Any]:
        """Sync attendance data for employee"""
        try:
            employee = self.get_employee_by_id(employee_id)
            
            if not employee.biometric_id:
                return {
                    "success": False,
                    "message": "Employee has no biometric ID assigned"
                }
            
            logger.info(f"Starting {sync_type} attendance sync for employee {employee_id}")
            
            if sync_type == "range" and start_date and end_date:
                # Validate date range
                if start_date > end_date:
                    return {
                        "success": False,
                        "message": "Start date must be before end date"
                    }
                
                if start_date > date.today():
                    return {
                        "success": False,
                        "message": "Start date cannot be in the future"
                    }
                
                result = await self.attendance_service.sync_employee_attendance_range(
                    employee_id=employee_id,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                # Full or incremental sync
                force_full = sync_type == "full"
                result = await self.attendance_service.sync_employee_attendance(
                    employee_id=employee_id,
                    force_full_sync=force_full
                )
            
            # Get updated last sync info
            if result.get("success"):
                try:
                    updated_last_sync = self.db.query(AttendanceSyncLog).filter(
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
                    
                except Exception as sync_info_error:
                    logger.error(f"Error fetching updated sync info: {sync_info_error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error syncing attendance for employee {employee_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Sync failed: {str(e)}"
            }
    
    async def recalculate_stats(self, employee_id: int, 
                              start_date: Optional[date] = None, 
                              end_date: Optional[date] = None) -> Dict[str, Any]:
        """Recalculate attendance statistics"""
        try:
            employee = self.get_employee_by_id(employee_id)
            
            # Use provided dates or default to current month
            if not start_date or not end_date:
                start_date, end_date = self.get_current_month_dates()
            
            logger.info(f"🧮 Recalculating stats for employee {employee_id}")
            logger.info(f"📊 Date range: {start_date} to {end_date}")
            
            result = await self.attendance_service.recalculate_employee_stats(
                employee_id=employee_id,
                start_date=start_date,
                end_date=end_date
            )
            
            logger.info(f"✅ Stats recalculated successfully for employee {employee_id}")
            
            return {
                'success': True,
                'summary': result['summary'],
                'calculation_details': result['calculation_details'],
                'records_analyzed': result['records_analyzed'],
                'message': 'Statistics recalculated successfully',
                'date_range': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'recalculated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"💥 Error recalculating stats for employee {employee_id}: {str(e)}")
            return {
                'success': False,
                'message': f'Recalculation failed: {str(e)}',
                'error_type': type(e).__name__
            }
    
    def add_attendance_exception(self, employee_id: int, exception_date: str, 
                               reason: str, category: str = "", current_user_id: int = None) -> Dict[str, Any]:
        """Add attendance exception for a specific date"""
        try:
            # Validate date
            try:
                parsed_date = datetime.strptime(exception_date, '%Y-%m-%d').date()
            except ValueError:
                return {
                    'success': False,
                    'message': 'Invalid date format'
                }
            
            # Validate reason
            if not reason or len(reason.strip()) < 5:
                return {
                    'success': False,
                    'message': 'Reason must be at least 5 characters long'
                }
            
            if len(reason.strip()) > 500:
                return {
                    'success': False,
                    'message': 'Reason must be less than 500 characters'
                }
            
            # Check if employee exists
            employee = self.get_employee_by_id(employee_id)
            
            # Use attendance service to add exception
            result = self.attendance_service.add_attendance_exception(
                employee_id=employee_id,
                date=parsed_date,
                reason=reason.strip(),
                created_by=current_user_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding attendance exception: {e}")
            return {
                'success': False,
                'message': 'Failed to add exception'
            }
    
    def remove_attendance_exception(self, employee_id: int, exception_date: str) -> Dict[str, Any]:
        """Remove attendance exception for a specific date"""
        try:
            # Validate date
            try:
                parsed_date = datetime.strptime(exception_date, '%Y-%m-%d').date()
            except ValueError:
                return {
                    'success': False,
                    'message': 'Invalid date format'
                }
            
            # Check if employee exists
            employee = self.get_employee_by_id(employee_id)
            
            # Use attendance service to remove exception
            result = self.attendance_service.remove_attendance_exception(
                employee_id=employee_id,
                date=parsed_date
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error removing attendance exception: {e}")
            return {
                'success': False,
                'message': 'Failed to remove exception'
            }
    
    async def test_biometric_connection(self) -> Dict[str, Any]:
        """Test connection to biometric API"""
        try:
            logger.info("Testing biometric API connection")
            
            async with BiometricAPIService() as api:
                # Test authentication
                if not api.token:
                    return {
                        'success': False,
                        'message': 'Failed to authenticate with biometric API',
                        'api_status': 'Authentication Failed'
                    }
                
                # Test basic data fetch
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
    
    def format_date_display(self, date_obj: date) -> str:
        """Format date for display"""
        try:
            return date_obj.strftime('%B %d, %Y')
        except:
            return str(date_obj)
    
    def format_time_display(self, time_obj) -> str:
        """Format time for display"""
        try:
            if hasattr(time_obj, 'strftime'):
                return time_obj.strftime('%H:%M')
            return str(time_obj)
        except:
            return 'N/A'
    
    def get_filter_summary_text(self, start_date: date, end_date: date) -> str:
        """Generate filter summary text"""
        if start_date == end_date:
            return f"Showing data for {self.format_date_display(start_date)}"
        else:
            return f"Showing data from {self.format_date_display(start_date)} to {self.format_date_display(end_date)}"
    
    def calculate_attendance_trends(self, records: List[Dict]) -> Dict[str, Any]:
        """Calculate attendance trends and insights"""
        if not records:
            return {
                'trend': 'No data',
                'insights': [],
                'recommendations': []
            }
        
        # Calculate weekly trends
        weekly_attendance = {}
        for record in records:
            week = record['date'].isocalendar()[1]  # Week number
            if week not in weekly_attendance:
                weekly_attendance[week] = {'present': 0, 'total': 0}
            
            weekly_attendance[week]['total'] += 1
            if record['is_present']:
                weekly_attendance[week]['present'] += 1
        
        # Analyze trends
        insights = []
        recommendations = []
        
        # Check for declining attendance
        if len(weekly_attendance) >= 2:
            weeks = sorted(weekly_attendance.keys())
            recent_weeks = weeks[-2:]
            
            if len(recent_weeks) == 2:
                prev_week_rate = (weekly_attendance[recent_weeks[0]]['present'] / 
                                weekly_attendance[recent_weeks[0]]['total'] * 100)
                curr_week_rate = (weekly_attendance[recent_weeks[1]]['present'] / 
                                weekly_attendance[recent_weeks[1]]['total'] * 100)
                
                if curr_week_rate < prev_week_rate - 10:
                    insights.append("📉 Attendance declining this week")
                    recommendations.append("Consider checking with employee about any issues")
                elif curr_week_rate > prev_week_rate + 10:
                    insights.append("📈 Attendance improving this week")
        
        # Check for punctuality issues
        late_count = sum(1 for record in records if record.get('is_late', False))
        if late_count > len(records) * 0.3:  # More than 30% late
            insights.append("⏰ Frequent late arrivals detected")
            recommendations.append("Review work schedule or discuss punctuality")
        
        return {
            'insights': insights,
            'recommendations': recommendations,
            'weekly_data': weekly_attendance
        }

    def render_attendance_table_fragment(self, request, employee_id: int, 
                                       start_date: date, end_date: date) -> str:
        """Render just the attendance table fragment for HTMX updates"""
        try:
            attendance_data = self.get_attendance_data(employee_id, start_date, end_date)
            
            context = {
                'request': request,
                'records': attendance_data['records'],
                'summary': attendance_data['summary'],
                'employee': attendance_data['employee'],
                'start_date': start_date,
                'end_date': end_date,
                'filter_summary': self.get_filter_summary_text(start_date, end_date)
            }
            
            return templates.TemplateResponse(
                "staff/attendance/fragments/attendance_table.html", 
                context
            ).body.decode()
            
        except Exception as e:
            logger.error(f"Error rendering table fragment: {str(e)}")
            return f'<div class="error">Error loading data: {str(e)}</div>'

    def render_stats_fragment(self, request, employee_id: int, 
                            start_date: date, end_date: date) -> str:
        """Render just the stats cards fragment for HTMX updates"""
        try:
            attendance_data = self.get_attendance_data(employee_id, start_date, end_date)
            
            context = {
                'request': request,
                'summary': attendance_data['summary'],
                'employee': attendance_data['employee']
            }
            
            return templates.TemplateResponse(
                "staff/attendance/fragments/stats_cards.html", 
                context
            ).body.decode()
            
        except Exception as e:
            logger.error(f"Error rendering stats fragment: {str(e)}")
            return f'<div class="error">Error loading stats: {str(e)}</div>'