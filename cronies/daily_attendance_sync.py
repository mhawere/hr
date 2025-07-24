#!/usr/bin/env python3
"""
Daily Attendance Sync Script - PRODUCTION VERSION
Safely syncs biometric attendance data for all employees with intelligent date range calculation
Works with existing database setup - no migrations required
Run via cron job: 0 6 * * * cd /hr && /usr/bin/python3 cronies/daily_attendance_sync.py
"""

import os
import sys
import logging
import asyncio
import traceback
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# Find the project root (where hr_system.db is located)
script_dir = Path(__file__).parent  # /hr/cronies
project_root = script_dir.parent    # /hr
sys.path.insert(0, str(project_root))

# CRITICAL: Change working directory to project root
os.chdir(project_root)

print(f"🏗️ Script directory: {script_dir}")
print(f"🏗️ Project root: {project_root}")
print(f"🏗️ Current working directory: {os.getcwd()}")
print(f"📊 Database file exists: {(project_root / 'hr_system.db').exists()}")

# Suppress Pydantic warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Import your existing modules - Import ALL models to avoid relationship issues
try:
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy import create_engine, desc, text
    
    # Import database and core models first
    from models.database import DATABASE_URL, Base
    print(f"✅ Using database URL: {DATABASE_URL}")
    
    from models.employee import Employee, User, Department
    from models.attendance import AttendanceSyncLog, ProcessedAttendance, RawAttendance
    from models.shift import Shift
    from models.performance import PerformanceRecord, Badge, EmployeeBadge
    from models.leave import Leave, LeaveType, LeaveBalance, PublicHoliday
    from models.report_template import ReportTemplate
    
    # Try to import StatusEnum, but provide fallback
    try:
        from models.employee import StatusEnum
    except ImportError:
        # Create a simple StatusEnum fallback
        import enum
        class StatusEnum(enum.Enum):
            ACTIVE = "Active"
            NOT_ACTIVE = "Not Active"
    
    # Import services - but avoid importing routers that might have conflicting models
    from services.biometric_service import BiometricAPIService
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script from the correct directory")
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Current directory: {os.getcwd()}")
    sys.exit(1)

# ✅ PRODUCTION CONFIGURATION - Updated per requirements
SCRIPT_NAME = "daily_attendance_sync"
MAX_CONCURRENT_SYNCS = 1  # ✅ Reduced to 1 as requested
DELAY_BETWEEN_EMPLOYEES = 3  # ✅ 3 seconds as requested
RETRY_FAILED_AFTER_HOURS = 6
MAX_DAYS_BACK = 3  # ✅ Max 3 days as requested
LOG_RETENTION_DAYS = 30

def verify_database_setup():
    """Verify database setup and connection"""
    try:
        print("=" * 100)
        print("🔍 DATABASE SETUP VERIFICATION")
        print("=" * 100)
        
        from models.database import DATABASE_URL, engine
        
        # Check database file
        project_root = Path(os.getcwd())
        db_file = project_root / "hr_system.db"
        
        print(f"📁 Project root: {project_root}")
        print(f"📊 Database file: {db_file}")
        print(f"📊 Database exists: {db_file.exists()}")
        
        if db_file.exists():
            file_size = db_file.stat().st_size
            print(f"📊 Database size: {file_size:,} bytes")
            print(f"📊 Database readable: {os.access(db_file, os.R_OK)}")
            print(f"📊 Database writable: {os.access(db_file, os.W_OK)}")
        
        print(f"🔗 Database URL: {DATABASE_URL}")
        
        # Test connection
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        try:
            # Test query
            result = db.execute(text("SELECT COUNT(*) FROM employees")).scalar()
            print(f"👥 Total employees: {result}")
            
            # Count attendance records
            raw_count = db.execute(text("SELECT COUNT(*) FROM raw_attendance")).scalar()
            processed_count = db.execute(text("SELECT COUNT(*) FROM processed_attendance")).scalar()
            
            print(f"📋 Raw attendance records: {raw_count}")
            print(f"📋 Processed attendance records: {processed_count}")
            
            # Show recent processed records
            recent = db.execute(text("""
                SELECT date, employee_id, status, is_present, created_at 
                FROM processed_attendance 
                ORDER BY created_at DESC 
                LIMIT 5
            """)).fetchall()
            
            print(f"📋 Recent processed records:")
            for record in recent:
                print(f"   {record[0]} - Employee {record[1]} - {record[2]} - Present: {record[3]} - Created: {record[4]}")
            
            # Show sync logs
            sync_logs = db.execute(text("""
                SELECT employee_id, emp_code, last_sync_date, sync_status, records_fetched, records_processed, created_at
                FROM attendance_sync_log 
                ORDER BY created_at DESC 
                LIMIT 3
            """)).fetchall()
            
            print(f"📋 Recent sync logs:")
            for log in sync_logs:
                print(f"   Employee {log[0]} ({log[1]}) - {log[2]} - {log[3]} - Fetched: {log[4]}, Processed: {log[5]} - {log[6]}")
            
            print("=" * 100)
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        print(f"🔍 Error details: {traceback.format_exc()}")
        return False

class MinimalAttendanceService:
    """Minimal attendance service for sync operations - PRODUCTION VERSION"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
    
    async def sync_employee_attendance_range(self, employee_id: int, start_date: date, end_date: date) -> Dict[str, Any]:
        """Minimal sync implementation - calls the biometric API directly"""
        try:
            self.logger.info(f"🔄 Starting sync for employee {employee_id} from {start_date} to {end_date}")
            
            # Get employee info
            employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
            if not employee or not employee.biometric_id:
                return {
                    "success": False,
                    "message": "Employee not found or no biometric ID"
                }
            
            # Use biometric service to fetch data
            async with BiometricAPIService() as api:
                # Calculate days back from today to start_date
                days_back = (date.today() - start_date).days + 1
                
                self.logger.info(f"📅 Fetching {days_back} days of data for employee {employee.biometric_id}")
                
                # Fetch all recent data and filter for this employee
                all_data = await api.fetch_all_recent_data(days_back=days_back)
                
                # Filter data for this specific employee
                employee_data = [
                    record for record in all_data 
                    if record.get('emp_code') == employee.biometric_id
                ]
                
                records_fetched = len(employee_data)
                records_saved = 0
                
                self.logger.info(f"📊 Found {records_fetched} records for employee {employee.biometric_id}")
                
                if employee_data:
                    # Process and save the data
                    records_saved = await self._process_attendance_data(employee, employee_data, start_date)
                
                # NOW PROCESS DAILY ATTENDANCE FROM RAW DATA
                self.logger.info(f"📅 Processing daily attendance records from {start_date} to {end_date}")
                days_processed = await self._process_daily_attendance_for_employee(employee_id, start_date, end_date)
                self.logger.info(f"📅 Processed {days_processed} days of daily attendance")
                
                # Log the sync operation
                await self._log_sync_operation(employee_id, start_date, end_date, records_fetched, records_saved, "success")
                
                return {
                    "success": True,
                    "message": f"Synced {records_saved} raw records, processed {days_processed} days",
                    "records_fetched": records_fetched,
                    "records_saved": records_saved,
                    "days_processed": days_processed
                }
                
        except Exception as e:
            # Log failed sync
            await self._log_sync_operation(employee_id, start_date, end_date, 0, 0, "failed", str(e))
            
            return {
                "success": False,
                "message": str(e),
                "records_fetched": 0,
                "records_saved": 0
            }

    def _get_shift_type_value(self, shift) -> str:
        """
        ✅ PRODUCTION HELPER: Safely get shift type value - works with current database setup
        """
        try:
            if not shift:
                return 'standard'
            
            # Get the raw value directly from database to bypass enum issues
            raw_value = self.db.execute(
                text("SELECT shift_type FROM shifts WHERE id = :shift_id"),
                {"shift_id": shift.id}
            ).scalar()
            
            if raw_value:
                # Normalize to lowercase to match database values
                normalized = str(raw_value).lower()
                if normalized in ['standard', 'dynamic']:
                    return normalized
                else:
                    self.logger.warning(f"⚠️ Unknown shift type '{raw_value}', defaulting to 'standard'")
                    return 'standard'
            
            # Fallback to standard
            self.logger.warning(f"⚠️ No shift type found for shift {shift.id}, defaulting to 'standard'")
            return 'standard'
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get shift type for shift {getattr(shift, 'id', 'unknown')}: {e}")
            return 'standard'

    async def _process_attendance_data(self, employee: Employee, raw_data: List[Dict], sync_date: date) -> int:
        """Process raw attendance data and save to database"""
        processed_count = 0
        
        try:
            # This is a simplified version - you might need to adapt based on your actual data structure
            for record in raw_data:
                # Parse punch_time if it's a string
                punch_time = record.get('punch_time')
                if isinstance(punch_time, str):
                    try:
                        punch_time = datetime.fromisoformat(punch_time.replace('Z', '+00:00'))
                    except:
                        continue
                
                # Parse upload_time if it's a string
                upload_time = record.get('upload_time')
                if isinstance(upload_time, str):
                    try:
                        upload_time = datetime.fromisoformat(upload_time)
                    except:
                        upload_time = None
                
                # Save raw attendance record
                raw_attendance = RawAttendance(
                    biometric_record_id=record.get('id'),
                    emp_code=employee.biometric_id,
                    employee_id=employee.id,
                    punch_time=punch_time,
                    punch_state=record.get('punch_state', '0'),
                    punch_state_display=record.get('punch_state_display', ''),
                    verify_type=record.get('verify_type'),
                    verify_type_display=record.get('verify_type_display', ''),
                    terminal_sn=record.get('terminal_sn', ''),
                    terminal_alias=record.get('terminal_alias', ''),
                    area_alias=record.get('area_alias', ''),
                    temperature=record.get('temperature'),
                    upload_time=upload_time,  # Now properly parsed
                    created_at=datetime.now()
                )
                
                # Check if record already exists
                existing = self.db.query(RawAttendance).filter(
                    RawAttendance.biometric_record_id == raw_attendance.biometric_record_id
                ).first()
                
                if not existing:
                    self.db.add(raw_attendance)
                    processed_count += 1
            
            # Commit the changes
            self.db.commit()
            
            return processed_count
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"❌ Error processing attendance data: {e}")
            return 0
    
    async def _process_daily_attendance_for_employee(self, employee_id: int, start_date: date, end_date: date) -> int:
        """Process daily attendance records for an employee - PRODUCTION VERSION"""
        
        self.logger.info(f"📅 Processing daily attendance for employee {employee_id}")
        self.logger.info(f"📊 Date range: {start_date} to {end_date}")
        
        # Don't process future dates
        today = date.today()
        if end_date > today:
            end_date = today
            self.logger.info(f"📅 Adjusted end date to today: {end_date} (excluded future dates)")
        
        # Skip entirely if start date is in the future
        if start_date > today:
            self.logger.info(f"⏭️ Skipping processing - all dates are in the future")
            return 0
        
        days_processed = 0
        current_date = start_date
        
        while current_date <= end_date:
            self.logger.info(f"📅 Processing day: {current_date} ({current_date.strftime('%A')})")
            
            try:
                await self._process_single_day_attendance(employee_id, current_date)
                days_processed += 1
                self.logger.debug(f"✅ Day {current_date} processed successfully")
            except Exception as day_error:
                self.logger.error(f"💥 Error processing {current_date}: {day_error}")
                # Continue processing other days
            
            current_date += timedelta(days=1)
        
        self.logger.info(f"📊 Daily processing complete: {days_processed} days processed")
        return days_processed
    
    async def _process_single_day_attendance(self, employee_id: int, target_date: date):
        """Process attendance for a single day - PRODUCTION VERSION with enum handling"""
        
        # Don't process future dates
        today = date.today()
        if target_date > today:
            self.logger.info(f"⏭️ Skipping {target_date} - future date")
            return
        
        self.logger.info(f"📅 Processing attendance for {target_date}")
        
        # Get employee info
        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            self.logger.error(f"❌ Employee {employee_id} not found during day processing")
            return
        
        # Check for approved leave FIRST
        from sqlalchemy import and_
        from models.leave import LeaveStatusEnum
        
        approved_leave = self.db.query(Leave).filter(
            and_(
                Leave.employee_id == employee_id,
                Leave.start_date <= target_date,
                Leave.end_date >= target_date,
                Leave.status == LeaveStatusEnum.ACTIVE
            )
        ).first()
        
        # If on leave, create leave record and RETURN immediately
        if approved_leave:
            self.logger.info(f"🏖️ Employee is on approved leave")
            
            self._create_or_update_processed_attendance(
                employee_id=employee_id,
                target_date=target_date,
                shift_id=employee.shift.id if employee.shift else None,
                status="on_leave",
                is_present=True,  # Leave counts as present for attendance rate
                leave_id=approved_leave.id,
                leave_type=approved_leave.leave_type.name if approved_leave.leave_type else "Leave"
            )
            return  # Exit here - don't process attendance records
        
        # Check for public holidays
        public_holiday = self.db.query(PublicHoliday).filter(
            and_(
                PublicHoliday.date == target_date,
                PublicHoliday.is_active == True
            )
        ).first()
        
        # If public holiday, create holiday record and RETURN immediately
        if public_holiday:
            self.logger.info(f"🎉 Public holiday: {public_holiday.name}")
            
            self._create_or_update_processed_attendance(
                employee_id=employee_id,
                target_date=target_date,
                shift_id=employee.shift.id if employee.shift else None,
                status="public_holiday",
                is_present=True,  # Public holidays count as present
                holiday_name=public_holiday.name
            )
            return  # Exit here
        
        # Get raw attendance records for this day
        from datetime import time
        start_datetime = datetime.combine(target_date, time.min)
        end_datetime = datetime.combine(target_date, time.max)
        
        raw_records = self.db.query(RawAttendance).filter(
            and_(
                RawAttendance.employee_id == employee_id,
                RawAttendance.punch_time >= start_datetime,
                RawAttendance.punch_time <= end_datetime
            )
        ).order_by(RawAttendance.punch_time).all()
        
        self.logger.info(f"📋 Found {len(raw_records)} raw attendance records for {target_date}")
        
        if len(raw_records) == 0:
            # No attendance records - determine if day off or absent
            shift_id = None
            is_day_off = False
            
            if employee.shift:
                shift = employee.shift
                shift_id = shift.id
                weekday = target_date.weekday()  # 0=Monday, 6=Sunday
                
                # ✅ PRODUCTION FIX: Handle shift_type safely regardless of enum vs string
                shift_type_value = self._get_shift_type_value(shift)
                
                # Check if this is a scheduled work day
                if shift_type_value == 'standard':
                    if weekday >= 5:  # Weekend
                        if weekday == 5:  # Saturday
                            if not (shift.saturday_start and shift.saturday_end):
                                is_day_off = True
                        else:  # Sunday
                            if not (shift.sunday_start and shift.sunday_end):
                                is_day_off = True
                elif shift_type_value == 'dynamic':  # Dynamic shift
                    day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                    day_name = day_names[weekday]
                    shift_start = getattr(shift, f'{day_name}_start', None)
                    shift_end = getattr(shift, f'{day_name}_end', None)
                    
                    if not (shift_start and shift_end):
                        is_day_off = True
            
            if is_day_off:
                self.logger.info(f"📅 Marking {target_date} as DAY OFF")
                status = "day_off"
                is_present = True  # Day off counts as present
            else:
                # Check if this is today
                if target_date == today:
                    self.logger.info(f"⏳ No attendance records yet today - marking as PENDING")
                    status = "pending"
                    is_present = False
                else:
                    self.logger.info(f"❌ No attendance records for past day - marking as ABSENT")
                    status = "absent"
                    is_present = False
            
            self._create_or_update_processed_attendance(
                employee_id=employee_id,
                target_date=target_date,
                shift_id=shift_id,
                status=status,
                is_present=is_present
            )
            return
        
        # Process punch records - SIMPLIFIED
        check_ins = [r for r in raw_records if r.punch_state == "0"]
        check_outs = [r for r in raw_records if r.punch_state == "1"]
        
        check_in_time = check_ins[0].punch_time if check_ins else None
        check_out_time = check_outs[-1].punch_time if check_outs else None
        
        # Calculate basic working time
        total_minutes = 0
        if check_in_time and check_out_time:
            working_duration = check_out_time - check_in_time
            total_minutes = int(working_duration.total_seconds() / 60)
        
        # Determine status
        status = "present"
        is_present = True
        
        if check_in_time and not check_out_time:
            if target_date == today:
                status = "in_progress"
            else:
                status = "half_day"
        elif not check_in_time and check_out_time:
            status = "half_day"
        
        # Basic shift information
        shift_id = employee.shift.id if employee.shift else None
        
        self._create_or_update_processed_attendance(
            employee_id=employee_id,
            target_date=target_date,
            check_in_time=check_in_time,
            check_out_time=check_out_time,
            shift_id=shift_id,
            total_working_minutes=total_minutes,
            total_working_hours=round(total_minutes / 60, 2),
            is_present=is_present,
            status=status
        )
        
        self.logger.info(f"✅ Processed attendance record saved for {target_date}")

    def _create_or_update_processed_attendance(self, employee_id: int, target_date: date, **kwargs):
        """Create or update processed attendance record"""
        
        # Check if record already exists
        from sqlalchemy import and_
        existing = self.db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.employee_id == employee_id,
                ProcessedAttendance.date == target_date
            )
        ).first()
        
        if existing:
            self.logger.info(f"🔄 Updating existing processed record for {target_date}")
            
            # Update fields
            for key, value in kwargs.items():
                setattr(existing, key, value)
            
            existing.updated_at = datetime.utcnow()
        else:
            self.logger.info(f"💾 Creating new processed record for {target_date}")
            
            employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
            processed_attendance = ProcessedAttendance(
                employee_id=employee_id,
                emp_code=employee.biometric_id if employee else "",
                date=target_date,
                **kwargs
            )
            self.db.add(processed_attendance)
        
        try:
            self.db.commit()
            self.logger.debug(f"✅ Processed attendance record committed for {target_date}")
        except Exception as commit_error:
            self.logger.error(f"💥 Failed to commit processed attendance: {commit_error}")
            self.db.rollback()
            raise commit_error
    
    async def _log_sync_operation(self, employee_id: int, start_date: date, end_date: date, 
                                records_fetched: int, records_saved: int, status: str, error_msg: str = None):
        """Log the sync operation"""
        try:
            employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
            
            sync_log = AttendanceSyncLog(
                employee_id=employee_id,
                emp_code=employee.biometric_id if employee else "unknown",
                last_sync_date=datetime.now().date(),
                sync_start_date=start_date,
                sync_end_date=end_date,
                records_fetched=records_fetched,
                records_processed=records_saved,
                sync_status=status,
                error_message=error_msg,
                sync_duration_seconds=0,  # Could calculate this if needed
                created_at=datetime.now()
            )
            
            self.db.add(sync_log)
            self.db.commit()
            
        except Exception as e:
            self.logger.error(f"❌ Error logging sync operation: {e}")
            self.db.rollback()

class SyncLogger:
    """Enhanced logging for the sync script"""
    
    def __init__(self):
        # Create logs directory in the project root if /var/log doesn't exist
        if os.path.exists("/var/log") and os.access("/var/log", os.W_OK):
            log_dir = Path("/var/log/attendance_sync")
        else:
            log_dir = Path("./logs")  # This will be in project root now
        
        log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = log_dir / f"{SCRIPT_NAME}_{date.today().isoformat()}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)  # Also log to stdout for cron
            ]
        )
        
        self.logger = logging.getLogger(SCRIPT_NAME)
        self.start_time = datetime.now()
        self.log_dir = log_dir
        
        # Cleanup old logs
        self._cleanup_old_logs(log_dir)
    
    def _cleanup_old_logs(self, log_dir: Path):
        """Remove old log files"""
        try:
            cutoff_date = date.today() - timedelta(days=LOG_RETENTION_DAYS)
            for log_file in log_dir.glob(f"{SCRIPT_NAME}_*.log"):
                try:
                    file_date_str = log_file.stem.split('_')[-1]
                    file_date = date.fromisoformat(file_date_str)
                    if file_date < cutoff_date:
                        log_file.unlink()
                        self.logger.info(f"🗑️ Cleaned up old log: {log_file.name}")
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cleanup old logs: {e}")
    
    def info(self, message: str):
        self.logger.info(message)
        
    def warning(self, message: str):
        self.logger.warning(message)
        
    def error(self, message: str):
        self.logger.error(message)
    
    def get_duration(self) -> float:
        """Get script duration in minutes"""
        return (datetime.now() - self.start_time).total_seconds() / 60

class AttendanceSyncScript:
    """Main sync script class - PRODUCTION VERSION"""
    
    def __init__(self):
        self.logger = SyncLogger()
        self.db_engine = None
        self.session = None
        
        # Script results
        self.results = {
            "started_at": datetime.now(),
            "total_employees": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "skipped_syncs": 0,
            "api_errors": 0,
            "errors": [],
            "employee_results": {}
        }
    
    def setup_database(self) -> bool:
        """Setup database connection"""
        try:
            self.logger.info(f"📂 Current directory: {os.getcwd()}")
            self.logger.info(f"🗄️ Database URL: {DATABASE_URL}")
            
            # Check if database file exists
            if DATABASE_URL.startswith("sqlite:///"):
                db_file = DATABASE_URL.replace("sqlite:///", "")
                if not db_file.startswith("/"):  # Relative path
                    db_file = os.path.join(os.getcwd(), db_file)
                
                self.logger.info(f"🔍 Looking for database at: {db_file}")
                
                if os.path.exists(db_file):
                    self.logger.info(f"✅ Database file found: {db_file}")
                    file_size = os.path.getsize(db_file)
                    self.logger.info(f"📊 Database size: {file_size:,} bytes")
                else:
                    self.logger.error(f"❌ Database file not found: {db_file}")
                    return False
            
            self.db_engine = create_engine(DATABASE_URL)
            SessionLocal = sessionmaker(bind=self.db_engine)
            self.session = SessionLocal()
            
            # Test connection with proper text() wrapper
            result = self.session.execute(text("SELECT 1")).fetchone()
            self.logger.info("✅ Database connection established")
            
            # Additional verification
            employee_count = self.session.execute(text("SELECT COUNT(*) FROM employees")).scalar()
            self.logger.info(f"👥 Found {employee_count} employees in database")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            self.logger.error(f"📍 Traceback: {traceback.format_exc()}")
            return False
    
    def cleanup_database(self):
        """Cleanup database connection"""
        if self.session:
            self.session.close()
        if self.db_engine:
            self.db_engine.dispose()
    
    async def test_api_connection(self) -> bool:
        """Test biometric API availability"""
        try:
            self.logger.info("🔍 Testing API connection...")
            async with BiometricAPIService() as api:
                test_result = await api.test_connection()
                if test_result.get('success'):
                    self.logger.info("✅ API connection successful")
                    return True
                else:
                    self.logger.error(f"❌ API test failed: {test_result.get('message', 'Unknown error')}")
                    return False
        except Exception as e:
            self.logger.error(f"❌ API connection error: {e}")
            return False
    
    def calculate_sync_date_range(self, employee: Employee) -> tuple[date, date]:
        """Calculate intelligent sync date range - OPTIMIZED VERSION"""
        
        today = date.today()
        yesterday = today - timedelta(days=1)
        max_days_back = MAX_DAYS_BACK
        
        # Get last successful sync
        last_sync = self.session.query(AttendanceSyncLog).filter(
            AttendanceSyncLog.employee_id == employee.id,
            AttendanceSyncLog.sync_status == "success"
        ).order_by(desc(AttendanceSyncLog.created_at)).first()
        
        # ✅ NEW: Check for very recent processed data (yesterday)
        has_yesterday_data = self.session.query(ProcessedAttendance).filter(
            ProcessedAttendance.employee_id == employee.id,
            ProcessedAttendance.date == yesterday
        ).first() is not None
        
        # ✅ OPTIMIZATION: If we have yesterday's data and recent sync, minimal sync
        if last_sync and has_yesterday_data:
            hours_since_sync = (datetime.now() - last_sync.created_at).total_seconds() / 3600
            
            if hours_since_sync < 24:  # Synced within last 24 hours
                start_date = yesterday
                reason = "Recent sync with current data - minimal sync (2 days)"
                return start_date, today
        
        # ✅ FALLBACK: Regular logic for other cases
        has_any_data = self.session.query(ProcessedAttendance).filter(
            ProcessedAttendance.employee_id == employee.id
        ).first() is not None
        
        three_days_ago = today - timedelta(days=max_days_back)
        has_recent_data = self.session.query(ProcessedAttendance).filter(
            ProcessedAttendance.employee_id == employee.id,
            ProcessedAttendance.date >= three_days_ago
        ).first() is not None
        
        if not has_any_data:
            start_date = three_days_ago
            reason = "No existing data - initial sync (4 days)"
        elif not has_recent_data:
            start_date = three_days_ago
            reason = "Missing recent data - recovery sync (4 days)"
        elif last_sync and last_sync.sync_end_date:
            days_since_sync = (today - last_sync.sync_end_date).days
            
            if days_since_sync <= 1:
                start_date = yesterday
                reason = "Regular incremental sync (2 days)"
            else:
                # Gap exists but limit to max days
                potential_start = last_sync.sync_end_date - timedelta(days=1)
                earliest_allowed = three_days_ago
                start_date = max(potential_start, earliest_allowed)
                actual_days = (today - start_date).days + 1
                reason = f"Gap recovery - {actual_days} days (limited to {max_days_back + 1})"
        else:
            start_date = three_days_ago
            reason = "Fallback sync (4 days)"
        
        days_to_sync = (today - start_date).days + 1
        self.logger.info(f"👤 {employee.employee_id}: {reason}")
        
        return start_date, today
    
    def get_employees_for_sync(self) -> List[Employee]:
        """Get employees that need syncing - IMPROVED PRODUCTION VERSION"""
        try:
            # Get all active employees with biometric IDs - use more flexible status check
            all_employees = self.session.query(Employee).filter(
                Employee.biometric_id.isnot(None),
                Employee.biometric_id != "",
                Employee.status == StatusEnum.ACTIVE  # Use the imported StatusEnum
            ).all()
            
            self.logger.info(f"🔍 Found {len(all_employees)} active employees with biometric IDs")
            
            employees_to_sync = []
            cutoff_time = datetime.now() - timedelta(hours=RETRY_FAILED_AFTER_HOURS)
            today = date.today()
            yesterday = today - timedelta(days=1)
            
            for employee in all_employees:
                try:
                    last_sync = self.session.query(AttendanceSyncLog).filter(
                        AttendanceSyncLog.employee_id == employee.id
                    ).order_by(desc(AttendanceSyncLog.created_at)).first()
                    
                    should_sync = False
                    reason = ""
                    
                    if not last_sync:
                        should_sync = True
                        reason = "Never synced"
                    elif last_sync.sync_status == "failed" and last_sync.created_at < cutoff_time:
                        should_sync = True
                        reason = f"Retry failed sync from {last_sync.created_at.strftime('%Y-%m-%d %H:%M')}"
                    elif last_sync.sync_status == "success":
                        hours_since_sync = (datetime.now() - last_sync.created_at).total_seconds() / 3600
                        
                        # ✅ PRODUCTION: Check if we actually have processed records for recent dates
                        has_recent_processed = self.session.query(ProcessedAttendance).filter(
                            ProcessedAttendance.employee_id == employee.id,
                            ProcessedAttendance.date >= yesterday
                        ).first() is not None
                        
                        # NEW LOGIC: If "successful" sync got 0 records, treat as needing retry
                        if last_sync.records_fetched == 0 and last_sync.records_processed == 0:
                            should_sync = True
                            reason = f"Previous sync got no data - retrying"
                        elif not has_recent_processed and hours_since_sync < 48:  # Within 48 hours but no processed records
                            should_sync = True
                            reason = f"Sync marked successful but no processed records found"
                        elif hours_since_sync > 20:  # Normal successful sync logic
                            should_sync = True
                            reason = f"Last sync {hours_since_sync:.1f} hours ago"
                    
                    if should_sync:
                        employees_to_sync.append(employee)
                        self.logger.info(f"📋 {employee.employee_id}: {reason}")
                    else:
                        self.results["skipped_syncs"] += 1
                        self.logger.info(f"⏭️ {employee.employee_id}: Skipped (recent successful sync with data)")
                        
                except Exception as emp_error:
                    self.logger.warning(f"⚠️ Error checking employee {employee.employee_id}: {emp_error}")
                    continue
            
            self.logger.info(f"📊 Found {len(employees_to_sync)} employees to sync, {self.results['skipped_syncs']} skipped")
            return employees_to_sync
            
        except Exception as e:
            self.logger.error(f"❌ Error getting employees: {e}")
            self.logger.error(f"📍 Traceback: {traceback.format_exc()}")
            return []
    
    async def sync_single_employee(self, employee: Employee, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Sync a single employee with intelligent date range"""
        async with semaphore:
            employee_result = {
                "employee_id": employee.employee_id,
                "employee_name": f"{employee.first_name} {employee.last_name}",
                "success": False,
                "message": "",
                "records_synced": 0,
                "sync_time": datetime.now(),
                "date_range": None
            }
            
            try:
                # Calculate intelligent date range
                start_date, end_date = self.calculate_sync_date_range(employee)
                days_to_sync = (end_date - start_date).days + 1
                employee_result["date_range"] = f"{start_date} to {end_date} ({days_to_sync} days)"
                
                self.logger.info(f"🔄 Syncing {employee.employee_id}: {employee_result['date_range']}")
                
                # Use our minimal attendance service
                attendance_service = MinimalAttendanceService(self.session)
                
                # Use range sync with calculated dates
                sync_result = await attendance_service.sync_employee_attendance_range(
                    employee_id=employee.id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if sync_result.get("success"):
                    employee_result["success"] = True
                    employee_result["message"] = sync_result.get("message", "Success")
                    employee_result["records_synced"] = sync_result.get("records_saved", 0)
                    
                    self.results["successful_syncs"] += 1
                    self.logger.info(f"✅ {employee.employee_id}: {employee_result['records_synced']} records")
                else:
                    employee_result["message"] = sync_result.get("message", "Unknown error")
                    self.results["failed_syncs"] += 1
                    self.results["errors"].append(f"{employee.employee_id}: {employee_result['message']}")
                    self.logger.error(f"❌ {employee.employee_id}: {employee_result['message']}")
                
                # Rate limiting - ✅ 3 seconds as requested
                await asyncio.sleep(DELAY_BETWEEN_EMPLOYEES)
                
            except Exception as e:
                employee_result["message"] = f"Exception: {str(e)}"
                self.results["failed_syncs"] += 1
                self.results["errors"].append(f"{employee.employee_id}: {str(e)}")
                self.logger.error(f"💥 {employee.employee_id}: {str(e)}")
            
            self.results["employee_results"][employee.employee_id] = employee_result
            return employee_result
    
    async def run_sync(self) -> Dict[str, Any]:
        """Run the main sync process - PRODUCTION VERSION"""
        self.logger.info("🚀 Starting daily attendance sync")
        self.logger.info(f"⚙️ Config: Max {MAX_CONCURRENT_SYNCS} concurrent, {DELAY_BETWEEN_EMPLOYEES}s delay, {MAX_DAYS_BACK} days max")
        
        try:
            # Setup database
            if not self.setup_database():
                self.results["errors"].append("Database setup failed")
                return self.results
            
            # Test API
            if not await self.test_api_connection():
                self.results["errors"].append("API connection failed")
                self.results["api_errors"] += 1
                return self.results
            
            # Get employees to sync
            employees = self.get_employees_for_sync()
            self.results["total_employees"] = len(employees)
            
            if not employees:
                self.logger.info("ℹ️ No employees need syncing")
                return self.results
            
            # Create semaphore for rate limiting - ✅ 1 concurrent as requested
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYNCS)
            
            # Create sync tasks
            tasks = [
                self.sync_single_employee(employee, semaphore)
                for employee in employees
            ]
            
            # Execute all sync tasks
            self.logger.info(f"⚡ Starting sync for {len(tasks)} employees...")
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Final summary
            duration = self.logger.get_duration()
            self.results["completed_at"] = datetime.now()
            self.results["duration_minutes"] = round(duration, 2)
            
            self.logger.info(f"🎉 Sync completed in {duration:.1f} minutes")
            self.logger.info(f"📊 Results: ✅{self.results['successful_syncs']} ❌{self.results['failed_syncs']} ⏭️{self.results['skipped_syncs']}")
            
            # Log any errors
            if self.results["errors"]:
                self.logger.warning(f"⚠️ {len(self.results['errors'])} errors occurred:")
                for error in self.results["errors"][:5]:  # Log first 5 errors
                    self.logger.warning(f"  - {error}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"💥 Critical sync error: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.results["errors"].append(f"Critical error: {str(e)}")
            return self.results
        
        finally:
            self.cleanup_database()
    
    def save_results_summary(self):
        """Save a summary file for monitoring"""
        try:
            summary_file = self.logger.log_dir / "last_sync_summary.json"
            
            import json
            
            # Create summary with key metrics
            summary = {
                "last_run": self.results["started_at"].isoformat(),
                "duration_minutes": self.results.get("duration_minutes", 0),
                "total_employees": self.results["total_employees"],
                "successful_syncs": self.results["successful_syncs"],
                "failed_syncs": self.results["failed_syncs"],
                "skipped_syncs": self.results["skipped_syncs"],
                "api_errors": self.results["api_errors"],
                "error_count": len(self.results["errors"]),
                "errors": self.results["errors"][:10],  # First 10 errors
                "success_rate": round(
                    (self.results["successful_syncs"] / max(self.results["total_employees"], 1)) * 100, 1
                ) if self.results["total_employees"] > 0 else 0,
                "employee_summary": {
                    emp_id: {
                        "success": result["success"],
                        "records_synced": result["records_synced"],
                        "date_range": result["date_range"],
                        "message": result["message"][:100] if result["message"] else ""
                    }
                    for emp_id, result in self.results["employee_results"].items()
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"📄 Summary saved to {summary_file}")
            self.logger.info(f"📈 Success rate: {summary['success_rate']}%")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save summary: {e}")

def main():
    """Main entry point"""
    # Verify database setup FIRST
    if not verify_database_setup():
        print("❌ Database verification failed - exiting")
        sys.exit(1)
    
    script = AttendanceSyncScript()
    
    try:
        # Check if another instance is running (simple file lock)
        lock_file = Path("/tmp/attendance_sync.lock")
        if lock_file.exists():
            script.logger.warning("⚠️ Another sync instance appears to be running (lock file exists)")
            # Check if lock file is stale (older than 2 hours)
            if (datetime.now().timestamp() - lock_file.stat().st_mtime) > 7200:
                script.logger.info("🔓 Removing stale lock file")
                lock_file.unlink()
            else:
                script.logger.error("❌ Exiting to avoid concurrent runs")
                sys.exit(1)
        
        # Create lock file
        lock_file.write_text(str(os.getpid()))
        script.logger.info(f"🔒 Created lock file with PID {os.getpid()}")
        
        try:
            # Run the sync
            results = asyncio.run(script.run_sync())
            
            # Save summary
            script.save_results_summary()
            
            # Final verification - show what was actually saved
            print("\n" + "=" * 100)
            print("🔍 FINAL VERIFICATION - CHECKING WHAT WAS SAVED")
            print("=" * 100)
            verify_database_setup()
            
            # Determine exit code based on results
            total_attempts = results["successful_syncs"] + results["failed_syncs"]
            
            if results["api_errors"] > 0:
                script.logger.error("❌ API errors occurred")
                sys.exit(2)  # API issues
            elif total_attempts == 0:
                script.logger.info("ℹ️ No sync attempts made")
                sys.exit(0)  # Nothing to do
            elif results["failed_syncs"] > results["successful_syncs"]:
                script.logger.error("❌ More failures than successes")
                sys.exit(3)  # Sync issues
            elif results["failed_syncs"] > 0:
                script.logger.warning("⚠️ Some syncs failed but majority succeeded")
                sys.exit(0)  # Partial success
            else:
                script.logger.info("✅ All syncs completed successfully")
                sys.exit(0)  # Success
                
        finally:
            # Remove lock file
            if lock_file.exists():
                lock_file.unlink()
                script.logger.info("🔓 Removed lock file")
    
    except KeyboardInterrupt:
        script.logger.info("⏹️ Sync interrupted by user")
        sys.exit(130)
    except Exception as e:
        script.logger.error(f"💥 Unexpected error: {str(e)}")
        script.logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()