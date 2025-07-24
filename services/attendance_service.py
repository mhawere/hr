"""
Enhanced Attendance Service - PRODUCTION READY v3.0 - FINAL VERSION
Business logic for processing attendance data with comprehensive logging and debugging

COMPLETE VERSION WITH ALL CRITICAL FIXES:
- Hour-based attendance calculation (matches manual calculation)
- Fixed lecturer weekend handling (day_off vs absent)
- Fixed public holiday processing with FAIR hour crediting
- Fixed shift time processing bugs
- Enhanced notes generation for late/early arrivals
- Proper incomplete record handling
- PUBLIC HOLIDAY FAIRNESS: Workers get full holiday credit even if they work shorter hours
- Comprehensive error handling and logging
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, time, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from models.employee import Employee
from models.attendance import RawAttendance, ProcessedAttendance, AttendanceSyncLog
from models.leave import Leave, LeaveStatusEnum, PublicHoliday, LeaveType
from models.shift import Shift
from services.biometric_service import BiometricAPIService
from fastapi import HTTPException
import logging
import traceback
import asyncio

logger = logging.getLogger(__name__)

class AttendanceService:
    
    LATE_ARRIVAL_GRACE_MINUTES = 25
    EARLY_DEPARTURE_THRESHOLD_MINUTES = 15

    def __init__(self, db: Session):
        self.db = db
        self.late_threshold_minutes = self.LATE_ARRIVAL_GRACE_MINUTES
        self.early_departure_threshold_minutes = self.EARLY_DEPARTURE_THRESHOLD_MINUTES
        logger.info("🏗️ AttendanceService initialized")
    
    def _is_lecturer(self, employee: Employee) -> bool:
        """Check if employee is a lecturer"""
        return employee.position and employee.position.lower() == 'lecturer'
    
    def _should_be_day_off(self, employee: Employee, target_date: date) -> bool:
        """Determine if a date should be marked as day_off for an employee"""
        
        # Check if employee is a lecturer
        is_lecturer = self._is_lecturer(employee)
        
        # For lecturers, weekends are automatically day_off
        if is_lecturer and target_date.weekday() >= 5:  # Weekend
            logger.info(f"📅 Lecturer weekend → DAY_OFF")
            return True
        
        # For regular employees, check weekend shift schedules
        weekday = target_date.weekday()
        
        if weekday >= 5:  # Weekend (Saturday=5, Sunday=6)
            if employee.shift:
                shift = employee.shift
                
                if shift.shift_type == 'standard':
                    if weekday == 5:  # Saturday
                        # Check if Saturday shift is defined
                        has_saturday_shift = (shift.saturday_start is not None and 
                                            shift.saturday_end is not None)
                        if not has_saturday_shift:
                            logger.info(f"📅 No Saturday shift defined → DAY_OFF")
                            return True
                        else:
                            logger.info(f"📅 Saturday shift defined: {shift.saturday_start}-{shift.saturday_end} → WORK DAY")
                            return False
                            
                    elif weekday == 6:  # Sunday
                        # Check if Sunday shift is defined
                        has_sunday_shift = (shift.sunday_start is not None and 
                                          shift.sunday_end is not None)
                        if not has_sunday_shift:
                            logger.info(f"📅 No Sunday shift defined → DAY_OFF")
                            return True
                        else:
                            logger.info(f"📅 Sunday shift defined: {shift.sunday_start}-{shift.sunday_end} → WORK DAY")
                            return False
                            
                else:  # Dynamic shift
                    day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                    day_name = day_names[weekday]
                    start_time = getattr(shift, f'{day_name}_start')
                    end_time = getattr(shift, f'{day_name}_end')
                    
                    # No shift schedule = day off
                    if not (start_time and end_time):
                        logger.info(f"📅 No {day_name} shift defined → DAY_OFF")
                        return True
                    else:
                        logger.info(f"📅 {day_name} shift defined: {start_time}-{end_time} → WORK DAY")
                        return False
            else:
                # No shift assigned = weekends are day off
                logger.info(f"📅 No shift assigned, weekend → DAY_OFF")
                return True
        
        return False

    async def sync_employee_attendance(
        self,
        employee_id: int,
        force_full_sync: bool = False
    ) -> Dict:
        """
        Sync attendance for a specific employee with comprehensive logging
        """
        
        logger.info("=" * 100)
        logger.info(f"🚀 STARTING ATTENDANCE SYNC")
        logger.info(f"👤 Employee ID: {employee_id}")
        logger.info(f"🔄 Force Full Sync: {force_full_sync}")
        logger.info(f"📅 Current Date: {date.today()}")
        logger.info("=" * 100)
        
        # Step 1: Validate Employee
        logger.info("📋 Step 1: Validating employee...")
        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        
        if not employee:
            logger.error(f"❌ Employee with ID {employee_id} not found")
            return {
                'success': False,
                'message': 'Employee not found',
                'error_code': 'EMPLOYEE_NOT_FOUND'
            }
        
        logger.info(f"✅ Employee found: {employee.first_name} {employee.last_name}")
        logger.info(f"📧 Email: {employee.email}")
        logger.info(f"🆔 Employee ID: {employee.employee_id}")
        logger.info(f"🔢 Biometric ID: '{employee.biometric_id}'")
        logger.info(f"💼 Position: {employee.position}")
        logger.info(f"👨‍🏫 Is Lecturer: {self._is_lecturer(employee)}")
        
        if not employee.biometric_id or not employee.biometric_id.strip():
            logger.error(f"❌ Employee {employee_id} has no biometric ID assigned")
            return {
                'success': False,
                'message': 'No biometric ID assigned to employee',
                'error_code': 'NO_BIOMETRIC_ID'
            }
        
        # Log employee details
        logger.info(f"🏢 Department: {employee.department.name if employee.department else 'None'}")
        logger.info(f"📊 Status: {employee.status}")
        logger.info(f"⏰ Shift: {employee.shift.name if employee.shift else 'None'}")
        
        if employee.shift:
            logger.info(f"🔧 Shift Type: {employee.shift.shift_type}")
            logger.info(f"🔧 Shift Active: {employee.shift.is_active}")
        else:
            logger.warning(f"⚠️ No shift assigned - this may affect processing")
        
        start_date = None
        end_date = None
        sync_type = None
        
        try:
            # Step 2: Determine Sync Range
            logger.info("📅 Step 2: Determining sync date range...")
            
            if force_full_sync:
                today = date.today()
                start_date = date(today.year, today.month, 1)
                end_date = today - timedelta(days=1)  # Exclude today
                sync_type = "full"
                logger.info(f"📅 Full month sync: {start_date} to {end_date}")
            else:
                yesterday = date.today() - timedelta(days=1)
                start_date = end_date = yesterday
                sync_type = "incremental"
                logger.info(f"📅 Incremental sync: {start_date}")
            
            logger.info(f"🎯 Sync Type: {sync_type}")
            logger.info(f"📊 Date Range: {start_date} to {end_date}")
            logger.info(f"📈 Days to sync: {(end_date - start_date).days + 1}")
            
            # Step 3: API Communication
            logger.info("🌐 Step 3: Communicating with biometric API...")
            
            raw_data = []
            records_saved = 0
            records_processed = 0
            records_duplicates = 0
            
            async with BiometricAPIService() as api:
                logger.info("✅ API connection established successfully")
                
                try:
                    raw_data = await api.fetch_attendance_data(
                        emp_code=employee.biometric_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    logger.info(f"📊 API Response Summary:")
                    logger.info(f"   Total records fetched: {len(raw_data)}")
                    
                    if len(raw_data) == 0:
                        logger.warning(f"⚠️ No attendance data found in API for employee {employee.biometric_id}")
                        logger.warning(f"   This could mean:")
                        logger.warning(f"   - Employee didn't punch during this period")
                        logger.warning(f"   - Biometric ID mismatch")
                        logger.warning(f"   - API data not available for this date range")
                    else:
                        # Log sample data for debugging
                        logger.info(f"📋 Sample records from API:")
                        for i, record in enumerate(raw_data[:3]):  # Show first 3 records
                            logger.info(f"   {i+1}. {record.get('punch_time')} - {record.get('punch_state_display', 'Unknown')}")
                        
                        if len(raw_data) > 3:
                            logger.info(f"   ... and {len(raw_data) - 3} more records")
                        
                        # Show date range of data
                        punch_times = [record.get('punch_time') for record in raw_data if record.get('punch_time')]
                        if punch_times:
                            logger.info(f"📅 Data date range: {min(punch_times)} to {max(punch_times)}")
                
                except Exception as api_error:
                    logger.error(f"💥 API fetch failed: {type(api_error).__name__}: {api_error}")
                    raise api_error
            
            # Step 4: Process Raw Data
            logger.info("💾 Step 4: Processing and saving raw attendance data...")
            
            if len(raw_data) > 0:
                logger.info(f"🔄 Processing {len(raw_data)} raw records...")
                
                for i, api_record in enumerate(raw_data, 1):
                    try:
                        logger.debug(f"🔄 Processing record {i}/{len(raw_data)}")
                        
                        # Parse the record
                        async with BiometricAPIService() as api:
                            parsed_record = api.parse_attendance_record(api_record)
                        
                        if not parsed_record:
                            logger.warning(f"⚠️ Failed to parse record {i}: {api_record}")
                            continue
                        
                        # Check for duplicates
                        existing = self.db.query(RawAttendance).filter(
                            RawAttendance.biometric_record_id == parsed_record['biometric_record_id']
                        ).first()
                        
                        if not existing:
                            # Create new raw attendance record
                            raw_attendance = RawAttendance(
                                employee_id=employee_id,
                                **parsed_record
                            )
                            self.db.add(raw_attendance)
                            records_saved += 1
                            
                            logger.debug(f"💾 Saved: {parsed_record['punch_time']} - {parsed_record.get('punch_state_display', 'Unknown')}")
                        else:
                            records_duplicates += 1
                            logger.debug(f"⏭️ Duplicate skipped: {parsed_record['punch_time']}")
                        
                        records_processed += 1
                        
                    except Exception as parse_error:
                        logger.error(f"💥 Error processing record {i}: {parse_error}")
                        logger.debug(f"Raw record: {api_record}")
                        continue
                
                # Commit raw data
                try:
                    self.db.commit()
                    logger.info(f"✅ Raw data committed to database")
                except Exception as commit_error:
                    logger.error(f"💥 Failed to commit raw data: {commit_error}")
                    self.db.rollback()
                    raise commit_error
                
                logger.info(f"📊 Raw Data Processing Summary:")
                logger.info(f"   Records processed: {records_processed}")
                logger.info(f"   New records saved: {records_saved}")
                logger.info(f"   Duplicates skipped: {records_duplicates}")
            else:
                logger.info(f"⏭️ No raw data to process")
            
            # Step 5: Process Daily Attendance
            logger.info("📅 Step 5: Processing daily attendance records...")
            
            days_processed = await self._process_daily_attendance_for_employee(
                employee_id, start_date, end_date
            )
            
            logger.info(f"📅 Processed {days_processed} days of attendance")
            
            # Step 6: Save Sync Log
            logger.info("📝 Step 6: Saving sync log...")
            
            sync_log = AttendanceSyncLog(
                employee_id=employee_id,
                emp_code=employee.biometric_id,
                last_sync_date=date.today(),
                sync_start_date=start_date,
                sync_end_date=end_date,
                records_fetched=len(raw_data),
                records_processed=records_processed,
                sync_status="success"
            )
            self.db.add(sync_log)
            
            # Final commit
            try:
                self.db.commit()
                logger.info(f"✅ Sync log saved successfully")
            except Exception as commit_error:
                logger.error(f"💥 Failed to save sync log: {commit_error}")
                self.db.rollback()
                raise commit_error
            
            # Step 7: Final Summary
            logger.info("=" * 100)
            logger.info(f"🎉 SYNC COMPLETED SUCCESSFULLY")
            logger.info(f"📊 Final Summary:")
            logger.info(f"   Employee: {employee.first_name} {employee.last_name}")
            logger.info(f"   Biometric ID: {employee.biometric_id}")
            logger.info(f"   Sync Type: {sync_type}")
            logger.info(f"   Date Range: {start_date} to {end_date}")
            logger.info(f"   API Records Fetched: {len(raw_data)}")
            logger.info(f"   New Records Saved: {records_saved}")
            logger.info(f"   Days Processed: {days_processed}")
            logger.info("=" * 100)
            
            return {
                'success': True,
                'message': f'{sync_type.capitalize()} sync completed successfully',
                'records_fetched': len(raw_data),
                'records_saved': records_saved,
                'days_processed': days_processed,
                'sync_type': sync_type,
                'employee_name': f"{employee.first_name} {employee.last_name}",
                'biometric_id': employee.biometric_id,
                'date_range': f"{start_date} to {end_date}"
            }
                    
        except HTTPException as api_error:
            logger.error(f"🌐 API error during sync: {api_error.detail}")
            self.db.rollback()
            
            # Log failed sync
            await self._log_failed_sync(employee_id, employee.biometric_id, start_date, end_date, str(api_error.detail))
            
            return {
                'success': False,
                'message': f'API error: {api_error.detail}',
                'error_code': 'API_ERROR'
            }
            
        except Exception as e:
            logger.error("=" * 100)
            logger.error(f"💥 SYNC FAILED - UNEXPECTED ERROR")
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {str(e)}")
            logger.error(f"Traceback:")
            logger.error(traceback.format_exc())
            logger.error("=" * 100)
            
            self.db.rollback()
            
            # Log failed sync
            await self._log_failed_sync(employee_id, employee.biometric_id, start_date, end_date, str(e))
            
            return {
                'success': False,
                'message': f'Sync failed: {str(e)}',
                'error_code': 'UNEXPECTED_ERROR',
                'error_type': type(e).__name__
            }

    async def sync_employee_attendance_range(
        self,
        employee_id: int,
        start_date: date,
        end_date: date
    ) -> Dict:
        """
        Sync attendance for a specific employee over a custom date range
        Excludes today and future dates - only processes completed days
        """
        
        logger.info("=" * 100)
        logger.info(f"🚀 STARTING RANGE ATTENDANCE SYNC")
        logger.info(f"👤 Employee ID: {employee_id}")
        logger.info(f"📅 Requested Date Range: {start_date} to {end_date}")
        logger.info(f"📊 Requested Total Days: {(end_date - start_date).days + 1}")
        logger.info("=" * 100)
        
        # Step 1: Validate Employee
        logger.info("📋 Step 1: Validating employee...")
        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        
        if not employee:
            logger.error(f"❌ Employee with ID {employee_id} not found")
            return {
                'success': False,
                'message': 'Employee not found',
                'error_code': 'EMPLOYEE_NOT_FOUND'
            }
        
        if not employee.biometric_id or not employee.biometric_id.strip():
            logger.error(f"❌ Employee {employee_id} has no biometric ID assigned")
            return {
                'success': False,
                'message': 'No biometric ID assigned to employee',
                'error_code': 'NO_BIOMETRIC_ID'
            }
        
        logger.info(f"✅ Employee: {employee.first_name} {employee.last_name} (ID: {employee.biometric_id})")
        logger.info(f"👨‍🏫 Is Lecturer: {self._is_lecturer(employee)}")
        
        # Step 2: Validate and adjust date range
        logger.info("📅 Step 2: Validating and adjusting date range...")
        
        today = date.today()
        yesterday = today - timedelta(days=1)
        original_end_date = end_date
        date_adjusted = False
        
        # Cap end date to exclude today and future dates
        if end_date >= today:
            end_date = yesterday
            date_adjusted = True
            
            logger.warning("=" * 80)
            logger.warning(f"⚠️  DATE RANGE AUTOMATIC ADJUSTMENT")
            logger.warning(f"📅 Requested end date: {original_end_date}")
            logger.warning(f"📅 Adjusted end date: {end_date}")
            logger.warning(f"🚫 Reason: Cannot process today ({today}) or future dates")
            logger.warning(f"📝 Range sync only processes completed days")
            logger.warning("=" * 80)
            
            # Validate that we still have a valid range
            if end_date < start_date:
                logger.error(f"❌ No valid dates to process after adjustment")
                logger.error(f"   Start date: {start_date}")
                logger.error(f"   Adjusted end date: {end_date}")
                logger.error(f"   All requested dates are today or in the future")
                
                return {
                    'success': False,
                    'message': f'No completed days available for processing. All requested dates are today or in the future.',
                    'error_code': 'NO_VALID_DATES',
                    'requested_range': f"{start_date} to {original_end_date}",
                    'today_date': today.isoformat(),
                    'suggestion': f'Try selecting an end date before {today.isoformat()}'
                }
        
        # Calculate final processing metrics
        total_days_to_process = (end_date - start_date).days + 1
        excluded_days = (original_end_date - end_date).days if date_adjusted else 0
        
        logger.info(f"📊 Final Date Range: {start_date} to {end_date}")
        logger.info(f"📊 Days to Process: {total_days_to_process}")
        if date_adjusted:
            logger.info(f"📊 Excluded Days: {excluded_days} (today/future)")
            logger.info(f"📊 Excluded Range: {end_date + timedelta(days=1)} to {original_end_date}")
        
        # Step 3: Calculate months to process
        logger.info("📊 Step 3: Calculating monthly processing strategy...")
        
        months_to_process = []
        current_date = start_date.replace(day=1)  # Start from first day of start month
        
        while current_date <= end_date:
            # Calculate end of current month
            if current_date.month == 12:
                month_end = current_date.replace(year=current_date.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = current_date.replace(month=current_date.month + 1, day=1) - timedelta(days=1)
            
            # Calculate actual start and end for this month within our range
            month_start = max(current_date, start_date)
            month_end = min(month_end, end_date)
            
            # Only add if this month has valid days to process
            if month_start <= month_end:
                days_in_month = (month_end - month_start).days + 1
                months_to_process.append({
                    'start': month_start,
                    'end': month_end,
                    'month_name': current_date.strftime('%B %Y'),
                    'days_count': days_in_month
                })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        logger.info(f"📅 Months to process: {len(months_to_process)}")
        for i, month_info in enumerate(months_to_process, 1):
            logger.info(f"   {i}. {month_info['month_name']}: {month_info['start']} to {month_info['end']} ({month_info['days_count']} days)")
        
        # Step 4: Process each month
        logger.info("🌐 Step 4: Starting month-by-month processing...")
        
        total_records_fetched = 0
        total_records_saved = 0
        total_days_processed = 0
        processing_errors = []
        
        try:
            async with BiometricAPIService() as api:
                logger.info("✅ API connection established successfully")
                
                for i, month_info in enumerate(months_to_process, 1):
                    logger.info(f"📅 Processing month {i}/{len(months_to_process)}: {month_info['month_name']}")
                    logger.info(f"   Date range: {month_info['start']} to {month_info['end']}")
                    logger.info(f"   Expected days: {month_info['days_count']}")
                    
                    try:
                        # Fetch data for this month
                        logger.info(f"🔍 Fetching attendance data from biometric API...")
                        
                        raw_data = await api.fetch_attendance_data(
                            emp_code=employee.biometric_id,
                            start_date=month_info['start'],
                            end_date=month_info['end']
                        )
                        
                        logger.info(f"📊 Month {i} - API returned {len(raw_data)} raw records")
                        total_records_fetched += len(raw_data)
                        
                        # Process and save raw data for this month
                        month_records_saved = 0
                        if len(raw_data) > 0:
                            logger.info(f"💾 Processing {len(raw_data)} raw records...")
                            
                            for j, api_record in enumerate(raw_data, 1):
                                try:
                                    parsed_record = api.parse_attendance_record(api_record)
                                    if not parsed_record:
                                        logger.debug(f"⚠️ Skipped unparseable record {j}")
                                        continue
                                    
                                    # Check for duplicates
                                    existing = self.db.query(RawAttendance).filter(
                                        RawAttendance.biometric_record_id == parsed_record['biometric_record_id']
                                    ).first()
                                    
                                    if not existing:
                                        raw_attendance = RawAttendance(
                                            employee_id=employee_id,
                                            **parsed_record
                                        )
                                        self.db.add(raw_attendance)
                                        month_records_saved += 1
                                        logger.debug(f"💾 Saved record {j}: {parsed_record.get('punch_time')}")
                                    else:
                                        logger.debug(f"⏭️ Skipped duplicate record {j}")
                                    
                                except Exception as parse_error:
                                    logger.error(f"💥 Error processing record {j} in month {i}: {parse_error}")
                                    processing_errors.append(f"Month {i}, Record {j}: {str(parse_error)}")
                                    continue
                            
                            # Commit raw data for this month
                            try:
                                self.db.commit()
                                logger.info(f"✅ Month {i} - {month_records_saved} new records saved to database")
                            except Exception as commit_error:
                                logger.error(f"💥 Failed to commit month {i} raw data: {commit_error}")
                                self.db.rollback()
                                raise commit_error
                        else:
                            logger.warning(f"⚠️ Month {i} - No raw attendance data found")
                        
                        total_records_saved += month_records_saved
                        
                        # Process daily attendance for this month
                        logger.info(f"📅 Processing daily attendance records...")
                        month_days_processed = await self._process_daily_attendance_for_employee(
                            employee_id, month_info['start'], month_info['end']
                        )
                        total_days_processed += month_days_processed
                        
                        logger.info(f"✅ Month {i} completed successfully:")
                        logger.info(f"   Raw records saved: {month_records_saved}")
                        logger.info(f"   Days processed: {month_days_processed}")
                        
                        # Brief pause between months to avoid API rate limiting
                        if i < len(months_to_process):
                            logger.debug(f"⏳ Brief pause before processing next month...")
                            await asyncio.sleep(2)
                    
                    except Exception as month_error:
                        error_msg = f"Month {i} ({month_info['month_name']}): {str(month_error)}"
                        logger.error(f"💥 Error processing {error_msg}")
                        processing_errors.append(error_msg)
                        # Continue with next month rather than failing entirely
                        continue
            
            # Step 5: Save comprehensive sync log
            logger.info("📝 Step 5: Saving range sync log...")
            
            sync_log = AttendanceSyncLog(
                employee_id=employee_id,
                emp_code=employee.biometric_id,
                last_sync_date=date.today(),
                sync_start_date=start_date,
                sync_end_date=end_date,
                records_fetched=total_records_fetched,
                records_processed=total_records_saved,
                sync_status="success" if not processing_errors else "partial_success"
            )
            self.db.add(sync_log)
            
            # Final commit
            try:
                self.db.commit()
                logger.info(f"✅ Range sync log saved successfully")
            except Exception as commit_error:
                logger.error(f"💥 Failed to save sync log: {commit_error}")
                self.db.rollback()
                raise commit_error
            
            # Step 6: Final Summary
            logger.info("=" * 100)
            logger.info(f"🎉 RANGE SYNC COMPLETED SUCCESSFULLY")
            logger.info(f"📊 Final Summary:")
            logger.info(f"   Employee: {employee.first_name} {employee.last_name}")
            logger.info(f"   Biometric ID: {employee.biometric_id}")
            logger.info(f"   Requested Range: {start_date} to {original_end_date}")
            if date_adjusted:
                logger.info(f"   Processed Range: {start_date} to {end_date} (auto-adjusted)")
                logger.info(f"   Excluded Days: {excluded_days} (today/future dates)")
            else:
                logger.info(f"   Processed Range: {start_date} to {end_date}")
            logger.info(f"   Months Processed: {len(months_to_process)}")
            logger.info(f"   API Records Fetched: {total_records_fetched}")
            logger.info(f"   New Records Saved: {total_records_saved}")
            logger.info(f"   Days Processed: {total_days_processed}")
            if processing_errors:
                logger.warning(f"   Processing Errors: {len(processing_errors)}")
                for error in processing_errors[:3]:  # Show first 3 errors
                    logger.warning(f"     - {error}")
                if len(processing_errors) > 3:
                    logger.warning(f"     ... and {len(processing_errors) - 3} more errors")
            logger.info("=" * 100)
            
            # Prepare response
            response = {
                'success': True,
                'message': 'Range sync completed successfully',
                'records_fetched': total_records_fetched,
                'records_saved': total_records_saved,
                'days_processed': total_days_processed,
                'months_processed': len(months_to_process),
                'sync_type': 'range',
                'employee_name': f"{employee.first_name} {employee.last_name}",
                'biometric_id': employee.biometric_id,
                'requested_range': f"{start_date} to {original_end_date}",
                'processed_range': f"{start_date} to {end_date}",
                'date_adjusted': date_adjusted,
                'total_days_requested': (original_end_date - start_date).days + 1,
                'total_days_processed': total_days_to_process
            }
            
            if date_adjusted:
                response.update({
                    'excluded_days': excluded_days,
                    'excluded_range': f"{end_date + timedelta(days=1)} to {original_end_date}",
                    'adjustment_reason': 'Excluded today and future dates'
                })
            
            if processing_errors:
                response.update({
                    'partial_success': True,
                    'processing_errors': len(processing_errors),
                    'error_summary': processing_errors[:5]  # Include first 5 errors
                })
            
            return response
            
        except HTTPException as api_error:
            logger.error(f"🌐 API error during range sync: {api_error.detail}")
            self.db.rollback()
            
            # Log failed sync
            await self._log_failed_sync(employee_id, employee.biometric_id, start_date, end_date, str(api_error.detail))
            
            return {
                'success': False,
                'message': f'API error: {api_error.detail}',
                'error_code': 'API_ERROR',
                'requested_range': f"{start_date} to {original_end_date}",
                'processed_range': f"{start_date} to {end_date}" if date_adjusted else None,
                'date_adjusted': date_adjusted
            }
            
        except Exception as e:
            logger.error("=" * 100)
            logger.error(f"💥 RANGE SYNC FAILED - UNEXPECTED ERROR")
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {str(e)}")
            logger.error(f"Traceback:")
            logger.error(traceback.format_exc())
            logger.error("=" * 100)
            
            self.db.rollback()
            
            # Log failed sync
            await self._log_failed_sync(employee_id, employee.biometric_id, start_date, end_date, str(e))
            
            return {
                'success': False,
                'message': f'Range sync failed: {str(e)}',
                'error_code': 'UNEXPECTED_ERROR',
                'error_type': type(e).__name__,
                'requested_range': f"{start_date} to {original_end_date}",
                'processed_range': f"{start_date} to {end_date}" if date_adjusted else None,
                'date_adjusted': date_adjusted
            }
    
    async def recalculate_employee_stats(
        self,
        employee_id: int,
        start_date: date,
        end_date: date
    ) -> Dict:
        """
        Recalculate attendance statistics from existing database records
        with detailed calculation breakdown using HOUR-BASED calculation
        """
        
        logger.info("=" * 80)
        logger.info(f"🧮 RECALCULATING ATTENDANCE STATISTICS (HOUR-BASED)")
        logger.info(f"👤 Employee ID: {employee_id}")
        logger.info(f"📅 Date Range: {start_date} to {end_date}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Validate Employee
            employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
            if not employee:
                raise HTTPException(
                    status_code=404,
                    detail=f"Employee with ID {employee_id} not found"
                )
            
            logger.info(f"✅ Employee: {employee.first_name} {employee.last_name}")
            logger.info(f"🆔 Employee ID: {employee.employee_id}")
            logger.info(f"🔢 Biometric ID: {employee.biometric_id}")

            is_lecturer = self._is_lecturer(employee)

            if is_lecturer:
                logger.info(f"👨‍🏫 Employee is a lecturer - using hour-based calculation with 3-day work week")
            else:
                logger.info(f"👨‍💼 Regular employee - using hour-based calculation")
            
            # Step 2: Get all processed attendance records in date range
            logger.info("📊 Step 2: Fetching processed attendance records...")
            
            records = self.db.query(ProcessedAttendance).filter(
                and_(
                    ProcessedAttendance.employee_id == employee_id,
                    ProcessedAttendance.date >= start_date,
                    ProcessedAttendance.date <= end_date
                )
            ).order_by(ProcessedAttendance.date).all()
            
            total_records = len(records)
            logger.info(f"📋 Found {total_records} processed attendance records")
            
            if total_records == 0:
                logger.warning("⚠️ No attendance records found in the specified date range")
                return {
                    'summary': {
                        'total_days': 0,
                        'total_work_days': 0,
                        'present_days': 0,
                        'absent_days': 0,
                        'late_days': 0,
                        'total_hours': 0.0,
                        'attendance_percentage': 0.0
                    },
                    'calculation_details': {
                        'total_records': 0,
                        'date_range': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
                        'message': 'No records found in the specified date range'
                    },
                    'records_analyzed': 0
                }
            
            # Step 3: Categorize records by type
            logger.info("📊 Step 3: Categorizing attendance records...")
            
            work_day_records = []
            day_off_records = []
            leave_records = []
            holiday_records = []
            pending_records = []
            in_progress_records = []
            
            for record in records:
                if record.status == "day_off":
                    day_off_records.append(record)
                elif record.status == "on_leave":
                    leave_records.append(record)
                elif record.status == "public_holiday":
                    holiday_records.append(record)
                elif record.status == "pending":
                    pending_records.append(record)
                elif record.status == "in_progress":
                    in_progress_records.append(record)
                else:
                    work_day_records.append(record)
            
            logger.info(f"📊 Record categorization:")
            logger.info(f"   Work days: {len(work_day_records)}")
            logger.info(f"   Days off: {len(day_off_records)}")
            logger.info(f"   Leave days: {len(leave_records)}")
            logger.info(f"   Public holidays: {len(holiday_records)}")
            logger.info(f"   Pending days: {len(pending_records)}")
            logger.info(f"   In progress: {len(in_progress_records)}")
            
            # Step 4: Calculate HOUR-BASED statistics
            logger.info("📊 Step 4: Calculating HOUR-BASED attendance statistics...")
            
            # Calculate total hours worked (including leave/holiday credits)
            total_hours_worked = 0.0
            actually_present_records = [r for r in work_day_records + in_progress_records if r.is_present]
            
            # Hours from actual work
            for record in actually_present_records:
                hours = record.total_working_hours or 0.0
                total_hours_worked += hours
            
            # Hours from leave and holiday records (credited hours)
            for record in leave_records + holiday_records:
                hours = record.total_working_hours or 0.0
                total_hours_worked += hours
            
            # Calculate required working hours for the period
            required_hours = self._calculate_required_hours(employee, start_date, end_date)
            
            # Hour-based attendance percentage
            if required_hours > 0:
                attendance_percentage = round((total_hours_worked / required_hours) * 100, 1)
                # Optional: Cap at reasonable maximum for display purposes
                attendance_percentage = min(120.0, attendance_percentage)
            else:
                attendance_percentage = 0.0
            
            # Present/Absent days (for reference)
            present_with_valid_reasons = actually_present_records + leave_records + holiday_records
            total_present_days = len(present_with_valid_reasons)
            actual_absent_records = [r for r in work_day_records if not r.is_present]
            absent_days = len(actual_absent_records)
            
            # Late arrivals
            late_records = [r for r in work_day_records + in_progress_records if r.is_late]
            late_days = len(late_records)
            
            # Calculate average daily hours
            average_daily_hours = round(total_hours_worked / len(actually_present_records), 2) if len(actually_present_records) > 0 else 0.0
            
            # Step 5: Log comprehensive results
            logger.info("=" * 80)
            logger.info(f"📊 HOUR-BASED CALCULATION COMPLETED")
            logger.info(f"📊 Results:")
            logger.info(f"   Date Range: {start_date} to {end_date}")
            logger.info(f"   Total Records: {total_records}")
            logger.info(f"   Hours Worked: {round(total_hours_worked, 1)}h")
            logger.info(f"   Hours Required: {round(required_hours, 1)}h")
            logger.info(f"   Attendance Rate: {attendance_percentage}% (HOUR-BASED)")
            logger.info(f"   Present Days: {total_present_days} (for reference)")
            logger.info(f"   Absent Days: {absent_days}")
            logger.info(f"   Late Days: {late_days}")
            logger.info(f"   Employee Type: {'Lecturer' if is_lecturer else 'Regular Employee'}")
            logger.info("=" * 80)
            
            # Prepare comprehensive summary
            summary = {
                'total_calendar_days': total_records,
                'total_eligible_days': len(work_day_records + in_progress_records + leave_records + holiday_records),
                'day_off_days': len(day_off_records),
                'leave_days': len(leave_records),
                'holiday_days': len(holiday_records),
                'pending_days': len(pending_records),
                'in_progress_days': len(in_progress_records),
                'actually_worked_days': len(actually_present_records),
                'present_days': total_present_days,
                'absent_days': absent_days,
                'late_days': late_days,
                'total_hours': round(total_hours_worked, 2),
                'required_hours': round(required_hours, 2),
                'attendance_percentage': attendance_percentage,  # HOUR-BASED
                'is_lecturer': is_lecturer,
                'calculation_method': 'hour_based'
            }
            
            # Detailed calculation breakdown
            calculation_details = {
                'calculation_method': 'hour_based',
                'total_records': total_records,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'days_span': (end_date - start_date).days + 1
                },
                'hour_based_calculation': {
                    'total_hours_worked': round(total_hours_worked, 2),
                    'required_hours': round(required_hours, 2),
                    'attendance_percentage': attendance_percentage,
                    'formula': f"{round(total_hours_worked, 1)}h ÷ {round(required_hours, 1)}h × 100",
                    'note': 'Hour-based calculation matches manual timesheet methodology'
                },
                'hours_breakdown': {
                    'actual_work_hours': round(sum(r.total_working_hours or 0 for r in actually_present_records), 2),
                    'leave_credited_hours': round(sum(r.total_working_hours or 0 for r in leave_records), 2),
                    'holiday_credited_hours': round(sum(r.total_working_hours or 0 for r in holiday_records), 2)
                },
                'employee_info': {
                    'employee_id': employee.employee_id,
                    'biometric_id': employee.biometric_id,
                    'name': f"{employee.first_name} {employee.last_name}",
                    'position': employee.position,
                    'is_lecturer': is_lecturer,
                    'department': employee.department.name if employee.department else 'No Department',
                    'shift': employee.shift.name if employee.shift else 'No Shift'
                },
                'calculation_timestamp': datetime.now().isoformat()
            }
            
            return {
                'summary': summary,
                'calculation_details': calculation_details,
                'records_analyzed': total_records
            }
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"💥 RECALCULATION FAILED")
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {str(e)}")
            logger.error(f"Traceback:")
            logger.error(traceback.format_exc())
            logger.error("=" * 80)
            raise e

    async def _log_failed_sync(self, employee_id: int, emp_code: str, start_date: date, end_date: date, error_message: str):
        """Log a failed sync attempt"""
        try:
            sync_log = AttendanceSyncLog(
                employee_id=employee_id,
                emp_code=emp_code,
                last_sync_date=date.today(),
                sync_start_date=start_date,
                sync_end_date=end_date,
                sync_status="failed",
                error_message=error_message
            )
            self.db.add(sync_log)
            self.db.commit()
            logger.info(f"📝 Failed sync logged successfully")
        except Exception as log_error:
            logger.error(f"💥 Failed to log sync failure: {log_error}")
    
    async def _process_daily_attendance_for_employee(
        self,
        employee_id: int,
        start_date: date,
        end_date: date
    ) -> int:
        """Process daily attendance records for an employee with detailed logging"""
        
        logger.info(f"📅 Processing daily attendance for employee {employee_id}")
        logger.info(f"📊 Date range: {start_date} to {end_date}")
        
        # Don't process future dates
        today = date.today()
        if end_date > today:
            end_date = today
            logger.info(f"📅 Adjusted end date to today: {end_date} (excluded future dates)")
        
        # Skip entirely if start date is in the future
        if start_date > today:
            logger.info(f"⏭️ Skipping processing - all dates are in the future")
            return 0
        
        days_processed = 0
        current_date = start_date
        
        while current_date <= end_date:
            logger.info(f"📅 Processing day: {current_date} ({current_date.strftime('%A')})")
            
            try:
                await self._process_single_day_attendance(employee_id, current_date)
                days_processed += 1
                logger.debug(f"✅ Day {current_date} processed successfully")
            except Exception as day_error:
                logger.error(f"💥 Error processing {current_date}: {day_error}")
                # Continue processing other days
            
            current_date += timedelta(days=1)
        
        logger.info(f"📊 Daily processing complete: {days_processed} days processed")
        return days_processed
    
    async def _process_single_day_attendance(self, employee_id: int, target_date: date):
        """Process attendance for a single day with FIXED priority order - FINAL VERSION with PUBLIC HOLIDAY FAIRNESS"""

        # Don't process future dates
        today = date.today()
        if target_date > today:
            logger.info(f"⏭️ Skipping {target_date} - future date")
            return
        
        logger.info(f"📅 Processing attendance for {target_date}")
        
        # Get employee info
        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            logger.error(f"❌ Employee {employee_id} not found during day processing")
            return
        
        logger.info(f"👤 Processing for: {employee.first_name} {employee.last_name}")
        
        is_lecturer = self._is_lecturer(employee)
        if is_lecturer:
            logger.info(f"👨‍🏫 Employee is a lecturer")
        
        # ✅ Check for approved leave FIRST (highest priority)
        approved_leave = self.db.query(Leave).filter(
            and_(
                Leave.employee_id == employee_id,
                Leave.start_date <= target_date,
                Leave.end_date >= target_date,
                Leave.status == LeaveStatusEnum.ACTIVE
            )
        ).first()
        
        if approved_leave:
            logger.info(f"🏖️ Employee is on approved leave: {approved_leave.leave_type.name}")
            logger.info(f"   Leave period: {approved_leave.start_date} to {approved_leave.end_date}")
            logger.info(f"   Reason: {approved_leave.reason}")
            
            # Calculate expected working hours for this day
            expected_hours = self._calculate_expected_daily_hours(employee, target_date)
            
            self._create_or_update_processed_attendance(
                employee_id=employee_id,
                target_date=target_date,
                shift_id=employee.shift.id if employee.shift else None,
                status="on_leave",
                is_present=True,  # ✅ Leave counts as present for attendance rate
                total_working_hours=expected_hours,  # ✅ Credit full day hours
                leave_id=approved_leave.id,
                leave_type=approved_leave.leave_type.name
            )
            return  # ✅ CRITICAL: Exit here - don't process attendance records
        
        # ✅ Check for actual attendance records SECOND (before public holiday processing)
        start_datetime = datetime.combine(target_date, time.min)
        end_datetime = datetime.combine(target_date, time.max)
        
        raw_records = self.db.query(RawAttendance).filter(
            and_(
                RawAttendance.employee_id == employee_id,
                RawAttendance.punch_time >= start_datetime,
                RawAttendance.punch_time <= end_datetime
            )
        ).order_by(RawAttendance.punch_time).all()
        
        logger.info(f"📋 Found {len(raw_records)} raw attendance records for {target_date}")
        
        # ✅ Check if this is a public holiday
        public_holiday = self.db.query(PublicHoliday).filter(
            and_(
                PublicHoliday.date == target_date,
                PublicHoliday.is_active == True
            )
        ).first()
        
        # ✅ IMPROVED: Handle public holiday with actual attendance
        if public_holiday:
            if len(raw_records) > 0:
                # Employee worked on public holiday - process actual attendance with full holiday credit
                logger.info(f"🎉 Employee worked on public holiday '{public_holiday.name}' - will credit full holiday hours")
                # Continue to process actual attendance below, but will override hours later
                
            else:
                # No attendance records on public holiday - apply automatic credit
                logger.info(f"🎉 Public holiday '{public_holiday.name}' - no attendance records, applying automatic credit")
                
                expected_hours = self._calculate_expected_daily_hours(employee, target_date)
                
                self._create_or_update_processed_attendance(
                    employee_id=employee_id,
                    target_date=target_date,
                    shift_id=employee.shift.id if employee.shift else None,
                    status="public_holiday",
                    is_present=True,
                    total_working_hours=expected_hours,
                    holiday_name=public_holiday.name
                )
                return
        
        # ✅ Check for day_off (weekends without shifts)
        if self._should_be_day_off(employee, target_date):
            if len(raw_records) > 0:
                # Employee worked on day off - process actual attendance
                logger.info(f"📅 Employee worked on day off - processing actual attendance")
                # Continue processing below
            else:
                # No attendance on day off
                logger.info(f"📅 Day off for employee - no attendance records")
                
                self._create_or_update_processed_attendance(
                    employee_id=employee_id,
                    target_date=target_date,
                    shift_id=employee.shift.id if employee.shift else None,
                    status="day_off",
                    is_present=False,
                    total_working_hours=0
                )
                return
        
        # ✅ PROCESS ACTUAL ATTENDANCE RECORDS
        if len(raw_records) == 0:
            # No records and not a special day
            shift_id = employee.shift.id if employee.shift else None
            
            if target_date == today:
                status = "pending"
                is_present = False
            else:
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
        
        # ✅ IMPROVED: Better handling of incomplete records
        # Analyze punch records first
        check_ins = [r for r in raw_records if r.punch_state == "0"]
        check_outs = [r for r in raw_records if r.punch_state == "1"]
        
        logger.info(f"📊 Initial punch analysis:")
        logger.info(f"   Check-ins (state 0): {len(check_ins)}")
        logger.info(f"   Check-outs (state 1): {len(check_outs)}")
        
        # Handle incomplete records (only check-in OR only check-out)
        if check_ins and not check_outs:
            # Only check-in found
            if target_date < today:
                # Past day with only check-in - mark as ABSENT
                logger.warning(f"⚠️ Only check-in found for past day - marking as ABSENT")
                
                check_in_time = check_ins[0].punch_time
                shift_start_time, shift_end_time, shift_id = self._get_shift_times(employee, target_date)
                
                # Calculate if late for record keeping
                is_late = False
                late_minutes = 0
                notes = "Check-in only, no check-out recorded"
                
                if check_in_time and shift_start_time:
                    expected_start = datetime.combine(target_date, shift_start_time)
                    if check_in_time > expected_start:
                        late_duration = check_in_time - expected_start
                        late_minutes = int(late_duration.total_seconds() / 60)
                        is_late = True
                
                # Add special context notes
                if public_holiday:
                    notes += f", Public holiday: {public_holiday.name}"
                if self._should_be_day_off(employee, target_date):
                    notes += ", Day off"
                
                self._create_or_update_processed_attendance(
                    employee_id=employee_id,
                    target_date=target_date,
                    check_in_time=check_in_time,
                    shift_id=shift_id,
                    expected_start_time=shift_start_time,
                    expected_end_time=shift_end_time,
                    is_present=False,  # Incomplete = absent
                    is_late=is_late,
                    late_minutes=late_minutes,
                    status="absent",
                    notes=notes
                )
                return
            else:
                # Today with only check-in - continue to in-progress logic below
                pass
                
        elif not check_ins and check_outs:
            # Only check-out found - handle based on day type
            logger.warning(f"⚠️ Only check-out found (no check-in)")
            
            check_out_time = check_outs[-1].punch_time
            shift_start_time, shift_end_time, shift_id = self._get_shift_times(employee, target_date)
            
            # ✅ CRITICAL: Check if this is a day off FIRST
            if self._should_be_day_off(employee, target_date):
                logger.info(f"📅 Check-out only on day off - marking as DAY_OFF")
                
                notes = "Check-out only, no check-in recorded, Day off"
                if public_holiday:
                    notes = f"Public holiday: {public_holiday.name}, Check-out only"
                
                self._create_or_update_processed_attendance(
                    employee_id=employee_id,
                    target_date=target_date,
                    check_out_time=check_out_time,
                    shift_id=shift_id,
                    expected_start_time=shift_start_time,
                    expected_end_time=shift_end_time,
                    is_present=False,
                    status="day_off",  # ✅ Mark as day_off, not absent
                    notes=notes
                )
                return
            
            # For regular working days with only check-out - mark as absent
            logger.warning(f"⚠️ Check-out only on working day - marking as ABSENT")
            
            # Calculate if early departure for record keeping
            is_early_departure = False
            early_departure_minutes = 0
            notes = "Check-out only, no check-in recorded"
            
            if check_out_time and shift_end_time:
                expected_end = datetime.combine(target_date, shift_end_time)
                if check_out_time < expected_end:
                    early_duration = expected_end - check_out_time
                    early_departure_minutes = int(early_duration.total_seconds() / 60)
                    is_early_departure = True
            
            # Add special context notes
            if public_holiday:
                notes += f", Public holiday: {public_holiday.name}"
            
            self._create_or_update_processed_attendance(
                employee_id=employee_id,
                target_date=target_date,
                check_out_time=check_out_time,
                shift_id=shift_id,
                expected_start_time=shift_start_time,
                expected_end_time=shift_end_time,
                is_present=False,
                is_early_departure=is_early_departure,
                early_departure_minutes=early_departure_minutes,
                status="absent",
                notes=notes
            )
            return

        # Check if target_date is today for in-progress handling
        if target_date == today:
            logger.info(f"📅 {target_date} is today - checking current status")
            
            if check_ins and not check_outs:
                # Employee has checked in but not out - mark as "in_progress"
                logger.info(f"✅ Employee checked in today but hasn't checked out yet - marking as IN PROGRESS")
                
                check_in_time = check_ins[0].punch_time
                shift_start_time, shift_end_time, shift_id = self._get_shift_times(employee, target_date)
                
                # Calculate if late
                is_late = False
                late_minutes = 0
                notes_list = []
                
                if check_in_time and shift_start_time:
                    expected_start = datetime.combine(target_date, shift_start_time)
                    if check_in_time > expected_start:
                        late_duration = check_in_time - expected_start
                        late_minutes = int(late_duration.total_seconds() / 60)
                        is_late = True
                        logger.info(f"⏰ Late arrival: {late_minutes} minutes after {shift_start_time}")

                # Add special context notes
                if public_holiday:
                    notes_list.append(f"Public holiday: {public_holiday.name}")
                if self._should_be_day_off(employee, target_date):
                    notes_list.append("Day off")

                self._create_or_update_processed_attendance(
                    employee_id=employee_id,
                    target_date=target_date,
                    check_in_time=check_in_time,
                    shift_id=shift_id,
                    expected_start_time=shift_start_time,
                    expected_end_time=shift_end_time,
                    is_present=True,
                    is_late=is_late,
                    late_minutes=late_minutes,
                    status="in_progress",
                    notes=", ".join(notes_list) if notes_list else None
                )
                return
            elif check_ins and check_outs:
                # Employee has both check in and out - process normally
                logger.info(f"✅ Employee has completed attendance for today - processing normally")
                # Continue with normal processing below
            else:
                # No check-in yet today - mark as "pending"
                logger.info(f"⏳ No check-in recorded yet today - marking as PENDING")
                self._create_or_update_processed_attendance(
                    employee_id=employee_id,
                    target_date=target_date,
                    status="pending",
                    is_present=False
                )
                return
        
        # ✅ PROCESS COMPLETE RECORDS (both check-in and check-out)
        logger.info(f"🔍 Analyzing complete punch patterns...")
        
        # Determine check-in and check-out times
        check_in_time = check_ins[0].punch_time if check_ins else None
        check_out_time = check_outs[-1].punch_time if check_outs else None
        
        if check_in_time:
            logger.info(f"🟢 First check-in: {check_in_time.strftime('%H:%M:%S')}")
        if check_out_time:
            logger.info(f"🔴 Last check-out: {check_out_time.strftime('%H:%M:%S')}")
        
        # Calculate actual working time
        total_minutes = 0
        actual_hours_worked = 0.0
        status = "present"
        
        if check_in_time and check_out_time:
            working_duration = check_out_time - check_in_time
            total_minutes = int(working_duration.total_seconds() / 60)
            actual_hours_worked = round(total_minutes / 60, 2)
            logger.info(f"⏱️ Actual working time: {total_minutes} minutes ({actual_hours_worked} hours)")
        else:
            # This shouldn't happen due to earlier checks, but just in case
            status = "absent"
            logger.warning(f"⚠️ Incomplete attendance record - marking as ABSENT")
        
        # Get shift information for calculations
        logger.info(f"⏰ Analyzing shift requirements...")
        shift_start_time, shift_end_time, shift_id = self._get_shift_times(employee, target_date)
        
        # Calculate late arrival and early departure
        is_late = False
        is_early_departure = False
        late_minutes = 0
        early_departure_minutes = 0
        
        if check_in_time and shift_start_time:
            expected_start = datetime.combine(target_date, shift_start_time)
            
            if check_in_time > expected_start:
                late_duration = check_in_time - expected_start
                late_minutes = int(late_duration.total_seconds() / 60)
                is_late = True
                
                logger.info(f"⏰ Late arrival: {late_minutes} minutes after {shift_start_time}")
                
                if late_minutes > 15:  # More than 15 minutes late
                    status = "late"
                    logger.info(f"📊 Status changed to LATE (more than 15 minutes)")
            else:
                logger.info(f"✅ On-time arrival (expected: {shift_start_time})")
        
        if check_out_time and shift_end_time:
            expected_end = datetime.combine(target_date, shift_end_time)
            
            if check_out_time < expected_end:
                early_duration = expected_end - check_out_time
                early_departure_minutes = int(early_duration.total_seconds() / 60)
                is_early_departure = True
                
                logger.info(f"⏰ Early departure: {early_departure_minutes} minutes before {shift_end_time}")
            else:
                logger.info(f"✅ Normal departure (expected: {shift_end_time})")
        
        # ✅ CRITICAL: PUBLIC HOLIDAY FAIRNESS - Determine final credited hours
        final_credited_hours = actual_hours_worked
        final_notes_list = []
        
        if public_holiday:
            # ✅ POLICY: Credit full public holiday hours instead of actual worked hours
            expected_holiday_hours = self._calculate_expected_daily_hours(employee, target_date)
            final_credited_hours = expected_holiday_hours  # Override with full credit
            
            final_notes_list.append(f"Public holiday: {public_holiday.name}")
            logger.info(f"🎉 PUBLIC HOLIDAY FAIRNESS: Actual {actual_hours_worked}h → Credited {final_credited_hours}h")
        
        if self._should_be_day_off(employee, target_date):
            final_notes_list.append("Day off")
        
        final_notes = ", ".join(final_notes_list) if final_notes_list else None
        
        # Final status determination and logging
        logger.info(f"🔍 SAVING RECORD DEBUG:")
        logger.info(f"   target_date: {target_date}")
        logger.info(f"   check_in_time: {check_in_time}")
        logger.info(f"   check_out_time: {check_out_time}")
        logger.info(f"   shift_start_time: {shift_start_time}")
        logger.info(f"   shift_end_time: {shift_end_time}")
        logger.info(f"   is_late: {is_late}")
        logger.info(f"   late_minutes: {late_minutes}")
        logger.info(f"   status: {status}")
        logger.info(f"   final_notes: {final_notes}")
        
        # ✅ CRITICAL: Create record with PUBLIC HOLIDAY FAIR crediting
        self._create_or_update_processed_attendance(
            employee_id=employee_id,
            target_date=target_date,
            check_in_time=check_in_time,
            check_out_time=check_out_time,
            shift_id=shift_id,
            expected_start_time=shift_start_time,
            expected_end_time=shift_end_time,
            total_working_minutes=int(final_credited_hours * 60),  # Use credited hours for minutes
            total_working_hours=final_credited_hours,  # ✅ CRITICAL: Use credited hours, not actual
            is_present=(status not in ["absent", "day_off", "pending"]),
            is_late=is_late,
            is_early_departure=is_early_departure,
            late_minutes=late_minutes,
            early_departure_minutes=early_departure_minutes,
            status=status,
            notes=final_notes  # ✅ This will include both public holiday note AND late/early notes
        )
        
        logger.info(f"✅ Processed attendance record saved with fair public holiday crediting")

    def _get_shift_times(self, employee: Employee, target_date: date) -> Tuple[Optional[time], Optional[time], Optional[int]]:
        """Get shift start/end times and shift_id for a given date - PRODUCTION READY VERSION"""
        
        shift_start_time = None
        shift_end_time = None
        shift_id = None
        
        # Check if employee has a shift assigned
        if employee.shift:
            shift = employee.shift
            shift_id = shift.id
            weekday = target_date.weekday()  # 0=Monday, 6=Sunday
            
            logger.info(f"🔍 SHIFT DEBUG for {target_date}:")
            logger.info(f"📋 Employee shift: {shift.name}")
            logger.info(f"🔧 Shift type: {shift.shift_type}")
            logger.info(f"📅 Weekday number: {weekday} ({['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][weekday]})")
            
            # ✅ CRITICAL FIX: Handle both enum and string types properly
            shift_type_str = str(shift.shift_type)
            
            # Extract the enum value if it's an enum object
            if hasattr(shift.shift_type, 'value'):
                shift_type_str = shift.shift_type.value
            
            # Normalize to lowercase for comparison
            shift_type_normalized = shift_type_str.lower().strip()
            
            logger.info(f"🔍 Normalized shift type: '{shift_type_normalized}'")
            
            # ✅ FIXED COMPARISON LOGIC
            if shift_type_normalized == 'standard':
                logger.info("✅ Detected STANDARD shift type")
                
                if weekday < 5:  # Monday-Friday (0-4)
                    shift_start_time = shift.weekday_start
                    shift_end_time = shift.weekday_end
                    logger.info(f"📋 Using WEEKDAY schedule: {shift_start_time} - {shift_end_time}")
                    logger.info(f"🔍 Raw shift data: weekday_start={shift.weekday_start}, weekday_end={shift.weekday_end}")
                    
                elif weekday == 5:  # Saturday
                    shift_start_time = shift.saturday_start
                    shift_end_time = shift.saturday_end
                    logger.info(f"📋 Using SATURDAY schedule: {shift_start_time} - {shift_end_time}")
                    logger.info(f"🔍 Raw shift data: saturday_start={shift.saturday_start}, saturday_end={shift.saturday_end}")
                    
                else:  # Sunday (weekday == 6)
                    shift_start_time = shift.sunday_start
                    shift_end_time = shift.sunday_end
                    logger.info(f"📋 Using SUNDAY schedule: {shift_start_time} - {shift_end_time}")
                    logger.info(f"🔍 Raw shift data: sunday_start={shift.sunday_start}, sunday_end={shift.sunday_end}")
                    
            elif shift_type_normalized == 'dynamic':
                logger.info("✅ Detected DYNAMIC shift type")
                
                # Dynamic shift - use day-specific times
                day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                day_name = day_names[weekday]
                
                # Use getattr to get the day-specific start/end times
                shift_start_time = getattr(shift, f'{day_name}_start', None)
                shift_end_time = getattr(shift, f'{day_name}_end', None)
                
                logger.info(f"📋 Using DYNAMIC {day_name} schedule: {shift_start_time} - {shift_end_time}")
                
            else:
                logger.error(f"❌ UNKNOWN shift type: '{shift_type_normalized}' (original: {shift.shift_type})")
                logger.error(f"   Type of shift.shift_type: {type(shift.shift_type)}")
                logger.error(f"   This is critical - shift times will be NULL!")
            
            # ✅ CRITICAL: Validate that times were actually retrieved
            if not shift_start_time or not shift_end_time:
                logger.error(f"❌ SHIFT TIMES NOT FOUND!")
                logger.error(f"   shift_start_time: {shift_start_time}")
                logger.error(f"   shift_end_time: {shift_end_time}")
                logger.error(f"   This will cause late calculation to fail!")
                logger.error(f"   Shift details: type={shift_type_normalized}, weekday={weekday}")
            else:
                logger.info(f"✅ Shift times retrieved successfully: {shift_start_time} - {shift_end_time}")
                
        else:
            logger.warning(f"⚠️ No shift assigned to employee - cannot determine work schedule")
        
        return shift_start_time, shift_end_time, shift_id

    def _create_or_update_processed_attendance(self, employee_id: int, target_date: date, **kwargs):
        """Create or update processed attendance record with logging and PERFECT notes generation"""
        
        # Get explicit notes passed in (like "Public holiday: Name")
        explicit_notes = kwargs.get('notes', '')
        
        # ✅ Generate automatic notes for late/early arrivals
        auto_notes = []
        
        # Check for late arrival
        if kwargs.get('is_late') and kwargs.get('late_minutes'):
            late_mins = kwargs.get('late_minutes')
            if late_mins > 0:
                auto_notes.append(f"{late_mins}min late")
        
        # Check for early departure  
        if kwargs.get('is_early_departure') and kwargs.get('early_departure_minutes'):
            early_mins = kwargs.get('early_departure_minutes')
            if early_mins > 0:
                auto_notes.append(f"{early_mins}min early")
        
        # ✅ COMBINE: Explicit notes FIRST, then auto-generated notes
        all_notes = []
        if explicit_notes:
            all_notes.append(explicit_notes)
        if auto_notes:
            all_notes.extend(auto_notes)
        
        # Set final combined notes
        if all_notes:
            kwargs['notes'] = ", ".join(all_notes)
            logger.info(f"📝 Final combined notes: {kwargs['notes']}")
        
        # Check if record already exists
        existing = self.db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.employee_id == employee_id,
                ProcessedAttendance.date == target_date
            )
        ).first()
        
        if existing:
            logger.info(f"🔄 Updating existing processed record for {target_date}")
            
            # Log what's being updated
            changes = []
            for key, value in kwargs.items():
                old_value = getattr(existing, key, None)
                if old_value != value:
                    changes.append(f"{key}: {old_value} -> {value}")
                setattr(existing, key, value)
            
            if changes:
                logger.info(f"📝 Changes made:")
                for change in changes:
                    logger.info(f"   {change}")
            
            existing.updated_at = datetime.utcnow()
        else:
            logger.info(f"💾 Creating new processed record for {target_date}")
            
            employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
            processed_attendance = ProcessedAttendance(
                employee_id=employee_id,
                emp_code=employee.biometric_id if employee else "",
                date=target_date,
                **kwargs
            )
            self.db.add(processed_attendance)
            
            # Log what's being created
            logger.info(f"📝 New record details:")
            for key, value in kwargs.items():
                logger.info(f"   {key}: {value}")
        
        try:
            self.db.commit()
            logger.debug(f"✅ Processed attendance record committed for {target_date}")
        except Exception as commit_error:
            logger.error(f"💥 Failed to commit processed attendance: {commit_error}")
            self.db.rollback()
            raise commit_error
    
    def get_employee_attendance_summary(
        self,
        employee_id: int,
        start_date: date,
        end_date: date
    ) -> Dict:
        """Get attendance summary for employee with HOUR-BASED calculation"""
        
        logger.info(f"📊 Getting attendance summary for employee {employee_id}")
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        
        # Get employee info
        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        # Check if employee is a lecturer
        is_lecturer = self._is_lecturer(employee)
        
        if is_lecturer:
            logger.info(f"👨‍🏫 Employee is a lecturer - using hour-based calculation")
        else:
            logger.info(f"👨‍💼 Regular employee - using hour-based calculation")
        
        # Get processed records
        records = self.db.query(ProcessedAttendance).filter(
            and_(
                ProcessedAttendance.employee_id == employee_id,
                ProcessedAttendance.date >= start_date,
                ProcessedAttendance.date <= end_date
            )
        ).order_by(ProcessedAttendance.date.desc()).all()
        
        logger.info(f"📋 Found {len(records)} processed attendance records")
        
        # Categorize records properly
        work_day_records = []
        day_off_records = []
        leave_records = []
        holiday_records = []
        in_progress_records = []
        pending_records = []
        
        for r in records:
            if r.status == "day_off":
                day_off_records.append(r)
            elif r.status == "on_leave":
                leave_records.append(r)
            elif r.status == "public_holiday":
                holiday_records.append(r)
            elif r.status == "in_progress":
                in_progress_records.append(r)
            elif r.status == "pending":
                pending_records.append(r)
            else:
                work_day_records.append(r)
        
        # ✅ HOUR-BASED CALCULATION
        # Calculate total hours worked (including leave/holiday credits)
        actually_present_records = [r for r in work_day_records + in_progress_records if r.is_present]
        
        total_hours_worked = 0.0
        # Hours from actual work
        for record in actually_present_records:
            hours = record.total_working_hours or 0.0
            total_hours_worked += hours
        
        # Hours from leave and holiday records (credited hours)
        for record in leave_records + holiday_records:
            hours = record.total_working_hours or 0.0
            total_hours_worked += hours
        
        # Calculate required working hours for the period
        required_hours = self._calculate_required_hours(employee, start_date, end_date)
        
        # Hour-based attendance percentage
        if required_hours > 0:
            attendance_percentage = round((total_hours_worked / required_hours) * 100, 1)
            # Optional: Cap at reasonable maximum for display purposes
            attendance_percentage = min(120.0, attendance_percentage)
        else:
            attendance_percentage = 0.0
        
        # Present/Absent days (for reference)
        present_with_valid_reasons = actually_present_records + leave_records + holiday_records
        total_present_days = len(present_with_valid_reasons)
        actual_absent_records = [r for r in work_day_records if not r.is_present]
        absent_days = len(actual_absent_records)
        
        # Late arrivals
        late_records = [r for r in work_day_records + in_progress_records if r.is_late]
        late_days = len(late_records)

        summary = {
            'total_calendar_days': len(records),
            'total_eligible_days': len(work_day_records + in_progress_records + leave_records + holiday_records),
            'day_off_days': len(day_off_records),
            'leave_days': len(leave_records),
            'holiday_days': len(holiday_records),
            'in_progress_days': len(in_progress_records),
            'pending_days': len(pending_records),
            'actually_worked_days': len(actually_present_records),
            'present_days': total_present_days,
            'absent_days': absent_days,
            'late_days': late_days,
            'total_hours': round(total_hours_worked, 2),
            'required_hours': round(required_hours, 2),
            'attendance_percentage': attendance_percentage,  # ✅ HOUR-BASED calculation
            'is_lecturer': is_lecturer,
            'calculation_method': 'hour_based'
        }
        
        logger.info(f"📊 Summary calculated (HOUR-BASED):")
        logger.info(f"   Hours worked: {round(total_hours_worked, 1)}h")
        logger.info(f"   Hours required: {round(required_hours, 1)}h")
        logger.info(f"   Attendance %: {attendance_percentage}% (HOUR-BASED)")
        
        return {
            'records': records,
            'summary': summary
        }

    def _calculate_expected_daily_hours(self, employee: Employee, target_date: date) -> float:
        """Calculate expected working hours for a specific day based on shift schedule"""
        
        if not employee.shift:
            return 9.0  # Default 9 hours if no shift
        
        shift = employee.shift
        weekday = target_date.weekday()  # 0=Monday, 6=Sunday
        
        shift_start_time = None
        shift_end_time = None
        
        if shift.shift_type == 'standard':
            if weekday < 5:  # Monday-Friday
                shift_start_time = shift.weekday_start  # Already a time object
                shift_end_time = shift.weekday_end      # Already a time object
            elif weekday == 5:  # Saturday
                shift_start_time = shift.saturday_start
                shift_end_time = shift.saturday_end
            else:  # Sunday
                shift_start_time = shift.sunday_start
                shift_end_time = shift.sunday_end
        else:  # Dynamic shift
            day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            day_name = day_names[weekday]
            shift_start_time = getattr(shift, f'{day_name}_start')
            shift_end_time = getattr(shift, f'{day_name}_end')
        
        if shift_start_time and shift_end_time:
            # Calculate hours difference
            start_dt = datetime.combine(target_date, shift_start_time)
            end_dt = datetime.combine(target_date, shift_end_time)
            
            # Handle overnight shifts
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)
            
            working_duration = end_dt - start_dt
            hours = working_duration.total_seconds() / 3600
            return round(hours, 2)
        
        return 9.0
    
    def _calculate_required_hours(self, employee: Employee, start_date: date, end_date: date) -> float:
        """Calculate total required working hours for the period - HOUR-BASED calculation"""
        
        is_lecturer = self._is_lecturer(employee)
        
        if is_lecturer:
            # Lecturers: 3 days × 9 hours × weeks in period
            total_days = (end_date - start_date).days + 1
            weeks_in_period = total_days / 7
            # 3 days per week × 9 hours per day = 27 hours per week
            return round(weeks_in_period * 27, 1)
        else:
            # Regular staff: Based on shift schedule and working days
            total_hours = 0.0
            current_date = start_date
            
            while current_date <= end_date:
                # Skip if it would be a day off
                if not self._should_be_day_off(employee, current_date):
                    # Skip public holidays (they get credited automatically)
                    public_holiday = self.db.query(PublicHoliday).filter(
                        and_(
                            PublicHoliday.date == current_date,
                            PublicHoliday.is_active == True
                        )
                    ).first()
                    
                    if not public_holiday:
                        daily_hours = self._calculate_expected_daily_hours(employee, current_date)
                        total_hours += daily_hours
                    else:
                        # Add expected hours for public holidays too (they get credited)
                        daily_hours = self._calculate_expected_daily_hours(employee, current_date)
                        total_hours += daily_hours
                
                current_date += timedelta(days=1)
            
            return round(total_hours, 1)
    
    def calculate_working_days(self, start_date: date, end_date: date, employee_id: int = None) -> int:
        """
        Calculate working days between two dates, considering company holidays and weekends
        Optionally consider employee-specific shift schedule
        IMPROVED: More realistic calculation for lecturers: 3 days per week (Monday-Friday only)
        """
        
        # Get public holidays in the date range
        holidays = self.db.query(PublicHoliday).filter(
            and_(
                PublicHoliday.date >= start_date,
                PublicHoliday.date <= end_date,
                PublicHoliday.is_active == True
            )
        ).all()
        
        holiday_dates = {holiday.date for holiday in holidays}
        
        # Get employee info if provided
        employee_shift = None
        is_lecturer = False
        if employee_id:
            employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
            if employee:
                if employee.shift:
                    employee_shift = employee.shift
                # Check if employee is a lecturer
                is_lecturer = self._is_lecturer(employee)
        
        # ✅ IMPROVED: More realistic calculation for lecturers
        if is_lecturer:
            # Count weekdays only (Monday-Friday)
            weekdays_in_period = 0
            current_date = start_date
            
            while current_date <= end_date:
                # Only count Monday-Friday as potential working days
                if current_date.weekday() < 5:  # 0=Monday, 4=Friday
                    # Skip public holidays
                    if current_date not in holiday_dates:
                        weekdays_in_period += 1
                current_date += timedelta(days=1)
            
            # ✅ IMPROVED: Lecturers work 3 out of every 5 weekdays (more realistic)
            # Use ceiling division to avoid under-counting small periods
            expected_working_days = max(1, int((weekdays_in_period * 3 + 4) // 5))  # Ceiling division
            
            logger.info(f"📊 Lecturer working days calculation:")
            logger.info(f"   Total period: {(end_date - start_date).days + 1} days")
            logger.info(f"   Weekdays in period (excluding holidays): {weekdays_in_period}")
            logger.info(f"   Expected working days (3/5 of weekdays): {expected_working_days}")
            logger.info(f"   Holidays excluded: {len([d for d in holiday_dates if start_date <= d <= end_date and d.weekday() < 5])}")
            
            return expected_working_days
        
        # ✅ EXISTING: Regular calculation for non-lecturers
        working_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            # Skip public holidays
            if current_date in holiday_dates:
                current_date += timedelta(days=1)
                continue
            
            # Check if it's a working day based on shift schedule
            is_working_day = False
            weekday = current_date.weekday()  # 0=Monday, 6=Sunday
            
            if employee_shift:
                if employee_shift.shift_type == 'standard':
                    if weekday < 5:  # Monday-Friday
                        is_working_day = bool(employee_shift.weekday_start and employee_shift.weekday_end)
                    elif weekday == 5:  # Saturday
                        is_working_day = bool(employee_shift.saturday_start and employee_shift.saturday_end)
                    else:  # Sunday
                        is_working_day = bool(employee_shift.sunday_start and employee_shift.sunday_end)
                else:  # Dynamic shift
                    day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                    day_name = day_names[weekday]
                    start_time = getattr(employee_shift, f'{day_name}_start')
                    end_time = getattr(employee_shift, f'{day_name}_end')
                    is_working_day = bool(start_time and end_time)
            else:
                # Default: Monday-Friday are working days
                is_working_day = weekday < 5
            
            if is_working_day:
                working_days += 1
            
            current_date += timedelta(days=1)
        
        return working_days

    def get_employee_debug_info(self, employee_id: int) -> Dict:
        """Get comprehensive debug information for an employee"""
        
        logger.info(f"🔍 Getting debug info for employee {employee_id}")
        
        # Get employee
        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return {"error": "Employee not found"}
        
        # Count raw attendance records
        raw_count = self.db.query(RawAttendance).filter(
            RawAttendance.employee_id == employee_id
        ).count()
        
        # Get recent raw records
        recent_raw = self.db.query(RawAttendance).filter(
            RawAttendance.employee_id == employee_id
        ).order_by(desc(RawAttendance.punch_time)).limit(10).all()
        
        # Count processed attendance records
        processed_count = self.db.query(ProcessedAttendance).filter(
            ProcessedAttendance.employee_id == employee_id
        ).count()
        
        # Get recent processed records
        recent_processed = self.db.query(ProcessedAttendance).filter(
            ProcessedAttendance.employee_id == employee_id
        ).order_by(desc(ProcessedAttendance.date)).limit(10).all()
        
        # Get sync logs
        sync_logs = self.db.query(AttendanceSyncLog).filter(
            AttendanceSyncLog.employee_id == employee_id
        ).order_by(desc(AttendanceSyncLog.created_at)).limit(5).all()
        
        debug_info = {
            "employee": {
                "id": employee.id,
                "employee_id": employee.employee_id,
                "name": f"{employee.first_name} {employee.last_name}",
                "biometric_id": employee.biometric_id,
                "position": employee.position,
                "is_lecturer": self._is_lecturer(employee),
                "shift": employee.shift.name if employee.shift else None,
                "shift_type": employee.shift.shift_type if employee.shift else None,
                "department": employee.department.name if employee.department else None,
                "status": employee.status.value if employee.status else None
            },
            "raw_attendance": {
                "total_count": raw_count,
                "recent_records": [
                    {
                        "id": r.id,
                        "punch_time": r.punch_time.isoformat(),
                        "punch_state": r.punch_state,
                        "punch_state_display": r.punch_state_display,
                        "biometric_record_id": r.biometric_record_id
                    }
                    for r in recent_raw
                ]
            },
            "processed_attendance": {
                "total_count": processed_count,
                "recent_records": [
                    {
                        "id": r.id,
                        "date": r.date.isoformat(),
                        "check_in": r.check_in_time.isoformat() if r.check_in_time else None,
                        "check_out": r.check_out_time.isoformat() if r.check_out_time else None,
                        "status": r.status,
                        "is_present": r.is_present,
                        "is_late": r.is_late,
                        "hours": r.total_working_hours,
                        "late_minutes": r.late_minutes,
                        "notes": r.notes
                    }
                    for r in recent_processed
                ]
            },
            "sync_logs": [
                {
                    "id": log.id,
                    "date": log.created_at.isoformat(),
                    "status": log.sync_status,
                    "records_fetched": log.records_fetched,
                    "records_processed": log.records_processed,
                    "sync_date_range": f"{log.sync_start_date} to {log.sync_end_date}",
                    "error": log.error_message
                }
                for log in sync_logs
            ]
        }
        
        logger.info(f"🔍 Debug info compiled for employee {employee_id}")
        return debug_info