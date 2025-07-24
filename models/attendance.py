"""
Attendance Management Models
Database models for storing raw and processed attendance data
"""

from sqlalchemy import Column, Integer, String, DateTime, Date, Time, Float, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from models.database import Base
from datetime import datetime, date

class RawAttendance(Base):
    """Raw attendance data from biometric devices"""
    __tablename__ = "raw_attendance"
    
    id = Column(Integer, primary_key=True, index=True)
    biometric_record_id = Column(Integer, unique=True, index=True)  # API record ID
    emp_code = Column(String(50), index=True, nullable=False)  # From biometric device
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)  # Our employee mapping
    
    # Raw data from API
    punch_time = Column(DateTime, nullable=False)
    punch_state = Column(String(10), nullable=False)  # "0" = in, "1" = out
    punch_state_display = Column(String(50), nullable=True)  # "Check In", "Check Out"
    verify_type = Column(Integer, nullable=True)  # 1=Fingerprint, 15=Face
    verify_type_display = Column(String(50), nullable=True)
    
    # Device information
    terminal_sn = Column(String(100), nullable=True)
    terminal_alias = Column(String(100), nullable=True)
    area_alias = Column(String(100), nullable=True)
    temperature = Column(Float, nullable=True)
    
    # Metadata
    upload_time = Column(DateTime, nullable=True)  # When device uploaded to API
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    employee = relationship("Employee", back_populates="raw_attendance_records")

class ProcessedAttendance(Base):
    """Daily processed attendance with calculations"""
    __tablename__ = "processed_attendance"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    emp_code = Column(String(50), index=True, nullable=False)  # Biometric ID for reference
    date = Column(Date, nullable=False, index=True)
    
    # Attendance times
    check_in_time = Column(DateTime, nullable=True)
    check_out_time = Column(DateTime, nullable=True)
    
    # Shift information (captured at time of processing)
    shift_id = Column(Integer, ForeignKey("shifts.id"), nullable=True)
    expected_start_time = Column(Time, nullable=True)  # Shift start time when recorded
    expected_end_time = Column(Time, nullable=True)    # Shift end time when recorded
    
    # Calculated values
    total_working_minutes = Column(Integer, default=0)  # Minutes worked
    total_working_hours = Column(Float, default=0.0)    # Hours worked (decimal)
    
    # Status indicators
    is_present = Column(Boolean, default=False)
    is_late = Column(Boolean, default=False)
    is_early_departure = Column(Boolean, default=False)
    late_minutes = Column(Integer, default=0)
    early_departure_minutes = Column(Integer, default=0)
    
    # Leave and holiday tracking
    leave_id = Column(Integer, ForeignKey("leaves.id"), nullable=True)
    leave_type = Column(String(50), nullable=True)  # Leave type name for quick reference
    holiday_name = Column(String(100), nullable=True)  # Public holiday name
    
    # Status
    status = Column(String(20), default="absent")  # present, absent, half_day, late, day_off, on_leave, public_holiday, pending, in_progress
    notes = Column(Text, nullable=True)  # For manual adjustments
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    processed_by = Column(Integer, ForeignKey("users.id"), nullable=True)  # Who processed/modified
    
    # Relationships
    employee = relationship("Employee", back_populates="processed_attendance_records")
    shift = relationship("Shift")
    leave = relationship("Leave", foreign_keys=[leave_id])
    processed_by_user = relationship("User")

class AttendanceSyncLog(Base):
    """Track sync operations for each employee"""
    __tablename__ = "attendance_sync_log"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    emp_code = Column(String(50), index=True, nullable=False)
    
    # Sync details
    last_sync_date = Column(Date, nullable=False)
    sync_start_date = Column(Date, nullable=False)  # What date range was synced
    sync_end_date = Column(Date, nullable=False)
    
    # Results
    records_fetched = Column(Integer, default=0)
    records_processed = Column(Integer, default=0)
    sync_status = Column(String(20), default="success")  # success, failed, partial
    error_message = Column(Text, nullable=True)
    
    # Metadata
    sync_duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    synced_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    employee = relationship("Employee")
    synced_by_user = relationship("User")