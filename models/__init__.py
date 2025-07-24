# models/__init__.py
"""
Models package initialization
Ensures proper import order for SQLAlchemy relationships
"""

from .database import Base, engine, SessionLocal, get_db, create_tables
from .employee import Employee, User, Department
from .attendance import RawAttendance, ProcessedAttendance, AttendanceSyncLog  
from .leave import Leave, LeaveType, LeaveBalance, PublicHoliday
from .performance import PerformanceRecord, Badge, EmployeeBadge
from .shift import Shift
from .custom_fields import CustomField
from .report_template import ReportTemplate

# Configure all relationships after imports
def configure_relationships():
    """Configure SQLAlchemy relationships after all models are imported"""
    # This ensures all models are available when relationships are established
    Base.registry.configure()

__all__ = [
    'Base', 'engine', 'SessionLocal', 'get_db', 'create_tables',
    'Employee', 'User', 'Department',
    'RawAttendance', 'ProcessedAttendance', 'AttendanceSyncLog',
    'Leave', 'LeaveType', 'LeaveBalance', 'PublicHoliday',
    'PerformanceRecord', 'Badge', 'EmployeeBadge',
    'Shift', 'CustomField', 'ReportTemplate',
    'configure_relationships'
]