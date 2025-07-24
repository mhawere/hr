"""
Leave Management Models
Database models for employee leave management system
"""

from sqlalchemy import Column, Integer, String, DateTime, Date, Float, Boolean, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from models.database import Base
from datetime import datetime, date
import enum

class LeaveStatusEnum(enum.Enum):
    ACTIVE = "Active"
    CANCELLED = "Cancelled"
    COMPLETED = "Completed"

class Leave(Base):
    """Employee leave records"""
    __tablename__ = "leaves"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    leave_type_id = Column(Integer, ForeignKey("leave_types.id"), nullable=False)
    
    # Leave period
    start_date = Column(Date, nullable=False, index=True)
    end_date = Column(Date, nullable=False, index=True)
    days_requested = Column(Float, nullable=False)  # Calculated working days
    
    # Details
    reason = Column(Text, nullable=False)
    comments = Column(Text, nullable=True)
    attachment = Column(String(255), nullable=True)  # File path
    
    # Status
    status = Column(Enum(LeaveStatusEnum), default=LeaveStatusEnum.ACTIVE)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships - Use string references to avoid import issues
    employee = relationship("Employee", back_populates="leaves")
    leave_type = relationship("LeaveType", back_populates="leaves")
    created_by_user = relationship("User", foreign_keys=[created_by])

class LeaveType(Base):
    """Types of leave (Annual, Sick, Maternal, etc.)"""
    __tablename__ = "leave_types"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False, unique=True)
    max_days_per_year = Column(Float, nullable=False)
    color = Column(String(7), default="#3B82F6")  # Hex color for calendar display
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    leaves = relationship("Leave", back_populates="leave_type")

class PublicHoliday(Base):
    """Public holidays that don't count as leave days"""
    __tablename__ = "public_holidays"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class LeaveBalance(Base):
    """Employee leave balances by type and year"""
    __tablename__ = "leave_balances"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    leave_type_id = Column(Integer, ForeignKey("leave_types.id"), nullable=False)
    year = Column(Integer, nullable=False, index=True)
    
    # Balance calculations
    earned_days = Column(Float, default=0.0)  # Based on working days attendance
    max_entitled_days = Column(Float, default=0.0)  # Based on calendar months
    used_days = Column(Float, default=0.0)
    remaining_days = Column(Float, default=0.0)
    
    # Metadata
    last_calculated = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships - Use string references
    employee = relationship("Employee", foreign_keys=[employee_id])
    leave_type = relationship("LeaveType", foreign_keys=[leave_type_id])