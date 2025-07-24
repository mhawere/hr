"""
Performance Management Database Models
"""

import enum
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Enum as SQLEnum, Date
from sqlalchemy.orm import relationship
from datetime import datetime, date

from .database import Base
from .employee import Employee, User

class PerformanceType(str, enum.Enum):
    ACHIEVEMENT = "achievement"
    WARNING = "warning"
    DISCIPLINARY = "disciplinary"
    COMMENDATION = "commendation"
    TRAINING = "training"
    GOAL_ACHIEVEMENT = "goal_achievement"
    PERFORMANCE_REVIEW = "performance_review"
    PROBATION = "probation"
    PROMOTION = "promotion"
    BONUS = "bonus"

class BadgeCategory(str, enum.Enum):
    ATTENDANCE = "attendance"
    TENURE = "tenure"
    PERFORMANCE = "performance"
    TRAINING = "training"
    TEAMWORK = "teamwork"
    INNOVATION = "innovation"
    SAFETY = "safety"
    CUSTOMER_SERVICE = "customer_service"
    LEADERSHIP = "leadership"

class BadgeLevel(str, enum.Enum):
    BRONZE = "bronze"
    SILVER = "silver" 
    GOLD = "gold"
    PLATINUM = "platinum"

class PerformanceRecord(Base):
    __tablename__ = "performance_records"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    record_type = Column(SQLEnum(PerformanceType), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    points = Column(Integer, default=0)
    effective_date = Column(Date, default=date.today)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    employee = relationship("Employee", back_populates="performance_records")
    created_by_user = relationship("User")

class Badge(Base):
    __tablename__ = "badges"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    category = Column(SQLEnum(BadgeCategory), nullable=False)
    level = Column(SQLEnum(BadgeLevel), nullable=False)
    criteria = Column(Text)
    points_required = Column(Integer, default=0)
    icon = Column(String(50), default="fas fa-award")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    employee_badges = relationship("EmployeeBadge", back_populates="badge")

class EmployeeBadge(Base):
    __tablename__ = "employee_badges"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    badge_id = Column(Integer, ForeignKey("badges.id"), nullable=False)
    earned_date = Column(DateTime, default=datetime.utcnow)
    awarded_by = Column(Integer, ForeignKey("users.id"))
    notes = Column(Text)
    
    # Relationships
    employee = relationship("Employee", back_populates="badges")
    badge = relationship("Badge", back_populates="employee_badges")
    awarded_by_user = relationship("User")