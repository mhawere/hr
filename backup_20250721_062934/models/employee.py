from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Enum, Date
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime, date
from models.report_template import ReportTemplate

import enum

class StatusEnum(enum.Enum):
    ACTIVE = "Active"
    NOT_ACTIVE = "Not Active"

class ContractStatusEnum(enum.Enum):
    ACTIVE = "Active"
    EXPIRED = "Expired"
    SUSPENDED = "Suspended"
    CANCELED = "Canceled"

class GenderEnum(enum.Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

class MaritalStatusEnum(enum.Enum):
    SINGLE = "Single"
    MARRIED = "Married"
    DIVORCED = "Divorced"
    WIDOWED = "Widowed"
    SEPARATED = "Separated"

class BloodGroupEnum(enum.Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"

class EmploymentTypeEnum(enum.Enum):
    FULL_TIME = "Full Time"
    PART_TIME = "Part Time"
    CONTRACT = "Contract"
    INTERNSHIP = "Internship"

class Employee(Base):
    __tablename__ = "employees"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), unique=True, index=True)  # System generated
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True)
    phone = Column(String(20))
    biometric_id = Column(String(20), nullable=True, index=True, unique=True)
    photo = Column(String(255))  # Path to photo file
    department_id = Column(Integer, ForeignKey("departments.id"))
    position = Column(String(100))
    hire_date = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(StatusEnum), default=StatusEnum.ACTIVE)
    contract_status = Column(Enum(ContractStatusEnum), default=ContractStatusEnum.ACTIVE)
    address = Column(Text)
    emergency_contact = Column(String(255))
    custom_fields = Column(Text)  # JSON string for custom field values
    shift_id = Column(Integer, ForeignKey("shifts.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    education = Column(Text, nullable=True)
    performance_records = relationship("PerformanceRecord", back_populates="employee")
    badges = relationship("EmployeeBadge", back_populates="employee")
    exceptions = relationship("AttendanceException", back_populates="employee")

    
    # New fields
    date_of_birth = Column(Date, nullable=True)
    gender = Column(Enum(GenderEnum), nullable=True)
    nationality = Column(String(100), nullable=True)
    national_id_number = Column(String(50), nullable=True)  # For Ugandans
    passport_number = Column(String(50), nullable=True)     # For non-Ugandans
    marital_status = Column(Enum(MaritalStatusEnum), nullable=True)
    religion = Column(String(100), nullable=True)
    blood_group = Column(Enum(BloodGroupEnum), nullable=True)
    tin_number = Column(String(50), nullable=True)
    nssf_number = Column(String(50), nullable=True)
    start_of_employment = Column(Date, nullable=True)
    end_of_employment = Column(Date, nullable=True)
    employment_type = Column(Enum(EmploymentTypeEnum), nullable=True)
    
    # Bank details
    bank_name = Column(String(100), nullable=True)
    branch_name = Column(String(100), nullable=True)
    account_title = Column(String(100), nullable=True)
    account_number = Column(String(50), nullable=True)
    
    # Relationships
    raw_attendance_records = relationship("RawAttendance", back_populates="employee")
    processed_attendance_records = relationship("ProcessedAttendance", back_populates="employee")
    leaves = relationship("Leave", back_populates="employee")
    department = relationship("Department", foreign_keys=[department_id], back_populates="employees")
    shift = relationship("Shift", foreign_keys=[shift_id], back_populates="employees")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)
    role = Column(String(20), default="HR")
    created_at = Column(DateTime, default=datetime.utcnow)

class Department(Base):
    __tablename__ = "departments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    code = Column(String(10), nullable=False, unique=True)  # For employee ID generation
    description = Column(Text)
    manager_id = Column(Integer, ForeignKey("employees.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships - specify foreign_keys to avoid ambiguity
    employees = relationship("Employee", foreign_keys=[Employee.department_id], back_populates="department")
    manager = relationship("Employee", foreign_keys=[manager_id], post_update=True)