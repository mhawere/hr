"""
Staff Management Router
Production-ready implementation with comprehensive employee management,
photo upload, ID card generation, document upload, and custom fields support.
Enhanced with performance optimizations, server-side pagination, and filtering.
"""

import logging
import json
import os
import uuid
import io
import base64
import csv
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta, date
from urllib.parse import urlencode
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy import and_, or_, func, desc, asc, text

import qrcode
from fastapi import (
    APIRouter, Request, Form, Depends, HTTPException, UploadFile, File, 
    Query, BackgroundTasks
)
from fastapi import status as http_status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from weasyprint import HTML, CSS
from pydantic import BaseModel, validator
from PIL import Image

# Import your existing models and dependencies
from models.database import get_db
from models.employee import (
    Employee, User, Department, StatusEnum, ContractStatusEnum,
    GenderEnum, MaritalStatusEnum, BloodGroupEnum, EmploymentTypeEnum
)
from routers.attendance import Shift
from models.custom_fields import CustomField
from utils.auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Constants
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
IMAGE_DIMENSIONS = (800, 800)  # Max dimensions for resizing

ALLOWED_DOCUMENT_EXTENSIONS = {'.pdf'}
MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB

# Upload directories
PHOTO_UPLOAD_DIR = Path("static/uploads/photos")
DOCUMENT_UPLOAD_DIR = Path("static/uploads/documents")

# Cache for frequently accessed data
_department_cache = {}
_department_cache_timestamp = None
CACHE_DURATION = 300  # 5 minutes

# Pydantic models for validation
class EmployeeCreate(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = ""
    biometric_id: Optional[str] = ""
    department_id: int
    position: str
    employment_status: str
    contract_status: str
    address: Optional[str] = ""
    emergency_contact: Optional[str] = ""
    shift_id: Optional[int] = None
    custom_fields: Optional[Dict[str, Any]] = {}
    
    @validator('first_name', 'last_name')
    def validate_names(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters long')
        if len(v.strip()) > 50:
            raise ValueError('Name must be less than 50 characters')
        return v.strip().title()
    
    @validator('email')
    def validate_email(cls, v):
        import re
        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', v):
            raise ValueError('Invalid email format')
        return v.lower().strip()
    
    @validator('position')
    def validate_position(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Position must be at least 2 characters long')
        return v.strip()

class EmployeeUpdate(EmployeeCreate):
    shift_id: Optional[int] = None

class EmployeeResponse(BaseModel):
    id: int
    employee_id: str
    first_name: str
    last_name: str
    email: str
    phone: Optional[str]
    photo: Optional[str]
    biometric_id: Optional[str]
    department_id: Optional[int]
    position: str
    status: str
    contract_status: str
    address: Optional[str]
    emergency_contact: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Helper Functions
def ensure_upload_directories():
    """Ensure upload directories exist with proper permissions"""
    try:
        PHOTO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        DOCUMENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set appropriate permissions (readable/writable by owner, readable by group)
        os.chmod(PHOTO_UPLOAD_DIR, 0o755)
        os.chmod(DOCUMENT_UPLOAD_DIR, 0o755)
        
        logger.info("Upload directories initialized successfully")
    except Exception as e:
        logger.error(f"Failed to create upload directories: {str(e)}")
        raise

def get_cached_departments(db: Session) -> List[Department]:
    """Get departments with caching for better performance"""
    global _department_cache, _department_cache_timestamp
    
    current_time = datetime.utcnow()
    
    # Check if cache is valid
    if (_department_cache_timestamp and 
        (current_time - _department_cache_timestamp).seconds < CACHE_DURATION and
        _department_cache):
        return _department_cache
    
    # Refresh cache
    try:
        departments = db.query(Department).filter(Department.is_active == True).order_by(Department.name).all()
        _department_cache = departments
        _department_cache_timestamp = current_time
        return departments
    except Exception as e:
        logger.error(f"Error fetching departments: {str(e)}")
        return _department_cache if _department_cache else []

def build_pagination_url(page: int, request_args: dict) -> str:
    """Helper function to build pagination URLs with current filters"""
    params = dict(request_args)
    params['page'] = page
    # Remove empty parameters
    params = {k: v for k, v in params.items() if v and v != ''}
    return f"/staff/view?{urlencode(params)}"

def generate_employee_id(department_code: str, db: Session) -> str:
    """Generate system employee ID based on department with collision handling"""
    try:
        # Get count of employees in this department
        count = db.query(Employee).join(
            Department, 
            Employee.department_id == Department.id
        ).filter(Department.code == department_code).count()
        
        # Try generating ID with incrementing numbers
        for attempt in range(10):  # Max 10 attempts
            next_number = count + attempt + 1
            potential_id = f"{department_code}{next_number:04d}"
            
            # Check if this ID already exists
            existing = db.query(Employee).filter(Employee.employee_id == potential_id).first()
            if not existing:
                return potential_id
        
        # Fallback to timestamp-based ID if collision persists
        timestamp = int(datetime.now().timestamp())
        fallback_id = f"{department_code}{timestamp % 10000:04d}"
        logger.warning(f"Using fallback employee ID: {fallback_id}")
        return fallback_id
        
    except Exception as e:
        logger.error(f"Error generating employee ID: {str(e)}")
        # Ultimate fallback
        timestamp = int(datetime.now().timestamp())
        return f"EMP{timestamp % 100000:05d}"

def validate_image_file(photo: UploadFile) -> bool:
    """Validate uploaded image file with comprehensive checks"""
    if not photo.filename:
        return False
    
    # Check file extension
    file_extension = Path(photo.filename).suffix.lower()
    if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid image format. Allowed formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )
    
    # Check MIME type
    if photo.content_type and not photo.content_type.startswith('image/'):
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only image files are allowed."
        )
    
    # Check file size
    photo.file.seek(0, 2)  # Seek to end
    file_size = photo.file.tell()
    photo.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Image too large. Maximum size: {MAX_IMAGE_SIZE // (1024*1024)}MB"
        )
    
    if file_size < 100:  # Minimum 100 bytes
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Image file appears to be corrupted or empty"
        )
    
    return True

def optional_int_query(value: str = None) -> Optional[int]:
    """Convert empty string or None to None, otherwise parse as int"""
    if not value or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None

def safely_parse_custom_fields(custom_fields_data) -> Dict[str, Any]:
    """Safely parse custom fields data, ensuring it returns a dictionary"""
    if not custom_fields_data:
        return {}
    
    try:
        if isinstance(custom_fields_data, str):
            parsed = json.loads(custom_fields_data)
            return parsed if isinstance(parsed, dict) else {}
        elif isinstance(custom_fields_data, dict):
            return custom_fields_data
        else:
            return {}
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        logger.warning(f"Failed to parse custom fields data: {str(e)}")
        return {}

def validate_document_file(document: UploadFile) -> bool:
    """Validate uploaded document file with security checks"""
    if not document.filename:
        return False
    
    # Check file extension
    file_extension = Path(document.filename).suffix.lower()
    if file_extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are allowed."
        )
    
    # Check MIME type for additional security
    if document.content_type and document.content_type not in ['application/pdf']:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are allowed."
        )
    
    # Check file size
    document.file.seek(0, 2)  # Seek to end
    file_size = document.file.tell()
    document.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_DOCUMENT_SIZE:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Document too large. Maximum size: {MAX_DOCUMENT_SIZE // (1024*1024)}MB"
        )
    
    if file_size < 100:  # Minimum 100 bytes
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Document file appears to be corrupted or empty"
        )
    
    return True

def resize_image(image_path: Path, max_dimensions: tuple = IMAGE_DIMENSIONS) -> None:
    """Resize image to fit within max dimensions while maintaining aspect ratio"""
    try:
        with Image.open(image_path) as img:
            # Verify it's actually an image
            img.verify()
            
        # Reopen for processing (verify() consumes the image)
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Only resize if image is larger than max dimensions
            if img.size[0] > max_dimensions[0] or img.size[1] > max_dimensions[1]:
                img.thumbnail(max_dimensions, Image.Resampling.LANCZOS)
            
            # Save with optimization
            img.save(image_path, 'JPEG', quality=85, optimize=True)
            
        logger.info(f"Image resized successfully: {image_path}")
            
    except Exception as e:
        logger.error(f"Error resizing image {image_path}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing image file"
        )

def save_uploaded_photo(photo: UploadFile) -> str:
    """Save uploaded photo with validation and processing"""
    if not photo.filename:
        return ""
    
    validate_image_file(photo)
    ensure_upload_directories()
    
    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}.jpg"  # Always save as JPG for consistency
    file_path = PHOTO_UPLOAD_DIR / unique_filename
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            content = photo.file.read()
            buffer.write(content)
        
        # Resize and optimize image
        resize_image(file_path)
        
        logger.info(f"Photo saved: {unique_filename}")
        return f"uploads/photos/{unique_filename}"
        
    except Exception as e:
        # Clean up file if error occurred
        if file_path.exists():
            file_path.unlink()
        logger.error(f"Error saving photo: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error saving photo"
        )

def save_uploaded_document(document: UploadFile) -> str:
    """Save uploaded document with validation and security checks"""
    if not document.filename:
        return ""
    
    validate_document_file(document)
    ensure_upload_directories()
    
    # Generate unique filename preserving extension
    file_extension = Path(document.filename).suffix.lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = DOCUMENT_UPLOAD_DIR / unique_filename
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            content = document.file.read()
            buffer.write(content)
        
        # Additional PDF validation by trying to read the file
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                if header != b'%PDF':
                    raise ValueError("Not a valid PDF file")
        except Exception:
            file_path.unlink()  # Delete invalid file
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file format"
            )
        
        logger.info(f"Document saved: {unique_filename}")
        return f"uploads/documents/{unique_filename}"
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file if error occurred
        if file_path.exists():
            file_path.unlink()
        logger.error(f"Error saving document: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error saving document"
        )

def delete_file(file_path: str) -> None:
    """Safely delete a file with proper error handling"""
    if not file_path:
        return
    
    try:
        full_path = Path("static") / file_path
        if full_path.exists() and full_path.is_file():
            # Additional security check - ensure file is within allowed directories
            allowed_dirs = [PHOTO_UPLOAD_DIR, DOCUMENT_UPLOAD_DIR]
            if not any(str(full_path).startswith(str(allowed_dir)) for allowed_dir in allowed_dirs):
                logger.warning(f"Attempted to delete file outside allowed directories: {file_path}")
                return
            
            full_path.unlink()
            logger.info(f"File deleted: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")

def cleanup_files_on_error(uploaded_files: List[str]) -> None:
    """Clean up uploaded files when an error occurs"""
    for file_path in uploaded_files:
        if file_path:
            delete_file(file_path)

def get_employee_by_id(db: Session, employee_id: int) -> Employee:
    """Get employee by ID with error handling and eager loading"""
    try:
        employee = db.query(Employee).options(
            selectinload(Employee.department)
        ).filter(Employee.id == employee_id).first()
        
        if not employee:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Employee with ID {employee_id} not found"
            )
        return employee
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error while fetching employee {employee_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error accessing employee data"
        )
    
def check_biometric_id_exists(db: Session, biometric_id: str, exclude_id: Optional[int] = None) -> bool:
    """Check if biometric ID already exists with proper error handling"""
    if not biometric_id or not biometric_id.strip():
        return False
    
    try:
        query = db.query(Employee).filter(Employee.biometric_id.ilike(biometric_id.strip()))
        if exclude_id:
            query = query.filter(Employee.id != exclude_id)
        return query.first() is not None
    except Exception as e:
        logger.error(f"Database error while checking biometric ID existence: {str(e)}")
        return False

def check_email_exists(db: Session, email: str, exclude_id: Optional[int] = None) -> bool:
    """Check if email already exists with proper error handling"""
    try:
        query = db.query(Employee).filter(Employee.email.ilike(email.strip()))
        if exclude_id:
            query = query.filter(Employee.id != exclude_id)
        return query.first() is not None
    except Exception as e:
        logger.error(f"Database error while checking email existence: {str(e)}")
        return False

def assign_employee_to_shift(employee_id: int, shift_id: int, db: Session):
    """Helper function to assign employee to shift and update count"""
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            logger.error(f"Employee {employee_id} not found when assigning to shift")
            return False
        
        old_shift_id = employee.shift_id
        employee.shift_id = shift_id
        
        # Update counts for both old and new shifts
        if old_shift_id:
            old_shift = db.query(Shift).filter(Shift.id == old_shift_id).first()
            if old_shift and hasattr(old_shift, 'update_employee_count'):
                old_shift.update_employee_count(db)
                logger.info(f"Updated employee count for old shift {old_shift_id}")
        
        if shift_id:
            new_shift = db.query(Shift).filter(Shift.id == shift_id).first()
            if new_shift and hasattr(new_shift, 'update_employee_count'):
                new_shift.update_employee_count(db)
                logger.info(f"Updated employee count for new shift {shift_id}")
        
        db.commit()
        logger.info(f"Employee {employee_id} assigned to shift {shift_id}")
        return True
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error assigning employee {employee_id} to shift {shift_id}: {e}")
        return False

def generate_qr_code(data: str) -> str:
    """Generate QR code and return as base64 string with error handling"""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error generating QR code: {str(e)}")
        return ""

def generate_qr_code_buffer(data: str) -> bytes:
    """Generate QR code and return as bytes"""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error generating QR code: {str(e)}")
        return b""

def get_logo_path() -> str:
    """Get logo path with fallback"""
    logo_paths = [
        Path("static/images/logo.png"),
        Path("static/images/logo.jpg"),
        Path("static/images/company_logo.png"),
        Path("static/assets/logo.png")
    ]
    
    for logo_path in logo_paths:
        if logo_path.exists():
            return f"/static/{logo_path.relative_to(Path('static'))}"
    
    return ""

def get_photo_path(employee: Employee) -> str:
    """Get employee photo path with fallback"""
    if employee.photo:
        photo_path = Path("static") / employee.photo
        if photo_path.exists():
            return f"/static/{employee.photo}"
    return ""

def process_education_data(form_data) -> List[Dict[str, Any]]:
    """Process education data from form submission"""
    education_entries = []
    education_indices = set()
    
    # Find all education field indices
    for key in form_data.keys():
        if key.startswith('education_level_'):
            try:
                index = int(key.split('_')[-1])
                education_indices.add(index)
            except ValueError:
                continue
    
    # Process each education entry
    for index in sorted(education_indices):
        education_level = form_data.get(f'education_level_{index}', '').strip()
        institution_name = form_data.get(f'institution_name_{index}', '').strip()
        field_of_study = form_data.get(f'field_of_study_{index}', '').strip()
        degree_title = form_data.get(f'degree_title_{index}', '').strip()
        graduation_year = form_data.get(f'graduation_year_{index}', '').strip()
        gpa_grade = form_data.get(f'gpa_grade_{index}', '').strip()
        
        # Only add entry if at least education level or institution is provided
        if education_level or institution_name:
            education_entry = {
                'education_level': education_level,
                'institution_name': institution_name,
                'field_of_study': field_of_study,
                'degree_title': degree_title,
                'graduation_year': int(graduation_year) if graduation_year.isdigit() else None,
                'gpa_grade': gpa_grade
            }
            education_entries.append(education_entry)
    
    return education_entries

def validate_education_data(education_entries: List[Dict[str, Any]]) -> None:
    """Validate education data entries"""
    for i, entry in enumerate(education_entries):
        # Validate required fields if any education data is provided
        if any(entry.values()):
            if not entry.get('education_level'):
                raise HTTPException(
                    status_code=http_status.HTTP_400_BAD_REQUEST,
                    detail=f"Education level is required for education entry #{i + 1}"
                )
            if not entry.get('institution_name'):
                raise HTTPException(
                    status_code=http_status.HTTP_400_BAD_REQUEST,
                    detail=f"Institution name is required for education entry #{i + 1}"
                )
            
            # Validate graduation year if provided
            if entry.get('graduation_year'):
                year = entry['graduation_year']
                current_year = datetime.now().year
                if year < 1950 or year > current_year + 10:
                    raise HTTPException(
                        status_code=http_status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid graduation year for education entry #{i + 1}. Must be between 1950 and {current_year + 10}"
                    )

def safely_parse_education_data(education_data) -> List[Dict[str, Any]]:
    """Safely parse education data, ensuring it returns a list"""
    if not education_data:
        return []
    
    try:
        if isinstance(education_data, str):
            parsed = json.loads(education_data)
            return parsed if isinstance(parsed, list) else []
        elif isinstance(education_data, list):
            return education_data
        else:
            return []
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        logger.warning(f"Failed to parse education data: {str(e)}")
        return []

def process_custom_fields(form_data, custom_fields: List[CustomField], existing_data: Dict = None) -> tuple:
    """Process custom fields from form data, returning (custom_field_data, uploaded_files)"""
    custom_field_data = {}
    uploaded_files = []
    
    # Parse existing custom data if provided
    if existing_data is None:
        existing_data = {}
    
    for field in custom_fields:
        field_key = f"custom_{field.field_name}"
        
        try:
            if field.field_type == 'document':
                # Handle document upload
                if field_key in form_data:
                    uploaded_file = form_data[field_key]
                    if hasattr(uploaded_file, 'filename') and uploaded_file.filename:
                        try:
                            # Delete old document if exists (only for updates)
                            old_document = existing_data.get(field.field_name)
                            if old_document:
                                delete_file(old_document)
                            
                            # Save new document
                            document_path = save_uploaded_document(uploaded_file)
                            custom_field_data[field.field_name] = document_path
                            uploaded_files.append(document_path)
                        except HTTPException as e:
                            # Clean up any files uploaded in this batch
                            cleanup_files_on_error(uploaded_files)
                            raise HTTPException(
                                status_code=http_status.HTTP_400_BAD_REQUEST,
                                detail=f"Error uploading {field.field_label}: {e.detail}"
                            )
                    else:
                        # Keep existing document if no new one uploaded (for updates)
                        if field.field_name in existing_data:
                            custom_field_data[field.field_name] = existing_data[field.field_name]
            else:
                # Handle other field types
                field_value = form_data.get(field_key)
                if field_value and str(field_value).strip():
                    # Additional validation based on field type
                    value = str(field_value).strip()
                    
                    if field.field_type == 'number':
                        try:
                            # Validate numeric input
                            float(value)
                        except ValueError:
                            raise HTTPException(
                                status_code=http_status.HTTP_400_BAD_REQUEST,
                                detail=f"{field.field_label} must be a valid number"
                            )
                    elif field.field_type == 'date':
                        try:
                            # Validate date format
                            datetime.fromisoformat(value)
                        except ValueError:
                            raise HTTPException(
                                status_code=http_status.HTTP_400_BAD_REQUEST,
                                detail=f"{field.field_label} must be a valid date"
                            )
                    
                    custom_field_data[field.field_name] = value
                elif field.field_name in existing_data:
                    # Keep existing value for updates
                    custom_field_data[field.field_name] = existing_data[field.field_name]
                    
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing custom field {field.field_name}: {str(e)}")
            # Continue with other fields but log the error
    
    return custom_field_data, uploaded_files

async def create_id_card_from_html(employee: Employee, request: Request) -> bytes:
    """Create ID card from HTML template"""
    try:
        # Calculate dates
        issue_date = datetime.now()
        expiry_date = issue_date + timedelta(days=365)
        
        # Convert employee photo to base64
        photo_base64 = ""
        if employee.photo:
            try:
                photo_path = Path("static") / employee.photo
                if photo_path.exists():
                    with open(photo_path, "rb") as image_file:
                        photo_bytes = image_file.read()
                        photo_base64 = base64.b64encode(photo_bytes).decode()
                        logger.info(f"Photo converted to base64, size: {len(photo_base64)} chars")
                else:
                    logger.warning(f"Photo file not found: {photo_path}")
            except Exception as e:
                logger.error(f"Error converting photo to base64: {e}")
        
        # Convert logo to base64
        logo_base64 = ""
        try:
            logo_path = Path("static/images/logo.png")
            if logo_path.exists():
                with open(logo_path, "rb") as logo_file:
                    logo_bytes = logo_file.read()
                    logo_base64 = base64.b64encode(logo_bytes).decode()
                    logger.info(f"Logo converted to base64, size: {len(logo_base64)} chars")
            else:
                logger.warning(f"Logo file not found: {logo_path}")
        except Exception as e:
            logger.error(f"Error converting logo to base64: {e}")

        # Convert background image to base64
        bg_base64 = ""
        try:
            bg_path = Path("static/images/bg_id.png")
            if bg_path.exists():
                with open(bg_path, "rb") as bg_file:
                    bg_bytes = bg_file.read()
                    bg_base64 = base64.b64encode(bg_bytes).decode()
                    logger.info(f"Background image converted to base64, size: {len(bg_base64)} chars")
            else:
                logger.warning(f"Background image not found: {bg_path}")
        except Exception as e:
            logger.error(f"Error converting background image to base64: {e}")
        
        # Generate QR code
        qr_code_base64 = ""
        try:
            department_name = employee.department.name if employee.department else "N/A"
            qr_data = f"IUEA-{employee.employee_id}|{employee.first_name} {employee.last_name}|{department_name}|{employee.position or 'Staff'}"
            qr_buffer = generate_qr_code_buffer(qr_data)
            if qr_buffer:
                qr_code_base64 = base64.b64encode(qr_buffer).decode()
                logger.info("QR code generated successfully")
            else:
                logger.warning("QR code generation returned empty buffer")
        except Exception as e:
            logger.error(f"Could not generate QR code: {e}")
        
        # Prepare context for template
        context = {
            "employee": employee,
            "issue_date": issue_date.strftime('%d/%m/%Y'),
            "expiry_date": expiry_date.strftime('%d/%m/%Y'),
            "photo_base64": photo_base64,
            "logo_base64": logo_base64,
            "bg_base64": bg_base64,
            "qr_code_base64": qr_code_base64,
        }
        
        logger.info(f"Rendering ID card for employee: {employee.first_name} {employee.last_name}")
        logger.info(f"Photo base64 available: {bool(photo_base64)}")
        logger.info(f"Logo base64 available: {bool(logo_base64)}")
        logger.info(f"Background base64 available: {bool(bg_base64)}")
        
        # Render HTML template
        html_content = templates.get_template("id.html").render(
            request=request,
            **context
        )
        
        # Convert HTML to PDF
        base_url = str(request.url_for('static', path=''))
        logger.info(f"Base URL for static files: {base_url}")
        
        html_doc = HTML(string=html_content, base_url=base_url)
        pdf_buffer = html_doc.write_pdf()
        
        logger.info("ID card generated successfully using HTML template")
        return pdf_buffer
        
    except Exception as e:
        logger.error(f"Error creating ID card from HTML: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating ID card: {str(e)}"
        )

def create_database_indexes(db: Session):
    """Create database indexes for better performance"""
    try:
        # These would typically be done via migrations, but shown here for reference
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_employee_name ON employees (first_name, last_name);",
            "CREATE INDEX IF NOT EXISTS idx_employee_email ON employees (email);",
            "CREATE INDEX IF NOT EXISTS idx_employee_biometric ON employees (biometric_id);",
            "CREATE INDEX IF NOT EXISTS idx_employee_department ON employees (department_id);",
            "CREATE INDEX IF NOT EXISTS idx_employee_status ON employees (status);",
            "CREATE INDEX IF NOT EXISTS idx_employee_created ON employees (created_at);",
        ]
        
        for index_sql in indexes:
            try:
                db.execute(text(index_sql))
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {str(e)}")
        
        db.commit()
        logger.info("Database indexes created/verified successfully")
        
    except Exception as e:
        logger.error(f"Error creating database indexes: {str(e)}")
        db.rollback()

# Routes
@router.get("/", response_class=HTMLResponse)
async def staff_home(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Staff management home page"""
    return RedirectResponse(url="/staff/view")

@router.get("/view", response_class=HTMLResponse)
async def view_employees(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    search: Optional[str] = Query(None, description="Search employees"),
    department_id: str = Query("", description="Filter by department"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(25, ge=1, le=100, description="Items per page"),
    sort_by: Optional[str] = Query("created_at", description="Sort by field")
):
    """View all employees with optimized filtering and pagination"""
    try:
        department_id = int(department_id) if department_id and department_id.strip() else None
        query = db.query(Employee).options(
            selectinload(Employee.department)
        )
        
        # Apply search filter
        if search and search.strip():
            search_term = f"%{search.strip()}%"
            query = query.filter(
                or_(
                    Employee.first_name.ilike(search_term),
                    Employee.last_name.ilike(search_term),
                    Employee.email.ilike(search_term),
                    Employee.employee_id.ilike(search_term),
                    Employee.position.ilike(search_term),
                    Employee.biometric_id.ilike(search_term)
                )
            )
        
        # Apply department filter
        if department_id:
            query = query.filter(Employee.department_id == department_id)
        
        # Apply status filter
        if status_filter and status_filter.strip():
            try:
                status_enum = StatusEnum(status_filter.strip())
                query = query.filter(Employee.status == status_enum)
            except ValueError:
                logger.warning(f"Invalid status filter: {status_filter}")
        
        # Apply sorting
        if sort_by:
            if sort_by == "name":
                query = query.order_by(Employee.first_name.asc(), Employee.last_name.asc())
            elif sort_by == "employee_id":
                query = query.order_by(Employee.employee_id.asc())
            elif sort_by == "department":
                query = query.join(Department, Employee.department_id == Department.id, isouter=True)\
                            .order_by(Department.name.asc())
            elif sort_by == "created_at":
                query = query.order_by(Employee.created_at.desc())
            else:
                # Default fallback
                query = query.order_by(Employee.created_at.desc())
        else:
            query = query.order_by(Employee.created_at.desc())
        
        # Get total count before pagination (optimized)
        total_count = query.count()
        
        # Get active employee count for stats
        active_count = db.query(Employee).filter(Employee.status == StatusEnum.ACTIVE).count()
        
        # Calculate pagination
        total_pages = (total_count + per_page - 1) // per_page
        if page > total_pages and total_pages > 0:
            page = total_pages
        
        offset = (page - 1) * per_page
        
        # Apply pagination and execute query
        employees = query.offset(offset).limit(per_page).all()
        
        # Get departments for filter dropdown (cached query)
        departments = get_cached_departments(db)
        
        # Get custom fields for display
        custom_fields = db.query(CustomField).filter(CustomField.is_active == True).all()
        
        # Prepare pagination helper function
        def build_pagination_url_helper(target_page):
            params = {
                'page': target_page,
                'per_page': per_page,
                'sort_by': sort_by
            }
            if search:
                params['search'] = search
            if department_id:
                params['department_id'] = department_id
            if status_filter:
                params['status_filter'] = status_filter
            
            # Remove empty parameters
            params = {k: v for k, v in params.items() if v and v != ''}
            return f"/staff/view?{urlencode(params)}"
        
        context = {
            "request": request,
            "user": current_user,
            "employees": employees,
            "custom_fields": custom_fields,
            "departments": departments,
            "total_count": total_count,
            "active_count": active_count,
            "current_page": page,
            "total_pages": total_pages,
            "per_page": per_page,
            "search": search or "",
            "department_id": department_id,
            "status_filter": status_filter,
            "sort_by": sort_by,
            "status_options": [emp_status.value for emp_status in StatusEnum],
            "build_pagination_url": build_pagination_url_helper,
            "page_title": "View Employees"
        }
        
        return templates.TemplateResponse("staff/view_employees.html", context)
        
    except Exception as e:
        logger.error(f"Error viewing employees: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading employees"
        )

@router.get("/add", response_class=HTMLResponse)
async def add_employee_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Display add employee form"""
    try:
        custom_fields = db.query(CustomField).filter(CustomField.is_active == True).all()
        departments = get_cached_departments(db)
        shifts = db.query(Shift).filter(Shift.is_active == True).order_by(Shift.name).all()

        # Get enum values safely
        employment_status_options = [emp_status.value for emp_status in StatusEnum]
        contract_status_options = [cont_status.value for cont_status in ContractStatusEnum]
        
        context = {
            "request": request,
            "user": current_user,
            "custom_fields": custom_fields,
            "departments": departments,
            "shifts": shifts,
            "status_options": employment_status_options,
            "contract_status_options": contract_status_options,
            "page_title": "Add Employee"
        }
        return templates.TemplateResponse("staff/add_employee.html", context)
        
    except Exception as e:
        logger.error(f"Error loading add employee page: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading form"
        )

@router.post("/add")
async def add_employee(
    request: Request,
    background_tasks: BackgroundTasks,
    # Basic Information
    first_name: str = Form(..., min_length=2, max_length=50),
    last_name: str = Form(..., min_length=2, max_length=50),
    email: str = Form(...),
    phone: str = Form(""),
    biometric_id: str = Form(""),
    department_id: int = Form(...),
    position: str = Form(..., min_length=2),
    status: str = Form(...),
    contract_status: str = Form(...),
    shift_id: Optional[int] = Form(None),
    photo: UploadFile = File(None),
    
    # Personal Details
    date_of_birth: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    nationality: Optional[str] = Form(None),
    national_id_number: Optional[str] = Form(""),
    passport_number: Optional[str] = Form(""),
    marital_status: Optional[str] = Form(None),
    religion: Optional[str] = Form(None),
    blood_group: Optional[str] = Form(None),
    
    # Tax & Employment Details
    tin_number: Optional[str] = Form(""),
    nssf_number: Optional[str] = Form(""),
    start_of_employment: Optional[str] = Form(None),
    end_of_employment: Optional[str] = Form(None),
    employment_type: Optional[str] = Form(None),
    
    # Bank Details
    bank_name: Optional[str] = Form(None),
    branch_name: Optional[str] = Form(""),
    account_title: Optional[str] = Form(""),
    account_number: Optional[str] = Form(""),
    
    # Additional Information
    address: str = Form(""),
    emergency_contact: str = Form(""),
    
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new employee"""
    uploaded_files = []
    photo_path = ""
    
    try:
        # Validate employee data - only basic fields for validation
        employee_data = EmployeeCreate(
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            biometric_id=biometric_id,
            department_id=department_id,
            position=position,
            employment_status=status,
            contract_status=contract_status,
            address=address,
            emergency_contact=emergency_contact
        )
        
        # Check if email already exists
        if check_email_exists(db, employee_data.email):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Email address already exists"
            )
        
        if employee_data.biometric_id and check_biometric_id_exists(db, employee_data.biometric_id):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Biometric ID already exists"
            )
        
        # Get department and validate
        department = db.query(Department).filter(
            Department.id == department_id,
            Department.is_active == True
        ).first()
        if not department:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST, 
                detail="Invalid department selected"
            )
        
        # Generate employee ID
        employee_id = generate_employee_id(department.code, db)
        
        # Handle photo upload
        if photo and photo.filename:
            photo_path = save_uploaded_photo(photo)
            uploaded_files.append(photo_path)
        
        # Handle form data
        form_data = await request.form()
        
        # Process education data
        education_entries = process_education_data(form_data)
        validate_education_data(education_entries)
        
        # Handle custom fields
        custom_fields = db.query(CustomField).filter(CustomField.is_active == True).all()
        custom_field_data, custom_uploaded_files = process_custom_fields(form_data, custom_fields)
        uploaded_files.extend(custom_uploaded_files)
        
        # Helper function to safely parse date
        def parse_date(date_str):
            if date_str and date_str.strip():
                try:
                    return datetime.strptime(date_str.strip(), '%Y-%m-%d').date()
                except ValueError:
                    return None
            return None
        
        # Helper function to safely get enum value
        def get_enum_value(enum_class, value):
            if value and value.strip():
                try:
                    return enum_class(value.strip())
                except ValueError:
                    return None
            return None
        
        # Create new employee with all fields
        new_employee = Employee(
            employee_id=employee_id,
            # Basic Information
            first_name=employee_data.first_name,
            last_name=employee_data.last_name,
            email=employee_data.email,
            phone=employee_data.phone,
            biometric_id=employee_data.biometric_id,
            photo=photo_path,
            department_id=department_id,
            position=employee_data.position,
            status=StatusEnum(status),
            contract_status=ContractStatusEnum(contract_status),
            shift_id=shift_id if shift_id else None,
            
            # Personal Details
            date_of_birth=parse_date(date_of_birth),
            gender=get_enum_value(GenderEnum, gender),
            nationality=nationality.strip() if nationality else None,
            national_id_number=national_id_number.strip() if national_id_number else None,
            passport_number=passport_number.strip() if passport_number else None,
            marital_status=get_enum_value(MaritalStatusEnum, marital_status),
            religion=religion.strip() if religion else None,
            blood_group=get_enum_value(BloodGroupEnum, blood_group),
            
            # Tax & Employment Details
            tin_number=tin_number.strip() if tin_number else None,
            nssf_number=nssf_number.strip() if nssf_number else None,
            start_of_employment=parse_date(start_of_employment),
            end_of_employment=parse_date(end_of_employment),
            employment_type=get_enum_value(EmploymentTypeEnum, employment_type),
            
            # Bank Details
            bank_name=bank_name.strip() if bank_name else None,
            branch_name=branch_name.strip() if branch_name else None,
            account_title=account_title.strip() if account_title else None,
            account_number=account_number.strip() if account_number else None,
            
            # Additional Information
            address=employee_data.address,
            emergency_contact=employee_data.emergency_contact,
            
            # Custom fields and education
            custom_fields=json.dumps(custom_field_data) if custom_field_data else None,
            education=json.dumps(education_entries) if education_entries else None
        )
        
        db.add(new_employee)
        db.commit()
        db.refresh(new_employee)
        
        logger.info(f"Employee {new_employee.employee_id} created by user {current_user.username}")
        
        return RedirectResponse(
            url=f"/staff/view?success=Employee {new_employee.first_name} {new_employee.last_name} added successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except ValueError as e:
        # Validation error - clean up uploaded files
        cleanup_files_on_error(uploaded_files)
        return await _handle_add_employee_error(request, current_user, db, str(e), locals())
        
    except HTTPException as e:
        cleanup_files_on_error(uploaded_files)
        return await _handle_add_employee_error(request, current_user, db, e.detail, locals())
        
    except Exception as e:
        db.rollback()
        cleanup_files_on_error(uploaded_files)
        logger.error(f"Error creating employee: {str(e)}")
        return await _handle_add_employee_error(request, current_user, db, "Error creating employee. Please try again.", locals())

async def _handle_add_employee_error(request, current_user, db, error_msg, form_vars):
    """Helper function to handle add employee errors"""
    try:
        # Get form data for re-display
        form_data = await request.form() if 'form_data' not in form_vars else form_vars.get('form_data')
        education_entries = process_education_data(form_data) if form_data else []
        
        custom_fields = db.query(CustomField).filter(CustomField.is_active == True).all()
        departments = get_cached_departments(db)
        shifts = db.query(Shift).filter(Shift.is_active == True).order_by(Shift.name).all()
        
        # Create form_data dictionary for template
        form_data_dict = {
            "first_name": form_vars.get('first_name', ''),
            "last_name": form_vars.get('last_name', ''),
            "email": form_vars.get('email', ''),
            "phone": form_vars.get('phone', ''),
            "biometric_id": form_vars.get('biometric_id', ''),
            "department_id": form_vars.get('department_id'),
            "position": form_vars.get('position', ''),
            "status": form_vars.get('status', ''),
            "contract_status": form_vars.get('contract_status', ''),
            "shift_id": form_vars.get('shift_id'),
            "date_of_birth": form_vars.get('date_of_birth'),
            "gender": form_vars.get('gender'),
            "nationality": form_vars.get('nationality'),
            "national_id_number": form_vars.get('national_id_number', ''),
            "passport_number": form_vars.get('passport_number', ''),
            "marital_status": form_vars.get('marital_status'),
            "religion": form_vars.get('religion'),
            "blood_group": form_vars.get('blood_group'),
            "tin_number": form_vars.get('tin_number', ''),
            "nssf_number": form_vars.get('nssf_number', ''),
            "start_of_employment": form_vars.get('start_of_employment'),
            "end_of_employment": form_vars.get('end_of_employment'),
            "employment_type": form_vars.get('employment_type'),
            "bank_name": form_vars.get('bank_name'),
            "branch_name": form_vars.get('branch_name', ''),
            "account_title": form_vars.get('account_title', ''),
            "account_number": form_vars.get('account_number', ''),
            "address": form_vars.get('address', ''),
            "emergency_contact": form_vars.get('emergency_contact', '')
        }
        
        # Add custom field values to form_data_dict
        if custom_fields and form_data:
            for field in custom_fields:
                field_key = f"custom_{field.field_name}"
                if field_key in form_data:
                    form_data_dict[field_key] = form_data[field_key]
        
        context = {
            "request": request,
            "user": current_user,
            "custom_fields": custom_fields,
            "departments": departments,
            "shifts": shifts,
            "status_options": [emp_status.value for emp_status in StatusEnum],
            "contract_status_options": [cont_status.value for cont_status in ContractStatusEnum],
            "education_data": education_entries,
            "error": error_msg,
            "form_data": form_data_dict,
            "page_title": "Add Employee"
        }
        return templates.TemplateResponse("staff/add_employee.html", context)
    except Exception as e:
        logger.error(f"Error handling add employee error: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading form"
        )

@router.get("/view/{employee_id}", response_class=HTMLResponse)
async def view_employee_details(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """View detailed employee information"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Parse custom field values and education data safely
        custom_field_values = safely_parse_custom_fields(employee.custom_fields)
        education_data = safely_parse_education_data(employee.education)
        
        # Calculate age if date of birth is available
        employee_age = None
        if employee.date_of_birth:
            try:
                today = date.today()
                birthday_this_year = employee.date_of_birth.replace(year=today.year)
                
                employee_age = today.year - employee.date_of_birth.year
                
                # Adjust if birthday hasn't occurred this year
                if today < birthday_this_year:
                    employee_age -= 1
            except Exception as e:
                logger.warning(f"Error calculating age for employee {employee_id}: {str(e)}")
                employee_age = None
        
        # Format dates for display
        formatted_dates = {}
        
        def safe_format_date(date_obj, field_name="unknown"):
            if not date_obj:
                return None
            
            try:
                if hasattr(date_obj, 'strftime'):
                    return date_obj.strftime('%d/%m/%Y')
                elif isinstance(date_obj, str):
                    try:
                        parsed_date = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
                        return parsed_date.strftime('%d/%m/%Y')
                    except ValueError:
                        try:
                            parsed_date = datetime.strptime(date_obj, '%Y-%m-%d')
                            return parsed_date.strftime('%d/%m/%Y')
                        except ValueError:
                            logger.warning(f"Could not parse date string '{date_obj}' for field {field_name}")
                            return str(date_obj)
                else:
                    logger.warning(f"Unknown date type {type(date_obj)} for field {field_name}: {date_obj}")
                    return str(date_obj) if date_obj else None
            except Exception as e:
                logger.error(f"Error formatting date for field {field_name}: {str(e)}")
                return str(date_obj) if date_obj else None
        
        # Format all date fields safely
        formatted_dates['date_of_birth'] = safe_format_date(employee.date_of_birth, 'date_of_birth')
        formatted_dates['start_of_employment'] = safe_format_date(employee.start_of_employment, 'start_of_employment')
        formatted_dates['end_of_employment'] = safe_format_date(employee.end_of_employment, 'end_of_employment')
        formatted_dates['created_at'] = safe_format_date(employee.created_at, 'created_at')
        formatted_dates['updated_at'] = safe_format_date(employee.updated_at, 'updated_at')
        
        # Get custom field definitions
        custom_fields = db.query(CustomField).filter(CustomField.is_active == True).all()
        
        context = {
            "request": request,
            "user": current_user,
            "employee": employee,
            "custom_fields": custom_fields,
            "custom_field_values": custom_field_values,
            "education_data": education_data,
            "formatted_dates": formatted_dates,
            "employee_age": employee_age,
            "page_title": f"{employee.first_name} {employee.last_name} - Details"
        }
        return templates.TemplateResponse("staff/view_employee.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error viewing employee details: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading employee details"
        )

@router.get("/edit/{employee_id}", response_class=HTMLResponse)
async def edit_employee_page(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Display edit employee form"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        custom_fields = db.query(CustomField).filter(CustomField.is_active == True).all()
        departments = get_cached_departments(db)
        shifts = db.query(Shift).order_by(Shift.is_active.desc(), Shift.name).all()

        # Parse custom field values and education data safely
        custom_field_values = safely_parse_custom_fields(employee.custom_fields)
        education_data = safely_parse_education_data(employee.education)
        
        context = {
            "request": request,
            "user": current_user,
            "employee": employee,
            "custom_fields": custom_fields,
            "custom_field_values": custom_field_values,
            "education_data": education_data,
            "departments": departments,
            "shifts": shifts,
            "status_options": [emp_status.value for emp_status in StatusEnum],
            "contract_status_options": [cont_status.value for cont_status in ContractStatusEnum],
            "page_title": "Edit Employee"
        }
        return templates.TemplateResponse("staff/edit_employee.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading edit employee page: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading form"
        )

@router.post("/edit/{employee_id}")
async def edit_employee(
    employee_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    # Basic Information
    first_name: str = Form(..., min_length=2, max_length=50),
    last_name: str = Form(..., min_length=2, max_length=50),
    email: str = Form(...),
    phone: str = Form(""),
    biometric_id: str = Form(""),
    department_id: int = Form(...),
    position: str = Form(..., min_length=2),
    status: str = Form(...),
    contract_status: str = Form(...),
    shift_id: Optional[int] = Form(None),
    photo: UploadFile = File(None),
    
    # Personal Details
    date_of_birth: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    nationality: Optional[str] = Form(None),
    national_id_number: Optional[str] = Form(""),
    passport_number: Optional[str] = Form(""),
    marital_status: Optional[str] = Form(None),
    religion: Optional[str] = Form(None),
    blood_group: Optional[str] = Form(None),
    
    # Tax & Employment Details
    tin_number: Optional[str] = Form(""),
    nssf_number: Optional[str] = Form(""),
    start_of_employment: Optional[str] = Form(None),
    end_of_employment: Optional[str] = Form(None),
    employment_type: Optional[str] = Form(None),
    
    # Bank Details
    bank_name: Optional[str] = Form(None),
    branch_name: Optional[str] = Form(""),
    account_title: Optional[str] = Form(""),
    account_number: Optional[str] = Form(""),
    
    # Additional Information
    address: str = Form(""),
    emergency_contact: str = Form(""),
    
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an existing employee"""
    uploaded_files = []
    
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Validate employee data - only basic fields for validation
        employee_data = EmployeeUpdate(
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            biometric_id=biometric_id,
            department_id=department_id,
            position=position,
            employment_status=status,
            contract_status=contract_status,
            address=address,
            emergency_contact=emergency_contact
        )
        
        # Check if email conflicts with other employees
        if check_email_exists(db, employee_data.email, employee_id):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Email address already exists"
            )
        
        if employee_data.biometric_id and check_biometric_id_exists(db, employee_data.biometric_id, employee_id):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Biometric ID already exists"
            )
        
        # Validate department
        department = db.query(Department).filter(
            Department.id == department_id,
            Department.is_active == True
        ).first()
        if not department:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST, 
                detail="Invalid department"
            )
        
        old_photo_path = employee.photo
        
        # Handle photo upload
        if photo and photo.filename:
            if old_photo_path:
                background_tasks.add_task(delete_file, old_photo_path)
            employee.photo = save_uploaded_photo(photo)
            uploaded_files.append(employee.photo)
        
        # Handle form data
        form_data = await request.form()
        
        # Process education data
        education_entries = process_education_data(form_data)
        validate_education_data(education_entries)
        
        # Handle custom fields
        custom_fields = db.query(CustomField).filter(CustomField.is_active == True).all()
        
        # Parse existing custom data
        existing_custom_data = safely_parse_custom_fields(employee.custom_fields)
        
        custom_field_data, custom_uploaded_files = process_custom_fields(
            form_data, custom_fields, existing_custom_data
        )
        uploaded_files.extend(custom_uploaded_files)
        
        # Helper function to safely parse date
        def parse_date(date_str):
            if date_str and date_str.strip():
                try:
                    return datetime.strptime(date_str.strip(), '%Y-%m-%d').date()
                except ValueError:
                    return None
            return None
        
        # Helper function to safely get enum value
        def get_enum_value(enum_class, value):
            if value and value.strip():
                try:
                    return enum_class(value.strip())
                except ValueError:
                    return None
            return None
        
        # Update basic employee information
        employee.first_name = employee_data.first_name
        employee.last_name = employee_data.last_name
        employee.email = employee_data.email
        employee.phone = employee_data.phone
        employee.biometric_id = employee_data.biometric_id
        employee.department_id = department_id
        employee.position = employee_data.position
        employee.status = StatusEnum(status)
        employee.contract_status = ContractStatusEnum(contract_status)
        employee.address = employee_data.address
        employee.emergency_contact = employee_data.emergency_contact
        employee.shift_id = shift_id if shift_id else None
        
        # Update personal details
        employee.date_of_birth = parse_date(date_of_birth)
        employee.gender = get_enum_value(GenderEnum, gender)
        employee.nationality = nationality.strip() if nationality else None
        employee.national_id_number = national_id_number.strip() if national_id_number else None
        employee.passport_number = passport_number.strip() if passport_number else None
        employee.marital_status = get_enum_value(MaritalStatusEnum, marital_status)
        employee.religion = religion.strip() if religion else None
        employee.blood_group = get_enum_value(BloodGroupEnum, blood_group)
        
        # Update tax & employment details
        employee.tin_number = tin_number.strip() if tin_number else None
        employee.nssf_number = nssf_number.strip() if nssf_number else None
        employee.start_of_employment = parse_date(start_of_employment)
        employee.end_of_employment = parse_date(end_of_employment)
        employee.employment_type = get_enum_value(EmploymentTypeEnum, employment_type)
        
        # Update bank details
        employee.bank_name = bank_name.strip() if bank_name else None
        employee.branch_name = branch_name.strip() if branch_name else None
        employee.account_title = account_title.strip() if account_title else None
        employee.account_number = account_number.strip() if account_number else None
        
        # Update custom fields and education
        employee.custom_fields = json.dumps(custom_field_data) if custom_field_data else None
        employee.education = json.dumps(education_entries) if education_entries else None
        employee.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Employee {employee.employee_id} updated by user {current_user.username}")
        
        return RedirectResponse(
            url=f"/staff/view?success=Employee {employee.first_name} {employee.last_name} updated successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except (ValueError, HTTPException) as e:
        # Handle validation and HTTP errors
        cleanup_files_on_error(uploaded_files)
        return await _handle_edit_employee_error(request, current_user, db, employee, str(e) if isinstance(e, ValueError) else e.detail)
        
    except Exception as e:
        db.rollback()
        cleanup_files_on_error(uploaded_files)
        logger.error(f"Error updating employee: {str(e)}")
        return await _handle_edit_employee_error(request, current_user, db, employee, "Error updating employee. Please try again.")

async def _handle_edit_employee_error(request, current_user, db, employee, error_msg):
    """Helper function to handle edit employee errors"""
    try:
        # Get form data for re-display
        form_data = await request.form()
        education_entries = process_education_data(form_data)
        
        # Parse existing custom data for re-display
        existing_custom_data = safely_parse_custom_fields(employee.custom_fields)
        
        custom_fields = db.query(CustomField).filter(CustomField.is_active == True).all()
        departments = get_cached_departments(db)
        shifts = db.query(Shift).order_by(Shift.is_active.desc(), Shift.name).all()
        
        context = {
            "request": request,
            "user": current_user,
            "employee": employee,
            "custom_fields": custom_fields,
            "custom_field_values": existing_custom_data,
            "education_data": education_entries,
            "departments": departments,
            "shifts": shifts,
            "status_options": [emp_status.value for emp_status in StatusEnum],
            "contract_status_options": [cont_status.value for cont_status in ContractStatusEnum],
            "error": error_msg,
            "page_title": "Edit Employee"
        }
        return templates.TemplateResponse("staff/edit_employee.html", context)
    except Exception as e:
        logger.error(f"Error handling edit employee error: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading form"
        )

@router.post("/delete/{employee_id}")
async def delete_employee(
    employee_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete an employee (soft delete by setting inactive)"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Instead of hard delete, set status to inactive
        employee.status = StatusEnum.NOT_ACTIVE
        employee.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Employee {employee.employee_id} deactivated by user {current_user.username}")
        
        return {
            "success": True, 
            "message": f"Employee {employee.first_name} {employee.last_name} deactivated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deactivating employee: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deactivating employee"
        )

# Document Management Routes
@router.get("/employee/{employee_id}/download/{field_name}")
async def download_employee_document(
    employee_id: int,
    field_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download employee document by field name"""
    try:
        # Get employee
        employee = get_employee_by_id(db, employee_id)
        
        # Parse custom fields to get document path
        custom_field_values = safely_parse_custom_fields(employee.custom_fields)
        
        # Get document path
        document_path = custom_field_values.get(field_name)
        if not document_path:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Handle both single document and array of documents
        if isinstance(document_path, list):
            document_path = document_path[0] if document_path else None
            if not document_path:
                raise HTTPException(
                    status_code=http_status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
        
        # Validate filename (prevent directory traversal)
        filename = document_path.split('/')[-1]
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )
        
        # Construct file path
        file_path = Path("static") / document_path
        
        # Additional security check - ensure file is within uploads directory
        uploads_dir = Path("static/uploads").resolve()
        try:
            file_path.resolve().relative_to(uploads_dir)
        except ValueError:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading employee document {employee_id}/{field_name}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error downloading file"
        )

@router.get("/employee/{employee_id}/preview/{field_name}")
async def preview_employee_document(
    employee_id: int,
    field_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Preview employee document by field name"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Parse custom fields to get document path
        custom_field_values = safely_parse_custom_fields(employee.custom_fields)
        
        # Get document path
        document_path = custom_field_values.get(field_name)
        if not document_path:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Handle array of documents - preview first one
        if isinstance(document_path, list):
            document_path = document_path[0] if document_path else None
            if not document_path:
                raise HTTPException(
                    status_code=http_status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
        
        # Construct file path
        file_path = Path("static") / document_path
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Return file for inline preview
        filename = document_path.split('/')[-1]
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error previewing employee document {employee_id}/{field_name}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error previewing file"
        )

@router.delete("/employee/{employee_id}/delete-document/{field_name}")
async def delete_employee_document(
    employee_id: int,
    field_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete employee document by field name"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Parse custom fields to get document path
        custom_field_values = safely_parse_custom_fields(employee.custom_fields)
        
        # Get document path
        document_path = custom_field_values.get(field_name)
        if not document_path:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Check if it's a single document or array of documents
        if isinstance(document_path, str):
            # Single document - delete the file
            delete_file(document_path)
            # Remove from custom fields
            del custom_field_values[field_name]
        elif isinstance(document_path, list):
            # Multiple documents - delete all files
            for doc_path in document_path:
                delete_file(doc_path)
            # Remove from custom fields
            del custom_field_values[field_name]
        
        # Update employee custom fields
        employee.custom_fields = json.dumps(custom_field_values) if custom_field_values else None
        employee.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Document(s) deleted for employee {employee_id}, field {field_name} by user {current_user.username}")
        
        return {"success": True, "message": "Document(s) deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting employee document {employee_id}/{field_name}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting document"
        )

# File Management Routes
@router.get("/download/{file_type}/{filename}")
async def download_file(
    file_type: str,
    filename: str,
    current_user: User = Depends(get_current_user)
):
    """Download uploaded files (photos or documents) with security checks"""
    try:
        # Validate file type
        if file_type not in ['photos', 'documents']:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND, 
                detail="File not found"
            )
        
        # Validate filename (prevent directory traversal)
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )
        
        # Construct file path
        file_path = Path("static/uploads") / file_type / filename
        
        # Additional security check - ensure file is within uploads directory
        uploads_dir = Path("static/uploads").resolve()
        try:
            file_path.resolve().relative_to(uploads_dir)
        except ValueError:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND, 
                detail="File not found"
            )
        
        # Determine media type
        media_type = "application/octet-stream"
        if file_type == "photos":
            media_type = "image/jpeg"
        elif file_type == "documents" and filename.endswith('.pdf'):
            media_type = "application/pdf"
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {file_type}/{filename}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error downloading file"
        )

# ID Card Generation Routes
@router.get("/generate-id-card/{employee_id}")
async def generate_id_card(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate ID card PDF using HTML template"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Generate PDF using HTML template
        pdf_buffer = await create_id_card_from_html(employee, request)
        
        # Log ID card generation
        logger.info(f"ID card generated for employee {employee.employee_id} by user {current_user.username}")
        
        # Return PDF response
        filename = f"ID_Card_{employee.first_name}_{employee.last_name}_{employee.employee_id}.pdf"
        
        return StreamingResponse(
            io.BytesIO(pdf_buffer),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating ID card for employee {employee_id}: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating ID card"
        )

@router.get("/debug-id-card/{employee_id}", response_class=HTMLResponse)
async def debug_id_card(
    employee_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug route to see HTML template before PDF conversion"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Calculate dates
        issue_date = datetime.now()
        expiry_date = issue_date + timedelta(days=365)
        
        # Generate QR code
        qr_code_base64 = ""
        try:
            department_name = employee.department.name if employee.department else "N/A"
            qr_data = f"IUEA-{employee.employee_id}|{employee.first_name} {employee.last_name}|{department_name}|{employee.position or 'Staff'}"
            qr_buffer = generate_qr_code_buffer(qr_data)
            if qr_buffer:
                qr_code_base64 = base64.b64encode(qr_buffer).decode()
        except Exception as e:
            logger.warning(f"Could not generate QR code: {e}")
        
        context = {
            "request": request,
            "employee": employee,
            "issue_date": issue_date.strftime('%d/%m/%Y'),
            "expiry_date": expiry_date.strftime('%d/%m/%Y'),
            "qr_code_base64": qr_code_base64,
        }
        
        return templates.TemplateResponse("id.html", context)
        
    except Exception as e:
        logger.error(f"Error in debug ID card: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Attendance Redirect
@router.get("/employee/{employee_id}/attendance", response_class=HTMLResponse)
async def employee_attendance_redirect(employee_id: int):
    """Redirect to biometric attendance page"""
    return RedirectResponse(url=f"/biometric/employee/{employee_id}/attendance")

# Custom Fields Management
@router.get("/custom-fields", response_class=HTMLResponse)
async def manage_custom_fields(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manage custom fields"""
    try:
        custom_fields = db.query(CustomField).order_by(CustomField.created_at.desc()).all()
        
        context = {
            "request": request,
            "user": current_user,
            "custom_fields": custom_fields,
            "page_title": "Manage Custom Fields"
        }
        return templates.TemplateResponse("staff/custom_fields.html", context)
        
    except Exception as e:
        logger.error(f"Error loading custom fields: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading custom fields"
        )

@router.post("/custom-fields/add")
async def add_custom_field(
    field_name: str = Form(..., min_length=2, max_length=50),
    field_label: str = Form(..., min_length=2, max_length=100),
    field_type: str = Form(...),
    field_options: str = Form(""),
    is_required: bool = Form(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a new custom field"""
    try:
        # Validate field_name (should be snake_case)
        import re
        if not re.match(r'^[a-z][a-z0-9_]*$', field_name):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Field name must start with a letter and contain only lowercase letters, numbers, and underscores"
            )
        
        # Check if field name already exists
        existing = db.query(CustomField).filter(CustomField.field_name == field_name).first()
        if existing:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST, 
                detail="Field name already exists"
            )
        
        # Validate field type
        valid_types = ['text', 'number', 'date', 'select', 'boolean', 'document']
        if field_type not in valid_types:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST, 
                detail="Invalid field type"
            )
        
        new_field = CustomField(
            field_name=field_name,
            field_label=field_label.strip(),
            field_type=field_type,
            field_options=field_options.strip() if field_options else "",
            is_required=is_required
        )
        
        db.add(new_field)
        db.commit()
        
        logger.info(f"Custom field '{field_name}' created by user {current_user.username}")
        
        return RedirectResponse(
            url="/staff/custom-fields?success=Custom field added successfully",
            status_code=http_status.HTTP_302_FOUND
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating custom field: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating custom field"
        )

@router.post("/custom-fields/toggle/{field_id}")
async def toggle_custom_field(
    field_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Toggle custom field active status"""
    try:
        field = db.query(CustomField).filter(CustomField.id == field_id).first()
        if not field:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND, 
                detail="Custom field not found"
            )
        
        field.is_active = not field.is_active
        db.commit()
        
        status_text = "activated" if field.is_active else "deactivated"
        logger.info(f"Custom field '{field.field_name}' {status_text} by user {current_user.username}")
        
        return {
            "success": True, 
            "message": f"Custom field {status_text} successfully", 
            "is_active": field.is_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error toggling custom field: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating custom field"
        )

@router.delete("/custom-fields/{field_id}")
async def delete_custom_field(
    field_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a custom field"""
    try:
        field = db.query(CustomField).filter(CustomField.id == field_id).first()
        if not field:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND, 
                detail="Custom field not found"
            )
        
        field_name = field.field_name
        db.delete(field)
        db.commit()
        
        logger.info(f"Custom field '{field_name}' deleted by user {current_user.username}")
        
        return {"success": True, "message": "Custom field deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting custom field: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting custom field"
        )

# Enhanced API Endpoints
@router.get("/api/search-suggestions")
async def get_search_suggestions(
    q: str = Query(..., min_length=2, max_length=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get search suggestions for employee search"""
    try:
        search_term = f"%{q.strip()}%"
        
        # Get suggestions from different fields
        suggestions = []
        
        # Employee names
        name_results = db.query(Employee.first_name, Employee.last_name)\
            .filter(
                or_(
                    Employee.first_name.ilike(search_term),
                    Employee.last_name.ilike(search_term)
                )
            ).limit(5).all()
        
        for first_name, last_name in name_results:
            full_name = f"{first_name} {last_name}"
            if full_name not in suggestions:
                suggestions.append(full_name)
        
        # Employee IDs
        id_results = db.query(Employee.employee_id)\
            .filter(Employee.employee_id.ilike(search_term))\
            .limit(3).all()
        
        for (employee_id,) in id_results:
            if employee_id not in suggestions:
                suggestions.append(employee_id)
        
        # Positions
        position_results = db.query(Employee.position.distinct())\
            .filter(Employee.position.ilike(search_term))\
            .limit(3).all()
        
        for (position,) in position_results:
            if position and position not in suggestions:
                suggestions.append(position)
        
        return {"suggestions": suggestions[:8]}  # Limit to 8 suggestions
        
    except Exception as e:
        logger.error(f"Error getting search suggestions: {str(e)}")
        return {"suggestions": []}

@router.get("/api/employees", response_model=List[EmployeeResponse])
async def get_employees_api(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    search: Optional[str] = Query(None),
    department_id: str = Query("", description="Filter by department"),
    status_filter: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    sort_by: Optional[str] = Query("created_at")
):
    """Optimized API endpoint to get employees with pagination"""
    try:
        department_id = int(department_id) if department_id and department_id.strip() else None
        query = db.query(Employee).options(selectinload(Employee.department))
        
        # Apply filters (same logic as view_employees)
        if search and search.strip():
            search_term = f"%{search.strip()}%"
            query = query.filter(
                or_(
                    Employee.first_name.ilike(search_term),
                    Employee.last_name.ilike(search_term),
                    Employee.email.ilike(search_term),
                    Employee.employee_id.ilike(search_term),
                    Employee.position.ilike(search_term),
                    Employee.biometric_id.ilike(search_term)
                )
            )
        
        if department_id:
            query = query.filter(Employee.department_id == department_id)
        
        if status_filter:
            try:
                status_enum = StatusEnum(status_filter)
                query = query.filter(Employee.status == status_enum)
            except ValueError:
                pass
        
        # Apply sorting
        if sort_by == "name":
            query = query.order_by(Employee.first_name.asc(), Employee.last_name.asc())
        elif sort_by == "employee_id":
            query = query.order_by(Employee.employee_id.asc())
        elif sort_by == "department":
            query = query.join(Department, Employee.department_id == Department.id, isouter=True)\
                        .order_by(Department.name.asc())
        else:
            query = query.order_by(Employee.created_at.desc())
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination
        offset = (page - 1) * per_page
        employees = query.offset(offset).limit(per_page).all()
        
        # Calculate pagination metadata
        total_pages = (total_count + per_page - 1) // per_page
        
        # Format response
        employee_data = []
        for emp in employees:
            employee_data.append(EmployeeResponse(
                id=emp.id,
                employee_id=emp.employee_id,
                first_name=emp.first_name,
                last_name=emp.last_name,
                email=emp.email,
                phone=emp.phone,
                photo=emp.photo,
                biometric_id=emp.biometric_id,
                department_id=emp.department_id,
                position=emp.position,
                status=emp.status.value,
                contract_status=emp.contract_status.value,
                address=emp.address,
                emergency_contact=emp.emergency_contact,
                created_at=emp.created_at,
                updated_at=emp.updated_at
            ))
        
        return employee_data
        
    except Exception as e:
        logger.error(f"Error in employees API: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching employees"
        )

@router.get("/api/stats")
async def get_staff_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive staff statistics with performance optimization"""
    try:
        # Use more efficient queries with proper aggregation
        stats_query = db.query(
            func.count(Employee.id).label('total_employees'),
            func.count(func.case([(Employee.status == StatusEnum.ACTIVE, 1)])).label('active_employees'),
            func.count(func.case([(Employee.status == StatusEnum.NOT_ACTIVE, 1)])).label('inactive_employees')
        ).first()
        
        total_employees = stats_query.total_employees
        active_employees = stats_query.active_employees
        inactive_employees = stats_query.inactive_employees
        
        # Department count
        departments_count = db.query(Department).filter(Department.is_active == True).count()
        
        # Department breakdown
        department_breakdown = db.query(
            Department.name,
            func.count(Employee.id).label('employee_count')
        ).join(Employee, Department.id == Employee.department_id, isouter=True)\
         .filter(Department.is_active == True)\
         .group_by(Department.id, Department.name)\
         .all()
        
        # Recent additions (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_additions = db.query(Employee)\
            .filter(Employee.created_at >= thirty_days_ago)\
            .count()
        
        # Biometric coverage
        biometric_coverage = db.query(Employee)\
            .filter(Employee.biometric_id.isnot(None))\
            .filter(Employee.biometric_id != '')\
            .count()
        
        biometric_percentage = (biometric_coverage / total_employees * 100) if total_employees > 0 else 0
        
        return {
            "total_employees": total_employees,
            "active_employees": active_employees,
            "inactive_employees": inactive_employees,
            "departments_count": departments_count,
            "recent_additions": recent_additions,
            "biometric_coverage": {
                "count": biometric_coverage,
                "percentage": round(biometric_percentage, 1)
            },
            "department_breakdown": [
                {"name": name, "count": count} 
                for name, count in department_breakdown
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting staff stats: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching statistics"
        )

@router.post("/api/bulk-update-status")
async def bulk_update_status(
    employee_ids: List[int] = Form(...),
    new_status: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Bulk update employee status for performance"""
    try:
        # Validate status
        try:
            status_enum = StatusEnum(new_status)
        except ValueError:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Invalid status value"
            )
        
        # Validate employee IDs exist
        employees = db.query(Employee).filter(Employee.id.in_(employee_ids)).all()
        if len(employees) != len(employee_ids):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="One or more employee IDs not found"
            )
        
        # Perform bulk update
        update_count = db.query(Employee)\
            .filter(Employee.id.in_(employee_ids))\
            .update({
                Employee.status: status_enum,
                Employee.updated_at: datetime.utcnow()
            }, synchronize_session=False)
        
        db.commit()
        
        logger.info(f"Bulk status update: {update_count} employees updated to {new_status} by {current_user.username}")
        
        return {
            "success": True,
            "message": f"Successfully updated {update_count} employees",
            "updated_count": update_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error in bulk status update: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating employee status"
        )

@router.get("/export")
async def export_employees(
    format: str = Query("csv", regex="^(csv|xlsx)$"),
    search: Optional[str] = Query(None),
    department_id: Optional[int] = Query(None),
    status_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Export employees with current filters applied"""
    try:
        # Use same query logic as view for consistency
        query = db.query(Employee).options(selectinload(Employee.department))
        
        # Apply same filters as the view
        if search and search.strip():
            search_term = f"%{search.strip()}%"
            query = query.filter(
                or_(
                    Employee.first_name.ilike(search_term),
                    Employee.last_name.ilike(search_term),
                    Employee.email.ilike(search_term),
                    Employee.employee_id.ilike(search_term),
                    Employee.position.ilike(search_term)
                )
            )
        
        if department_id:
            query = query.filter(Employee.department_id == department_id)
        
        if status_filter:
            try:
                status_enum = StatusEnum(status_filter)
                query = query.filter(Employee.status == status_enum)
            except ValueError:
                pass
        
        employees = query.order_by(Employee.created_at.desc()).all()
        
        if format == "csv":
            # Generate CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Headers
            writer.writerow([
                'Employee ID', 'First Name', 'Last Name', 'Email', 'Phone',
                'Department', 'Position', 'Status', 'Biometric ID', 'Address',
                'Emergency Contact', 'Created Date'
            ])
            
            # Data
            for emp in employees:
                writer.writerow([
                    emp.employee_id,
                    emp.first_name,
                    emp.last_name,
                    emp.email,
                    emp.phone or '',
                    emp.department.name if emp.department else '',
                    emp.position,
                    emp.status.value,
                    emp.biometric_id or '',
                    emp.address or '',
                    emp.emergency_contact or '',
                    emp.created_at.strftime('%Y-%m-%d %H:%M:%S')
                ])
            
            output.seek(0)
            
            # Return CSV response
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"employees_export_{timestamp}.csv"
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode('utf-8')),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        # Add XLSX export if needed
        else:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="XLSX export not yet implemented"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting employees: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error exporting data"
        )

# Additional document routes for array handling
@router.get("/employee/{employee_id}/download/{field_name}/{doc_index}")
async def download_specific_document(
    employee_id: int,
    field_name: str,
    doc_index: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download specific document from array by index"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Parse custom fields
        custom_field_values = safely_parse_custom_fields(employee.custom_fields)
        
        # Get document array
        document_array = custom_field_values.get(field_name, [])
        
        # Handle single document
        if isinstance(document_array, str):
            if doc_index == 0:
                document_path = document_array
            else:
                raise HTTPException(
                    status_code=http_status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
        elif isinstance(document_array, list):
            if doc_index >= len(document_array):
                raise HTTPException(
                    status_code=http_status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
            document_path = document_array[doc_index]
        else:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Construct file path and validate
        filename = document_path.split('/')[-1]
        file_path = Path("static") / document_path
        
        if not file_path.exists():
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading specific document: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error downloading file"
        )

@router.delete("/employee/{employee_id}/delete-document/{field_name}/{doc_index}")
async def delete_single_document_from_array(
    employee_id: int,
    field_name: str,
    doc_index: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a single document from an array of documents"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        # Parse custom fields
        custom_field_values = safely_parse_custom_fields(employee.custom_fields)
        
        # Get document array
        document_array = custom_field_values.get(field_name, [])
        if not isinstance(document_array, list) or doc_index >= len(document_array):
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Delete the specific document file
        document_to_delete = document_array[doc_index]
        delete_file(document_to_delete)
        
        # Remove from array
        document_array.pop(doc_index)
        
        # Update custom fields
        if document_array:
            custom_field_values[field_name] = document_array
        else:
            # Remove field if no documents left
            del custom_field_values[field_name]
        
        employee.custom_fields = json.dumps(custom_field_values) if custom_field_values else None
        employee.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Single document deleted for employee {employee_id}, field {field_name}, index {doc_index} by user {current_user.username}")
        
        return {"success": True, "message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting single document: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting document"
        )

# Testing and Debug Routes (remove in production)
@router.get("/test-photo/{employee_id}")
async def test_photo_paths(
    employee_id: int, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Test different photo path combinations - DEBUG ONLY"""
    try:
        employee = get_employee_by_id(db, employee_id)
        
        debug_info = {
            "employee_id": employee.employee_id,
            "photo_field": employee.photo,
            "photo_exists": bool(employee.photo and Path("static") / employee.photo),
            "debug_complete": "Check logs for detailed debug info"
        }
        
        # Comprehensive photo debugging
        if employee.photo:
            photo_path = Path("static") / employee.photo
            debug_info.update({
                "photo_path": str(photo_path),
                "file_exists": photo_path.exists(),
                "is_file": photo_path.is_file() if photo_path.exists() else False,
                "file_size": photo_path.stat().st_size if photo_path.exists() else 0
            })
        
        logger.info(f"Photo debug for employee {employee.employee_id}: {debug_info}")
        
        return debug_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in photo test: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error testing photo paths"
        )

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint with system information"""
    try:
        # Check upload directories
        photo_dir_exists = PHOTO_UPLOAD_DIR.exists()
        doc_dir_exists = DOCUMENT_UPLOAD_DIR.exists()
        
        # Check cache status
        cache_status = {
            "departments_cached": bool(_department_cache),
            "cache_timestamp": _department_cache_timestamp.isoformat() if _department_cache_timestamp else None,
            "cache_age_seconds": (datetime.utcnow() - _department_cache_timestamp).seconds if _department_cache_timestamp else None
        }
        
        return {
            "status": "healthy", 
            "service": "staff_management",
            "timestamp": datetime.utcnow().isoformat(),
            "upload_directories": {
                "photos": photo_dir_exists,
                "documents": doc_dir_exists
            },
            "cache": cache_status,
            "version": "2.0.0",
            "features": [
                "server_side_pagination",
                "optimized_queries",
                "department_caching",
                "bulk_operations",
                "export_functionality",
                "performance_improvements"
            ]
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Performance monitoring endpoint
@router.get("/api/performance-metrics")
async def get_performance_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get performance metrics for monitoring"""
    try:
        # Database query performance
        start_time = datetime.utcnow()
        
        # Sample query to test DB performance
        sample_count = db.query(Employee).count()
        
        db_query_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Cache performance
        cache_hit = bool(_department_cache and _department_cache_timestamp)
        cache_age = (datetime.utcnow() - _department_cache_timestamp).seconds if _department_cache_timestamp else None
        
        # File system checks
        photo_dir_size = sum(f.stat().st_size for f in PHOTO_UPLOAD_DIR.rglob('*') if f.is_file()) if PHOTO_UPLOAD_DIR.exists() else 0
        doc_dir_size = sum(f.stat().st_size for f in DOCUMENT_UPLOAD_DIR.rglob('*') if f.is_file()) if DOCUMENT_UPLOAD_DIR.exists() else 0
        
        return {
            "database": {
                "query_time_seconds": db_query_time,
                "sample_record_count": sample_count,
                "connection_status": "healthy"
            },
            "cache": {
                "hit": cache_hit,
                "age_seconds": cache_age,
                "status": "active" if cache_hit else "empty"
            },
            "storage": {
                "photos_size_mb": round(photo_dir_size / (1024 * 1024), 2),
                "documents_size_mb": round(doc_dir_size / (1024 * 1024), 2),
                "total_size_mb": round((photo_dir_size + doc_dir_size) / (1024 * 1024), 2)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching performance metrics"
        )

# Initialization and cleanup
async def initialize_staff_module():
    """Initialize staff module with all required setup"""
    try:
        # Ensure upload directories exist
        ensure_upload_directories()
        
        # Initialize database indexes (if needed)
        # Note: This should typically be done via migrations
        # create_database_indexes(db)
        
        logger.info("Staff management module initialized successfully")
        logger.info(f"Upload directories: Photos={PHOTO_UPLOAD_DIR}, Documents={DOCUMENT_UPLOAD_DIR}")
        logger.info(f"Allowed image formats: {ALLOWED_IMAGE_EXTENSIONS}")
        logger.info(f"Allowed document formats: {ALLOWED_DOCUMENT_EXTENSIONS}")
        logger.info(f"Max image size: {MAX_IMAGE_SIZE // (1024*1024)}MB")
        logger.info(f"Max document size: {MAX_DOCUMENT_SIZE // (1024*1024)}MB")
        
    except Exception as e:
        logger.error(f"Failed to initialize staff management module: {str(e)}")
        raise

# Initialize on module import
try:
    ensure_upload_directories()
    logger.info("Staff management router initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize staff management router: {str(e)}")

# Router configuration summary
logger.info("Staff Management Router Configuration:")
logger.info("=" * 50)
logger.info("Routes configured:")
logger.info("  GET  /staff/")
logger.info("  GET  /staff/view")
logger.info("  GET  /staff/add")
logger.info("  POST /staff/add")
logger.info("  GET  /staff/view/{employee_id}")
logger.info("  GET  /staff/edit/{employee_id}")
logger.info("  POST /staff/edit/{employee_id}")
logger.info("  POST /staff/delete/{employee_id}")
logger.info("  GET  /staff/generate-id-card/{employee_id}")
logger.info("  GET  /staff/debug-id-card/{employee_id}")
logger.info("  GET  /staff/download/{file_type}/{filename}")
logger.info("  GET  /staff/employee/{employee_id}/download/{field_name}")
logger.info("  GET  /staff/employee/{employee_id}/preview/{field_name}")
logger.info("  DELETE /staff/employee/{employee_id}/delete-document/{field_name}")
logger.info("  GET  /staff/employee/{employee_id}/attendance")
logger.info("  GET  /staff/custom-fields")
logger.info("  POST /staff/custom-fields/add")
logger.info("  POST /staff/custom-fields/toggle/{field_id}")
logger.info("  DELETE /staff/custom-fields/{field_id}")
logger.info("  GET  /staff/api/employees")
logger.info("  GET  /staff/api/search-suggestions")
logger.info("  GET  /staff/api/stats")
logger.info("  POST /staff/api/bulk-update-status")
logger.info("  GET  /staff/export")
logger.info("  GET  /staff/health")
logger.info("  GET  /staff/api/performance-metrics")
logger.info("=" * 50)
logger.info("Features enabled:")
logger.info("  ✓ Server-side pagination and filtering")
logger.info("  ✓ Optimized database queries with eager loading")
logger.info("  ✓ Department caching for performance")
logger.info("  ✓ Comprehensive file upload validation")
logger.info("  ✓ Photo resizing and optimization")
logger.info("  ✓ ID card generation with QR codes")
logger.info("  ✓ Custom fields support")
logger.info("  ✓ Education data management")
logger.info("  ✓ Document management and preview")
logger.info("  ✓ Search suggestions and autocomplete")
logger.info("  ✓ Bulk operations for efficiency")
logger.info("  ✓ Data export functionality")
logger.info("  ✓ Performance monitoring")
logger.info("  ✓ Comprehensive error handling")
logger.info("  ✓ Security validations")
logger.info("  ✓ Production-ready logging")
logger.info("=" * 50)