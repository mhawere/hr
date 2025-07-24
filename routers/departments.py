"""
Department Management Router
Production-ready implementation with comprehensive CRUD operations,
validation, error handling, and security features.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Request, Form, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy import and_, or_, func, desc, asc
from pydantic import BaseModel, validator
import re

from models.database import get_db
from models.employee import Department, Employee, User
from utils.auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Pydantic models for validation
class DepartmentCreate(BaseModel):
    name: str
    code: str
    description: Optional[str] = ""
    manager_id: Optional[int] = None
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Department name must be at least 2 characters long')
        if len(v.strip()) > 100:
            raise ValueError('Department name must be less than 100 characters')
        return v.strip()
    
    @validator('code')
    def validate_code(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Department code must be at least 2 characters long')
        if len(v.strip()) > 10:
            raise ValueError('Department code must be less than 10 characters')
        if not re.match(r'^[A-Z0-9]+$', v.upper().strip()):
            raise ValueError('Department code must contain only letters and numbers')
        return v.upper().strip()
    
    @validator('description')
    def validate_description(cls, v):
        if v and len(v.strip()) > 500:
            raise ValueError('Description must be less than 500 characters')
        return v.strip() if v else ""

class DepartmentUpdate(DepartmentCreate):
    is_active: bool = True

class DepartmentResponse(BaseModel):
    id: int
    name: str
    code: str
    description: Optional[str]
    manager_id: Optional[int]
    is_active: bool
    employee_count: int = 0
    
    class Config:
        from_attributes = True

# Helper functions
def get_department_by_id(db: Session, dept_id: int) -> Department:
    """Get department by ID with error handling."""
    department = db.query(Department).filter(Department.id == dept_id).first()
    if not department:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Department with ID {dept_id} not found"
        )
    return department

def check_department_exists(db: Session, name: str, code: str, exclude_id: Optional[int] = None) -> bool:
    """Check if department with given name or code already exists."""
    query = db.query(Department).filter(
        or_(Department.name.ilike(name.strip()), Department.code == code.upper().strip())
    )
    if exclude_id:
        query = query.filter(Department.id != exclude_id)
    return query.first() is not None

def get_employee_count_for_department(db: Session, dept_id: int) -> int:
    """Get employee count for a department."""
    return db.query(Employee).filter(Employee.department_id == dept_id).count()

def can_delete_department(db: Session, dept_id: int) -> tuple[bool, str]:
    """Check if department can be deleted."""
    employee_count = get_employee_count_for_department(db, dept_id)
    if employee_count > 0:
        return False, f"Cannot delete department with {employee_count} employees. Please reassign employees first."
    return True, ""

def get_departments_with_stats(
    db: Session, 
    search: Optional[str] = None,
    active_only: bool = False,
    sort_by: str = "name",
    sort_order: str = "asc",
    offset: int = 0,
    limit: int = 100
) -> tuple[List[Department], int]:
    """Get departments with employee statistics and filtering."""
    
    # Base query
    query = db.query(Department)
    
    # Apply filters
    if search:
        search_term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Department.name.ilike(search_term),
                Department.code.ilike(search_term),
                Department.description.ilike(search_term)
            )
        )
    
    if active_only:
        query = query.filter(Department.is_active == True)
    
    # Get total count for pagination
    total_count = query.count()
    
    # Apply sorting
    sort_column = getattr(Department, sort_by, Department.name)
    if sort_order.lower() == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(asc(sort_column))
    
    # Apply pagination
    departments = query.offset(offset).limit(limit).all()
    
    return departments, total_count

# Routes
@router.get("/", response_class=HTMLResponse)
async def list_departments(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    search: Optional[str] = Query(None, description="Search departments"),
    active_only: bool = Query(False, description="Show only active departments"),
    sort_by: str = Query("name", description="Sort by field"),
    sort_order: str = Query("asc", description="Sort order"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page")
):
    """List all departments with filtering, sorting, and pagination."""
    try:
        offset = (page - 1) * per_page
        departments, total_count = get_departments_with_stats(
            db, search, active_only, sort_by, sort_order, offset, per_page
        )
        
        # Get employee counts for each department
        dept_data = []
        for dept in departments:
            employee_count = get_employee_count_for_department(db, dept.id)
            dept_data.append({
                'department': dept,
                'employee_count': employee_count
            })
        
        # Calculate pagination info
        total_pages = (total_count + per_page - 1) // per_page
        
        context = {
            "request": request,
            "user": current_user,
            "departments": dept_data,
            "total_count": total_count,
            "current_page": page,
            "total_pages": total_pages,
            "per_page": per_page,
            "search": search or "",
            "active_only": active_only,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "page_title": "Manage Departments"
        }
        return templates.TemplateResponse("departments/list.html", context)
        
    except Exception as e:
        logger.error(f"Error listing departments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching departments"
        )

@router.get("/add", response_class=HTMLResponse)
async def add_department_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Display add department form."""
    try:
        # Get all active employees for manager selection
        employees = db.query(Employee).filter(
            Employee.status == "ACTIVE"
        ).order_by(Employee.first_name, Employee.last_name).all()
        
        context = {
            "request": request,
            "user": current_user,
            "employees": employees,
            "page_title": "Add Department"
        }
        return templates.TemplateResponse("departments/add.html", context)
        
    except Exception as e:
        logger.error(f"Error loading add department page: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while loading the form"
        )

@router.post("/add")
async def add_department(
    request: Request,
    name: str = Form(..., min_length=2, max_length=100),
    code: str = Form(..., min_length=2, max_length=10),
    description: str = Form("", max_length=500),
    manager_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new department."""
    try:
        # Validate input using Pydantic
        dept_data = DepartmentCreate(
            name=name,
            code=code,
            description=description,
            manager_id=manager_id if manager_id else None
        )
        
        # Check if department already exists
        if check_department_exists(db, dept_data.name, dept_data.code):
            employees = db.query(Employee).filter(
                Employee.status == "ACTIVE"
            ).order_by(Employee.first_name, Employee.last_name).all()
            
            context = {
                "request": request,
                "user": current_user,
                "employees": employees,
                "error": "Department name or code already exists",
                "form_data": dept_data.dict(),
                "page_title": "Add Department"
            }
            return templates.TemplateResponse("departments/add.html", context)
        
        # Validate manager exists if provided
        if dept_data.manager_id:
            manager = db.query(Employee).filter(
                Employee.id == dept_data.manager_id,
                Employee.status == "ACTIVE"
            ).first()
            if not manager:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Selected manager not found or inactive"
                )
        
        # Create new department
        new_department = Department(
            name=dept_data.name,
            code=dept_data.code,
            description=dept_data.description,
            manager_id=dept_data.manager_id
        )
        
        db.add(new_department)
        db.commit()
        db.refresh(new_department)
        
        logger.info(f"Department '{new_department.name}' created by user {current_user.username}")
        
        return RedirectResponse(
            url="/departments/?success=Department created successfully",
            status_code=status.HTTP_302_FOUND
        )
        
    except ValueError as e:
        # Validation error
        employees = db.query(Employee).filter(
            Employee.status == "ACTIVE"
        ).order_by(Employee.first_name, Employee.last_name).all()
        
        context = {
            "request": request,
            "user": current_user,
            "employees": employees,
            "error": str(e),
            "form_data": {
                "name": name,
                "code": code,
                "description": description,
                "manager_id": manager_id
            },
            "page_title": "Add Department"
        }
        return templates.TemplateResponse("departments/add.html", context)
        
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Database integrity error creating department: {str(e)}")
        
        employees = db.query(Employee).filter(
            Employee.status == "ACTIVE"
        ).order_by(Employee.first_name, Employee.last_name).all()
        
        context = {
            "request": request,
            "user": current_user,
            "employees": employees,
            "error": "Department name or code already exists",
            "form_data": {
                "name": name,
                "code": code,
                "description": description,
                "manager_id": manager_id
            },
            "page_title": "Add Department"
        }
        return templates.TemplateResponse("departments/add.html", context)
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating department: {str(e)}")
        
        employees = db.query(Employee).filter(
            Employee.status == "ACTIVE"
        ).order_by(Employee.first_name, Employee.last_name).all()
        
        context = {
            "request": request,
            "user": current_user,
            "employees": employees,
            "error": "An unexpected error occurred. Please try again.",
            "form_data": {
                "name": name,
                "code": code,
                "description": description,
                "manager_id": manager_id
            },
            "page_title": "Add Department"
        }
        return templates.TemplateResponse("departments/add.html", context)

@router.get("/edit/{dept_id}", response_class=HTMLResponse)
async def edit_department_page(
    dept_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Display edit department form."""
    try:
        department = get_department_by_id(db, dept_id)
        
        # Get all active employees for manager selection
        employees = db.query(Employee).filter(
            Employee.status == "ACTIVE"
        ).order_by(Employee.first_name, Employee.last_name).all()
        
        # Get employee count for this department
        employee_count = get_employee_count_for_department(db, dept_id)
        
        context = {
            "request": request,
            "user": current_user,
            "department": department,
            "employees": employees,
            "employee_count": employee_count,
            "page_title": "Edit Department"
        }
        return templates.TemplateResponse("departments/edit.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading edit department page: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while loading the form"
        )

@router.post("/edit/{dept_id}")
async def edit_department(
    dept_id: int,
    request: Request,
    name: str = Form(..., min_length=2, max_length=100),
    code: str = Form(..., min_length=2, max_length=10),
    description: str = Form("", max_length=500),
    manager_id: Optional[int] = Form(None),
    is_active: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an existing department."""
    try:
        department = get_department_by_id(db, dept_id)
        
        # Validate input using Pydantic
        dept_data = DepartmentUpdate(
            name=name,
            code=code,
            description=description,
            manager_id=manager_id if manager_id else None,
            is_active=is_active
        )
        
        # Check if name or code conflicts with other departments
        if check_department_exists(db, dept_data.name, dept_data.code, dept_id):
            employees = db.query(Employee).filter(
                Employee.status == "ACTIVE"
            ).order_by(Employee.first_name, Employee.last_name).all()
            
            employee_count = get_employee_count_for_department(db, dept_id)
            
            context = {
                "request": request,
                "user": current_user,
                "department": department,
                "employees": employees,
                "employee_count": employee_count,
                "error": "Department name or code already exists",
                "page_title": "Edit Department"
            }
            return templates.TemplateResponse("departments/edit.html", context)
        
        # Validate manager exists if provided
        if dept_data.manager_id:
            manager = db.query(Employee).filter(
                Employee.id == dept_data.manager_id,
                Employee.status == "ACTIVE"
            ).first()
            if not manager:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Selected manager not found or inactive"
                )
        
        # Check if deactivating department with employees
        if not dept_data.is_active and department.is_active:
            employee_count = get_employee_count_for_department(db, dept_id)
            if employee_count > 0:
                employees = db.query(Employee).filter(
                    Employee.status == "ACTIVE"
                ).order_by(Employee.first_name, Employee.last_name).all()
                
                context = {
                    "request": request,
                    "user": current_user,
                    "department": department,
                    "employees": employees,
                    "employee_count": employee_count,
                    "error": f"Cannot deactivate department with {employee_count} employees. Please reassign employees first.",
                    "page_title": "Edit Department"
                }
                return templates.TemplateResponse("departments/edit.html", context)
        
        # Update department
        department.name = dept_data.name
        department.code = dept_data.code
        department.description = dept_data.description
        department.manager_id = dept_data.manager_id
        department.is_active = dept_data.is_active
        
        db.commit()
        
        logger.info(f"Department '{department.name}' updated by user {current_user.username}")
        
        return RedirectResponse(
            url="/departments/?success=Department updated successfully",
            status_code=status.HTTP_302_FOUND
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        # Validation error
        employees = db.query(Employee).filter(
            Employee.status == "ACTIVE"
        ).order_by(Employee.first_name, Employee.last_name).all()
        
        employee_count = get_employee_count_for_department(db, dept_id)
        
        context = {
            "request": request,
            "user": current_user,
            "department": department,
            "employees": employees,
            "employee_count": employee_count,
            "error": str(e),
            "page_title": "Edit Department"
        }
        return templates.TemplateResponse("departments/edit.html", context)
        
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Database integrity error updating department: {str(e)}")
        
        employees = db.query(Employee).filter(
            Employee.status == "ACTIVE"
        ).order_by(Employee.first_name, Employee.last_name).all()
        
        employee_count = get_employee_count_for_department(db, dept_id)
        
        context = {
            "request": request,
            "user": current_user,
            "department": department,
            "employees": employees,
            "employee_count": employee_count,
            "error": "Department name or code already exists",
            "page_title": "Edit Department"
        }
        return templates.TemplateResponse("departments/edit.html", context)
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating department: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating the department"
        )

@router.post("/delete/{dept_id}")
async def delete_department(
    dept_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a department."""
    try:
        department = get_department_by_id(db, dept_id)
        
        # Check if department can be deleted
        can_delete, error_message = can_delete_department(db, dept_id)
        if not can_delete:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": error_message}
            )
        
        dept_name = department.name
        db.delete(department)
        db.commit()
        
        logger.info(f"Department '{dept_name}' deleted by user {current_user.username}")
        
        return JSONResponse(
            content={"success": True, "message": f"Department '{dept_name}' deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting department: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": "An error occurred while deleting the department"}
        )

@router.post("/toggle/{dept_id}")
async def toggle_department_status(
    dept_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Toggle department active status."""
    try:
        department = get_department_by_id(db, dept_id)
        
        # Check if deactivating department with employees
        if department.is_active:
            employee_count = get_employee_count_for_department(db, dept_id)
            if employee_count > 0:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "message": f"Cannot deactivate department with {employee_count} employees. Please reassign employees first."
                    }
                )
        
        department.is_active = not department.is_active
        db.commit()
        
        status_text = "activated" if department.is_active else "deactivated"
        logger.info(f"Department '{department.name}' {status_text} by user {current_user.username}")
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"Department '{department.name}' {status_text} successfully",
                "is_active": department.is_active
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error toggling department status: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": "An error occurred while updating the department status"}
        )

@router.get("/api/", response_model=List[DepartmentResponse])
async def get_departments_api(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    search: Optional[str] = Query(None),
    active_only: bool = Query(False),
    sort_by: str = Query("name"),
    sort_order: str = Query("asc"),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """API endpoint to get departments with filtering and pagination."""
    try:
        departments, total_count = get_departments_with_stats(
            db, search, active_only, sort_by, sort_order, offset, limit
        )
        
        # Build response with employee counts
        response_data = []
        for dept in departments:
            employee_count = get_employee_count_for_department(db, dept.id)
            dept_response = DepartmentResponse(
                id=dept.id,
                name=dept.name,
                code=dept.code,
                description=dept.description,
                manager_id=dept.manager_id,
                is_active=dept.is_active,
                employee_count=employee_count
            )
            response_data.append(dept_response)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in departments API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching departments"
        )

@router.get("/{dept_id}/employees")
async def get_department_employees(
    dept_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get employees in a specific department."""
    try:
        department = get_department_by_id(db, dept_id)
        
        employees = db.query(Employee).filter(
            Employee.department_id == dept_id
        ).order_by(Employee.first_name, Employee.last_name).all()
        
        context = {
            "request": request,
            "user": current_user,
            "department": department,
            "employees": employees,
            "page_title": f"{department.name} - Employees"
        }
        return templates.TemplateResponse("departments/employees.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting department employees: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching department employees"
        )

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "departments"}