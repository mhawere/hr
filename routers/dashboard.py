from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func

from models.database import get_db
from models.employee import Employee, User, Department
from models.custom_fields import CustomField
from utils.auth import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get dashboard statistics
    total_employees = db.query(Employee).count()
    active_employees = db.query(Employee).filter(Employee.status == "ACTIVE").count()
    departments_count = db.query(Department).filter(Department.is_active == True).count()
    
    # New hires this month
    from datetime import datetime, timedelta
    thirty_days_ago = datetime.now() - timedelta(days=30)
    new_hires = db.query(Employee).filter(Employee.created_at >= thirty_days_ago).count()
    
    # Recent employees (last 5)
    recent_employees = db.query(Employee).order_by(Employee.created_at.desc()).limit(5).all()
    
    # Department distribution
    dept_stats = db.query(
        Department.name.label('department_name'), 
        func.count(Employee.id).label('count')
    ).join(Employee, Department.id == Employee.department_id, isouter=True).group_by(Department.name).all()
    
    # Contracts expiring soon (next 30 days)
    contracts_expiring = db.query(Employee).filter(Employee.contract_status == "EXPIRED").count()
    
    context = {
        "request": request,
        "user": current_user,
        "total_employees": total_employees,
        "active_employees": active_employees,
        "departments_count": departments_count,
        "new_hires": new_hires,
        "contracts_expiring": contracts_expiring,
        "recent_employees": recent_employees,
        "dept_stats": dept_stats,
        "page_title": "Dashboard"
    }
    
    return templates.TemplateResponse("dashboard.html", context)
