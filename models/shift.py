"""
Shift Management Models
Database models for employee shift management
"""

from sqlalchemy import Column, Integer, String, DateTime, Time, Boolean, Text, Enum, event
from sqlalchemy.orm import relationship, Session
from models.database import Base
from datetime import datetime
import enum
import logging

logger = logging.getLogger(__name__)

class ShiftTypeEnum(enum.Enum):
    STANDARD = "standard"
    DYNAMIC = "dynamic"

class Shift(Base):
    """Employee shift schedules"""
    __tablename__ = "shifts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Store lowercase values to match existing database
    shift_type = Column(
        Enum(ShiftTypeEnum, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=ShiftTypeEnum.STANDARD
    )
    
    # Standard shift fields
    weekday_start = Column(Time, nullable=True)
    weekday_end = Column(Time, nullable=True)
    
    # Weekend day fields
    saturday_start = Column(Time, nullable=True)
    saturday_end = Column(Time, nullable=True)
    sunday_start = Column(Time, nullable=True)
    sunday_end = Column(Time, nullable=True)
    
    # Dynamic shift fields
    monday_start = Column(Time, nullable=True)
    monday_end = Column(Time, nullable=True)
    tuesday_start = Column(Time, nullable=True)
    tuesday_end = Column(Time, nullable=True)
    wednesday_start = Column(Time, nullable=True)
    wednesday_end = Column(Time, nullable=True)
    thursday_start = Column(Time, nullable=True)
    thursday_end = Column(Time, nullable=True)
    friday_start = Column(Time, nullable=True)
    friday_end = Column(Time, nullable=True)
    
    is_active = Column(Boolean, default=True, nullable=False)
    assigned_employees_count = Column(Integer, default=0, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    employees = relationship("Employee", foreign_keys="Employee.shift_id", back_populates="shift")
    
    def update_employee_count(self, db_session: Session):
        """Update the assigned employees count"""
        try:
            from models.employee import Employee
            count = db_session.query(Employee).filter(Employee.shift_id == self.id).count()
            self.assigned_employees_count = count
            return count
        except Exception as e:
            logger.error(f"Error updating employee count for shift {self.id}: {e}")
            return self.assigned_employees_count

# Event listeners to automatically update employee count when employees change
@event.listens_for(Session, 'after_flush')
def update_shift_counts_after_flush(session, flush_context):
    """Update shift employee counts after database flush"""
    try:
        from models.employee import Employee
        
        # Get all shifts that need count updates
        shifts_to_update = set()
        
        # Check new, dirty, and deleted objects
        for obj in session.new | session.dirty | session.deleted:
            if isinstance(obj, Employee) and hasattr(obj, 'shift_id'):
                if obj.shift_id:
                    shifts_to_update.add(obj.shift_id)
                
                # Also check if shift_id was changed
                if hasattr(obj, '__dict__') and obj in session.dirty:
                    state = session.identity_map._mutable_attrs.get(obj)
                    if state and 'shift_id' in state:
                        old_shift_id = state['shift_id'][0] if state['shift_id'][0] != obj.shift_id else None
                        if old_shift_id:
                            shifts_to_update.add(old_shift_id)
        
        # Update counts for affected shifts
        for shift_id in shifts_to_update:
            count = session.query(Employee).filter(Employee.shift_id == shift_id).count()
            session.execute(
                Shift.__table__.update()
                .where(Shift.__table__.c.id == shift_id)
                .values(assigned_employees_count=count)
            )
            
    except Exception as e:
        logger.error(f"Error updating shift counts: {e}")