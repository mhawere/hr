# services/attendance_badge_service.py
"""
Attendance Badge Automation Service
Automatically awards attendance-based badges based on actual attendance data
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc

from models.employee import Employee
from models.attendance import ProcessedAttendance
from models.performance import Badge, EmployeeBadge, BadgeCategory, BadgeLevel
from services.attendance_service import AttendanceService

logger = logging.getLogger(__name__)

class AttendanceBadgeService:
    def __init__(self, db: Session):
        self.db = db
        self.attendance_service = AttendanceService(db)
    
    async def process_all_attendance_badges(self, start_date: Optional[date] = None) -> Dict[str, int]:
        """
        Process attendance badges for all employees
        Returns summary of badges awarded
        """
        if not start_date:
            start_date = date(2024, 1, 10)  # Default start date
        
        logger.info(f"🏆 Starting attendance badge processing from {start_date}")
        
        # Get all employees with biometric IDs
        employees = self.db.query(Employee).filter(
            Employee.biometric_id.isnot(None),
            Employee.biometric_id != ""
        ).all()
        
        badge_summary = {
            'employees_processed': 0,
            'perfect_month_awarded': 0,
            'punctuality_champion_awarded': 0,
            'attendance_superstar_awarded': 0,
            'total_badges_awarded': 0
        }
        
        for employee in employees:
            try:
                employee_badges = await self.process_employee_attendance_badges(
                    employee.id, start_date
                )
                
                badge_summary['employees_processed'] += 1
                badge_summary['perfect_month_awarded'] += employee_badges.get('perfect_month', 0)
                badge_summary['punctuality_champion_awarded'] += employee_badges.get('punctuality_champion', 0)
                badge_summary['attendance_superstar_awarded'] += employee_badges.get('attendance_superstar', 0)
                
            except Exception as e:
                logger.error(f"Error processing badges for employee {employee.id}: {str(e)}")
                continue
        
        badge_summary['total_badges_awarded'] = (
            badge_summary['perfect_month_awarded'] + 
            badge_summary['punctuality_champion_awarded'] + 
            badge_summary['attendance_superstar_awarded']
        )
        
        logger.info(f"✅ Badge processing complete: {badge_summary}")
        return badge_summary
    
    async def process_employee_attendance_badges(self, employee_id: int, start_date: date) -> Dict[str, int]:
        """
        Process attendance badges for a specific employee
        """
        employee_badges = {
            'perfect_month': 0,
            'punctuality_champion': 0,
            'attendance_superstar': 0
        }
        
        try:
            # Check Perfect Month badges
            perfect_months = await self.check_perfect_month_badges(employee_id, start_date)
            employee_badges['perfect_month'] = perfect_months
            
            # Check Punctuality Champion badges
            punctuality_badges = await self.check_punctuality_champion_badges(employee_id, start_date)
            employee_badges['punctuality_champion'] = punctuality_badges
            
            # Check Attendance Superstar badges
            superstar_badges = await self.check_attendance_superstar_badges(employee_id, start_date)
            employee_badges['attendance_superstar'] = superstar_badges
            
            return employee_badges
            
        except Exception as e:
            logger.error(f"Error processing badges for employee {employee_id}: {str(e)}")
            return employee_badges
    
    async def check_perfect_month_badges(self, employee_id: int, start_date: date) -> int:
        """
        Check and award Perfect Month badges (100% attendance for one month)
        """
        badges_awarded = 0
        
        # Get Perfect Month badge
        perfect_month_badge = self.db.query(Badge).filter(
            and_(
                Badge.name == "Perfect Month",
                Badge.is_active == True
            )
        ).first()
        
        if not perfect_month_badge:
            logger.warning("Perfect Month badge not found")
            return 0
        
        # Check each month from start_date to current month
        current_date = date.today()
        check_date = date(start_date.year, start_date.month, 1)
        
        while check_date < current_date:
            # Get last day of the month
            if check_date.month == 12:
                last_day = date(check_date.year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = date(check_date.year, check_date.month + 1, 1) - timedelta(days=1)
            
            # Don't check future months or incomplete current month
            if last_day >= current_date:
                break
            
            # Check if badge already awarded for this month
            badge_key = f"perfect_month_{check_date.strftime('%Y_%m')}"
            existing_badge = self.db.query(EmployeeBadge).filter(
                and_(
                    EmployeeBadge.employee_id == employee_id,
                    EmployeeBadge.badge_id == perfect_month_badge.id,
                    EmployeeBadge.notes.contains(badge_key)
                )
            ).first()
            
            if existing_badge:
                check_date = self._next_month(check_date)
                continue
            
            # Check attendance for this month
            if await self.has_perfect_attendance_for_month(employee_id, check_date, last_day):
                # Award the badge
                employee_badge = EmployeeBadge(
                    employee_id=employee_id,
                    badge_id=perfect_month_badge.id,
                    notes=f"Perfect attendance for {check_date.strftime('%B %Y')} - {badge_key}",
                    earned_date=last_day  # Award on last day of perfect month
                )
                
                self.db.add(employee_badge)
                badges_awarded += 1
                
                logger.info(f"🏆 Perfect Month badge awarded to employee {employee_id} for {check_date.strftime('%B %Y')}")
            
            check_date = self._next_month(check_date)
        
        if badges_awarded > 0:
            self.db.commit()
        
        return badges_awarded
    
    async def check_punctuality_champion_badges(self, employee_id: int, start_date: date) -> int:
        """
        Check and award Punctuality Champion badges (No late arrivals for 30 days)
        """
        badges_awarded = 0
        
        # Get Punctuality Champion badge
        punctuality_badge = self.db.query(Badge).filter(
            and_(
                Badge.name == "Punctuality Champion",
                Badge.is_active == True
            )
        ).first()
        
        if not punctuality_badge:
            logger.warning("Punctuality Champion badge not found")
            return 0
        
        # Check for 30-day periods of perfect punctuality
        current_date = date.today()
        check_date = start_date
        
        while check_date <= current_date - timedelta(days=30):
            end_date = check_date + timedelta(days=29)  # 30-day period
            
            # Check if badge already awarded for this period
            badge_key = f"punctuality_{check_date.strftime('%Y_%m_%d')}"
            existing_badge = self.db.query(EmployeeBadge).filter(
                and_(
                    EmployeeBadge.employee_id == employee_id,
                    EmployeeBadge.badge_id == punctuality_badge.id,
                    EmployeeBadge.notes.contains(badge_key)
                )
            ).first()
            
            if existing_badge:
                check_date += timedelta(days=1)
                continue
            
            # Check punctuality for 30-day period
            if await self.has_perfect_punctuality_for_period(employee_id, check_date, end_date):
                # Award the badge
                employee_badge = EmployeeBadge(
                    employee_id=employee_id,
                    badge_id=punctuality_badge.id,
                    notes=f"Perfect punctuality from {check_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} - {badge_key}",
                    earned_date=end_date
                )
                
                self.db.add(employee_badge)
                badges_awarded += 1
                
                logger.info(f"🏆 Punctuality Champion badge awarded to employee {employee_id} for period {check_date} to {end_date}")
                
                # Skip ahead to avoid overlapping periods
                check_date = end_date + timedelta(days=1)
            else:
                check_date += timedelta(days=1)
        
        if badges_awarded > 0:
            self.db.commit()
        
        return badges_awarded
    
    async def check_attendance_superstar_badges(self, employee_id: int, start_date: date) -> int:
        """
        Check and award Attendance Superstar badges (Perfect attendance for 3 months)
        """
        badges_awarded = 0
        
        # Get Attendance Superstar badge
        superstar_badge = self.db.query(Badge).filter(
            and_(
                Badge.name == "Attendance Superstar",
                Badge.is_active == True
            )
        ).first()
        
        if not superstar_badge:
            logger.warning("Attendance Superstar badge not found")
            return 0
        
        # Check for 3-month periods of perfect attendance
        current_date = date.today()
        check_date = date(start_date.year, start_date.month, 1)
        
        while check_date <= current_date:
            # Calculate 3-month period
            end_month_date = self._add_months(check_date, 3)
            last_day_of_period = date(end_month_date.year, end_month_date.month, 1) - timedelta(days=1)
            
            # Don't check incomplete periods
            if last_day_of_period >= current_date:
                break
            
            # Check if badge already awarded for this period
            badge_key = f"superstar_{check_date.strftime('%Y_%m')}"
            existing_badge = self.db.query(EmployeeBadge).filter(
                and_(
                    EmployeeBadge.employee_id == employee_id,
                    EmployeeBadge.badge_id == superstar_badge.id,
                    EmployeeBadge.notes.contains(badge_key)
                )
            ).first()
            
            if existing_badge:
                check_date = self._next_month(check_date)
                continue
            
            # Check perfect attendance for 3-month period
            if await self.has_perfect_attendance_for_period(employee_id, check_date, last_day_of_period):
                # Award the badge
                employee_badge = EmployeeBadge(
                    employee_id=employee_id,
                    badge_id=superstar_badge.id,
                    notes=f"Perfect attendance for 3 months from {check_date.strftime('%B %Y')} - {badge_key}",
                    earned_date=last_day_of_period
                )
                
                self.db.add(employee_badge)
                badges_awarded += 1
                
                logger.info(f"🏆 Attendance Superstar badge awarded to employee {employee_id} for 3-month period starting {check_date.strftime('%B %Y')}")
                
                # Skip ahead 3 months to avoid overlapping
                check_date = self._add_months(check_date, 3)
            else:
                check_date = self._next_month(check_date)
        
        if badges_awarded > 0:
            self.db.commit()
        
        return badges_awarded
    
    async def has_perfect_attendance_for_month(self, employee_id: int, start_date: date, end_date: date) -> bool:
        """
        Check if employee has perfect attendance for a specific month
        """
        try:
            # Get all attendance records for the month
            attendance_records = self.db.query(ProcessedAttendance).filter(
                and_(
                    ProcessedAttendance.employee_id == employee_id,
                    ProcessedAttendance.date >= start_date,
                    ProcessedAttendance.date <= end_date
                )
            ).all()
            
            if not attendance_records:
                return False
            
            # Get working days in the month (exclude weekends)
            working_days = self._get_working_days_in_period(start_date, end_date)
            
            if len(attendance_records) < working_days:
                return False
            
            # Check if all records show present
            for record in attendance_records:
                if not record.is_present:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking perfect attendance: {str(e)}")
            return False
    
    async def has_perfect_punctuality_for_period(self, employee_id: int, start_date: date, end_date: date) -> bool:
        """
        Check if employee has perfect punctuality (no late arrivals) for a period
        """
        try:
            # Get all attendance records for the period where employee was present
            late_records = self.db.query(ProcessedAttendance).filter(
                and_(
                    ProcessedAttendance.employee_id == employee_id,
                    ProcessedAttendance.date >= start_date,
                    ProcessedAttendance.date <= end_date,
                    ProcessedAttendance.is_present == True,
                    ProcessedAttendance.is_late == True
                )
            ).count()
            
            return late_records == 0
            
        except Exception as e:
            logger.error(f"Error checking punctuality: {str(e)}")
            return False
    
    async def has_perfect_attendance_for_period(self, employee_id: int, start_date: date, end_date: date) -> bool:
        """
        Check if employee has perfect attendance for a period (multiple months)
        """
        try:
            # Check each month in the period
            current_month = date(start_date.year, start_date.month, 1)
            
            while current_month <= end_date:
                # Get last day of current month
                if current_month.month == 12:
                    last_day = date(current_month.year + 1, 1, 1) - timedelta(days=1)
                else:
                    last_day = date(current_month.year, current_month.month + 1, 1) - timedelta(days=1)
                
                # Adjust for the actual period
                month_start = max(current_month, start_date)
                month_end = min(last_day, end_date)
                
                if not await self.has_perfect_attendance_for_month(employee_id, month_start, month_end):
                    return False
                
                current_month = self._next_month(current_month)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking period attendance: {str(e)}")
            return False
    
    def _get_working_days_in_period(self, start_date: date, end_date: date) -> int:
        """
        Calculate number of working days (Monday-Friday) in a period
        """
        working_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            # Monday = 0, Sunday = 6
            if current_date.weekday() < 5:  # Monday to Friday
                working_days += 1
            current_date += timedelta(days=1)
        
        return working_days
    
    def _next_month(self, current_date: date) -> date:
        """Get first day of next month"""
        if current_date.month == 12:
            return date(current_date.year + 1, 1, 1)
        else:
            return date(current_date.year, current_date.month + 1, 1)
    
    def _add_months(self, current_date: date, months: int) -> date:
        """Add months to a date"""
        month = current_date.month - 1 + months
        year = current_date.year + month // 12
        month = month % 12 + 1
        return date(year, month, 1)