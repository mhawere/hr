from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from .database import Base
from datetime import datetime

class CustomField(Base):
    __tablename__ = "custom_fields"
    
    id = Column(Integer, primary_key=True, index=True)
    field_name = Column(String(100), nullable=False)
    field_label = Column(String(100), nullable=False)
    field_type = Column(String(50), nullable=False)  # text, number, date, select, boolean
    field_options = Column(Text)  # For select fields, JSON string of options
    is_required = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
