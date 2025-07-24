"""
🔮 ATTENDANCE PREDICTION & MACHINE LEARNING MODULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Advanced ML-powered attendance prediction and pattern analysis
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import joblib
from dataclasses import dataclass

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer

# Database imports
from sqlalchemy.orm import Session
from sqlalchemy import func, case, and_, or_, desc
from models.employee import Employee, Department
from models.attendance import ProcessedAttendance

# Setup logging
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs("models/ml_models", exist_ok=True)

@dataclass
class PredictionResult:
    """Standard prediction result format"""
    employee_id: str
    name: str
    department: str
    position: str
    probability: float
    risk_level: str
    factors: List[str]
    confidence: str

class AttendancePredictionEngine:
    """🔮 Advanced Attendance Prediction Engine with ML Models"""
    
    def __init__(self):
        self.late_prediction_model = None
        self.early_departure_model = None
        self.absence_prediction_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.models_trained = False
        self.model_version = "1.0"
        
        # Try to load existing models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models if they exist"""
        try:
            model_path = "models/ml_models"
            
            if os.path.exists(f"{model_path}/late_prediction_model.joblib"):
                self.late_prediction_model = joblib.load(f"{model_path}/late_prediction_model.joblib")
                self.scaler = joblib.load(f"{model_path}/scaler.joblib")
                
                with open(f"{model_path}/feature_columns.txt", 'r') as f:
                    self.feature_columns = f.read().strip().split('\n')
                
                self.models_trained = True
                logger.info("✅ Pre-trained ML models loaded successfully")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {str(e)}")
            self.models_trained = False

    def _save_models(self):
        """Save trained models to disk"""
        try:
            model_path = "models/ml_models"
            os.makedirs(model_path, exist_ok=True)
            
            if self.late_prediction_model:
                joblib.dump(self.late_prediction_model, f"{model_path}/late_prediction_model.joblib")
                joblib.dump(self.scaler, f"{model_path}/scaler.joblib")
                
                with open(f"{model_path}/feature_columns.txt", 'w') as f:
                    f.write('\n'.join(self.feature_columns))
                
                logger.info("✅ ML models saved successfully")
                
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")

    async def train_prediction_models(self, db: Session) -> Dict[str, Any]:
        """Train ML models for attendance prediction"""
        try:
            logger.info("🚀 Starting ML model training...")
            
            # Get comprehensive training data
            training_data = await self._prepare_training_data(db)
            
            if len(training_data) < 100:
                return {
                    "error": "Insufficient data for training (need at least 100 records)",
                    "records_found": len(training_data),
                    "minimum_required": 100
                }
            
            logger.info(f"📊 Training data prepared: {len(training_data)} records")
            
            # Feature engineering
            features_df = await self._engineer_features(training_data, db)
            logger.info(f"🔧 Feature engineering completed: {features_df.shape[1]} features")
            
            # Train models
            results = {}
            
            # 1. Late arrival prediction
            late_results = await self._train_late_prediction_model(features_df)
            results["late_prediction"] = late_results
            logger.info("✅ Late prediction model trained")
            
            # 2. Early departure prediction  
            early_results = await self._train_early_departure_model(features_df)
            results["early_departure"] = early_results
            logger.info("✅ Early departure model trained")
            
            # Save models
            self._save_models()
            self.models_trained = True
            
            return {
                "training_completed": True,
                "models_trained": ["late_prediction", "early_departure"],
                "training_data_size": len(training_data),
                "feature_count": features_df.shape[1],
                "model_performance": results,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "insights": [
                    f"🎯 Trained on {len(training_data)} attendance records",
                    f"🔧 Generated {features_df.shape[1]} predictive features",
                    f"📈 Late prediction accuracy: {late_results.get('accuracy', 0):.1%}",
                    f"📉 Early departure accuracy: {early_results.get('accuracy', 0):.1%}",
                    "💾 Models saved and ready for predictions"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error training prediction models: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def _prepare_training_data(self, db: Session) -> pd.DataFrame:
        """Prepare comprehensive training dataset"""
        
        # Get last 6 months of data for training
        end_date = date.today()
        start_date = end_date - timedelta(days=180)
        
        logger.info(f"📅 Extracting training data from {start_date} to {end_date}")
        
        query = db.query(
            ProcessedAttendance.employee_id,
            ProcessedAttendance.date,
            ProcessedAttendance.is_present,
            ProcessedAttendance.is_late,
            ProcessedAttendance.check_in_time,
            ProcessedAttendance.check_out_time,
            ProcessedAttendance.late_minutes,
            ProcessedAttendance.total_working_hours,
            ProcessedAttendance.status,
            Employee.first_name,
            Employee.last_name,
            Employee.position,
            Employee.hire_date,
            Employee.date_of_birth,
            Employee.employee_id.label('emp_code'),
            Department.name.label('department_name')
        ).join(Employee, ProcessedAttendance.employee_id == Employee.id
        ).join(Department, Employee.department_id == Department.id, isouter=True
        ).filter(
            ProcessedAttendance.date >= start_date,
            ProcessedAttendance.date <= end_date,
            Employee.status == "ACTIVE"
        ).all()
        
        logger.info(f"📊 Raw query returned {len(query)} records")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'employee_id': row.employee_id,
            'emp_code': row.emp_code,
            'date': row.date,
            'is_present': row.is_present,
            'is_late': row.is_late,
            'check_in_time': row.check_in_time,
            'check_out_time': row.check_out_time,
            'late_minutes': row.late_minutes or 0,
            'total_hours': float(row.total_working_hours) if row.total_working_hours else 0,
            'status': row.status,
            'position': row.position or 'Unknown',
            'department': row.department_name or 'Unknown',
            'hire_date': row.hire_date,
            'birth_date': row.date_of_birth,
            'full_name': f"{row.first_name} {row.last_name}"
        } for row in query])
        
        logger.info(f"✅ DataFrame created with {len(df)} rows, {len(df.columns)} columns")
        return df

    async def _engineer_features(self, df: pd.DataFrame, db: Session) -> pd.DataFrame:
        """Engineer comprehensive features for prediction models"""
        
        logger.info("🔧 Starting feature engineering...")
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by employee and date
        df = df.sort_values(['employee_id', 'date']).reset_index(drop=True)
        
        # Basic date features
        df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        # Employee demographic features
        today = pd.Timestamp.now().date()
        
        # Handle tenure calculation safely
        df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')
        df['tenure_days'] = (df['date'] - df['hire_date']).dt.days.fillna(365)
        
        # Handle age calculation safely
        df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
        df['age'] = ((df['date'] - df['birth_date']).dt.days / 365.25).fillna(30)
        
        # Working hours analysis
        df['total_hours'] = pd.to_numeric(df['total_hours'], errors='coerce').fillna(8)
        df['is_short_day'] = (df['total_hours'] < 7).astype(int)
        df['is_long_day'] = (df['total_hours'] > 9).astype(int)
        df['is_normal_hours'] = ((df['total_hours'] >= 7) & (df['total_hours'] <= 9)).astype(int)
        
        # Early departure indicator
        df['left_early'] = (df['total_hours'] < 7.5).astype(int)
        
        # Season features
        df['season_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['season_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['season_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        df['season_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        # Simple rolling features (without complex calculations for now)
        for employee_id in df['employee_id'].unique():
            emp_mask = df['employee_id'] == employee_id
            emp_df = df[emp_mask].copy()
            
            # Simple 7-day rolling averages
            df.loc[emp_mask, 'late_freq_last_7d'] = emp_df['is_late'].rolling(
                window=7, min_periods=1).mean().fillna(0)
            df.loc[emp_mask, 'avg_hours_last_7d'] = emp_df['total_hours'].rolling(
                window=7, min_periods=1).mean().fillna(8)
        
        # Department encoding
        dept_dummies = pd.get_dummies(df['department'], prefix='dept')
        df = pd.concat([df, dept_dummies], axis=1)
        
        # Position features
        df['position'] = df['position'].fillna('Unknown').str.lower()
        df['position_manager'] = df['position'].str.contains('manager', na=False).astype(int)
        df['position_senior'] = df['position'].str.contains('senior', na=False).astype(int)
        df['position_junior'] = df['position'].str.contains('junior', na=False).astype(int)
        
        # Lag features
        df['prev_day_late'] = df.groupby('employee_id')['is_late'].shift(1).fillna(0)
        df['prev_day_hours'] = df.groupby('employee_id')['total_hours'].shift(1).fillna(8)
        
        logger.info(f"✅ Feature engineering completed: {df.shape[1]} total features")
        return df

    async def _train_late_prediction_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train model to predict late arrivals"""
        
        # Filter to present days only (can't be late if absent)
        present_df = df[df['is_present'] == True].copy()
        
        if len(present_df) < 50:
            return {"error": "Insufficient data for late prediction model"}
        
        # Features for late prediction
        feature_cols = [
            'day_of_week', 'month', 'is_monday', 'is_friday', 'is_weekend',
            'is_month_start', 'is_month_end', 'tenure_days', 'age',
            'late_freq_last_7d', 'late_freq_last_14d', 'late_freq_last_30d',
            'absent_freq_last_7d', 'absent_freq_last_14d', 'absent_freq_last_30d',
            'avg_hours_last_7d', 'avg_hours_last_14d', 'avg_hours_last_30d',
            'prev_day_late', 'prev_day_absent', 'prev_day_hours',
            'season_spring', 'season_summer', 'season_fall', 'season_winter'
        ]
        
        # Add department and position features
        dept_cols = [col for col in present_df.columns if col.startswith('dept_')]
        position_cols = [col for col in present_df.columns if col.startswith('position_')]
        feature_cols.extend(dept_cols + position_cols)
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in present_df.columns]
        
        X = present_df[available_features].fillna(0)
        y = present_df['is_late'].astype(int)
        
        logger.info(f"🎯 Training late prediction model with {len(available_features)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.late_prediction_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced',
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        self.late_prediction_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.late_prediction_model.predict(X_test_scaled)
        y_prob = self.late_prediction_model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.5
        
        # Feature importance
        feature_importance = list(zip(available_features, self.late_prediction_model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        self.feature_columns = available_features
        
        return {
            "model_type": "RandomForestClassifier",
            "accuracy": round(accuracy, 3),
            "auc_score": round(auc_score, 3),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(available_features),
            "top_features": feature_importance[:10],
            "class_distribution": y.value_counts().to_dict(),
            "model_params": self.late_prediction_model.get_params()
        }

    async def _train_early_departure_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train model to predict early departures"""
        
        # Filter to present days only
        present_df = df[df['is_present'] == True].copy()
        
        if len(present_df) < 50:
            return {"error": "Insufficient data for early departure model"}
        
        # Features for early departure prediction
        feature_cols = [
            'day_of_week', 'month', 'is_monday', 'is_friday', 'is_weekend',
            'check_in_hour', 'late_check_in', 'very_late_check_in',
            'tenure_days', 'age', 'is_late',
            'early_dept_freq_last_7d', 'early_dept_freq_last_14d', 'early_dept_freq_last_30d',
            'avg_hours_last_7d', 'avg_hours_last_14d', 'avg_hours_last_30d',
            'prev_day_hours', 'prev_day_left_early',
            'season_spring', 'season_summer', 'season_fall', 'season_winter'
        ]
        
        # Add department and position features
        dept_cols = [col for col in present_df.columns if col.startswith('dept_')]
        position_cols = [col for col in present_df.columns if col.startswith('position_')]
        feature_cols.extend(dept_cols + position_cols)
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in present_df.columns]
        
        X = present_df[available_features].fillna(0)
        y = present_df['left_early'].astype(int)
        
        logger.info(f"🎯 Training early departure model with {len(available_features)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features (use same scaler as late prediction for consistency)
        early_scaler = StandardScaler()
        X_train_scaled = early_scaler.fit_transform(X_train)
        X_test_scaled = early_scaler.transform(X_test)
        
        # Train model
        self.early_departure_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight='balanced',
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        self.early_departure_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.early_departure_model.predict(X_test_scaled)
        y_prob = self.early_departure_model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.5
        
        # Feature importance
        feature_importance = list(zip(available_features, self.early_departure_model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "model_type": "RandomForestClassifier",
            "accuracy": round(accuracy, 3),
            "auc_score": round(auc_score, 3),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(available_features),
            "top_features": feature_importance[:10],
            "class_distribution": y.value_counts().to_dict()
        }

    async def predict_tomorrow_late_arrivals(self, db: Session, limit: int = 10, 
                                           department_filter: str = None, 
                                           risk_threshold: float = 30.0) -> Dict[str, Any]:
        """🔮 Predict who is most likely to be late tomorrow"""
        
        if not self.models_trained or not self.late_prediction_model:
            return {
                "error": "Late prediction model not trained yet",
                "suggestion": "Please train the models first using the train_attendance_prediction_models function",
                "models_available": self.models_trained
            }
        
        try:
            tomorrow = date.today() + timedelta(days=1)
            logger.info(f"🔮 Predicting late arrivals for {tomorrow}")
            
            # Get active employees
            employees_query = db.query(Employee).filter(Employee.status == "ACTIVE")
            
            if department_filter:
                employees_query = employees_query.join(Department).filter(
                    Department.name.ilike(f"%{department_filter}%")
                )
            
            employees = employees_query.all()
            logger.info(f"👥 Analyzing {len(employees)} employees")
            
            predictions = []
            
            for emp in employees:
                # Get employee's recent history
                recent_data = await self._get_employee_recent_data(db, emp.id, days=30)
                
                if len(recent_data) < 5:  # Need some history
                    continue
                
                # Engineer features for tomorrow
                features = await self._engineer_tomorrow_features(db, emp, tomorrow, recent_data)
                
                if features is not None and len(features) == len(self.feature_columns):
                    # Make prediction
                    features_array = np.array(features).reshape(1, -1)
                    features_scaled = self.scaler.transform(features_array)
                    late_probability = self.late_prediction_model.predict_proba(features_scaled)[0][1]
                    
                    # Only include if above threshold
                    if late_probability * 100 >= risk_threshold:
                        # Get department info
                        dept = db.query(Department).filter(Department.id == emp.department_id).first()
                        
                        # Get risk factors
                        risk_factors = await self._analyze_late_risk_factors(recent_data, tomorrow)
                        
                        predictions.append({
                            "employee_id": emp.employee_id,
                            "name": f"{emp.first_name} {emp.last_name}",
                            "department": dept.name if dept else "Unknown",
                            "position": emp.position,
                            "late_probability": round(late_probability * 100, 1),
                            "risk_factors": risk_factors,
                            "recent_late_pattern": await self._get_recent_late_pattern(recent_data),
                            "risk_level": self._categorize_risk(late_probability)
                        })
            
            # Sort by probability and get top predictions
            predictions.sort(key=lambda x: x["late_probability"], reverse=True)
            top_predictions = predictions[:limit]
            
            # Generate insights
            insights = self._generate_late_prediction_insights(predictions, tomorrow)
            
            return {
                "prediction_date": tomorrow.isoformat(),
                "day_of_week": tomorrow.strftime("%A"),
                "total_employees_analyzed": len(employees),
                "employees_with_sufficient_data": len([p for p in predictions if p]),
                "high_risk_employees": len([p for p in predictions if p["late_probability"] > 70]),
                "medium_risk_employees": len([p for p in predictions if 40 <= p["late_probability"] <= 70]),
                "top_late_predictions": top_predictions,
                "all_at_risk": predictions if len(predictions) <= 20 else predictions[:20],
                "insights": insights,
                "model_confidence": "High" if len(predictions) > 20 else "Medium",
                "threshold_used": risk_threshold,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting late arrivals: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def predict_early_departures_future_date(self, db: Session, target_date: date, 
                                             limit: int = 10, department_filter: str = None) -> Dict[str, Any]:
        """🔮 Predict early departures for a future date"""
        
        try:
            logger.info(f"🔮 Predicting early departures for {target_date}")
            
            # Get employees who are likely to be present on target date
            employees_query = db.query(Employee).filter(Employee.status == "ACTIVE")
            
            if department_filter:
                employees_query = employees_query.join(Department).filter(
                    Department.name.ilike(f"%{department_filter}%")
                )
            
            employees = employees_query.all()
            predictions = []
            
            for emp in employees:
                # Get historical early departure pattern
                early_pattern = await self._analyze_early_departure_pattern(db, emp.id, days=30)
                
                # Calculate risk for target date
                risk_factors = []
                risk_score = 0.0
                
                # Base risk from historical pattern
                historical_rate = early_pattern.get("early_departure_rate", 0) / 100
                risk_score += historical_rate * 0.6
                
                # Day of week effects
                if target_date.weekday() == 4:  # Friday
                    risk_score += 0.4
                    risk_factors.append("Friday early departure tendency")
                
                # Month-end effect
                if target_date.day >= 28:
                    risk_score += 0.2
                    risk_factors.append("End of month completion rush")
                
                # Pre-weekend effect
                if target_date.weekday() == 4:  # Friday
                    risk_factors.append("Pre-weekend departure pattern")
                
                # High historical pattern
                if historical_rate > 0.3:
                    risk_factors.append("High historical early departure rate")
                elif historical_rate > 0.1:
                    risk_factors.append("Moderate early departure tendency")
                
                # Only include significant risks
                if risk_score > 0.3:
                    dept = db.query(Department).filter(Department.id == emp.department_id).first()
                    
                    predictions.append({
                        "employee_id": emp.employee_id,
                        "name": f"{emp.first_name} {emp.last_name}",
                        "department": dept.name if dept else "Unknown",
                        "position": emp.position,
                        "early_departure_probability": round(min(risk_score * 100, 100), 1),
                        "risk_factors": risk_factors,
                        "historical_pattern": early_pattern,
                        "risk_level": self._categorize_risk(risk_score)
                    })
            
            # Sort by probability
            predictions.sort(key=lambda x: x["early_departure_probability"], reverse=True)
            
            # Generate insights
            insights = [
                f"📅 Early departure predictions for {target_date.strftime('%A, %B %d, %Y')}",
                f"🔍 Analyzed {len(employees)} employees"
            ]
            
            if predictions:
                high_risk = len([p for p in predictions if p["early_departure_probability"] > 70])
                if high_risk > 0:
                    insights.append(f"🚨 {high_risk} employees at high risk of leaving early")
                
                if target_date.weekday() == 4:
                    insights.append(f"🎯 Friday effect: {len(predictions)} employees flagged")
                    insights.append("💡 Consider flexible Friday schedules or early wrap-up meetings")
            else:
                insights.append("✅ No significant early departure risks detected")
            
            return {
                "prediction_date": target_date.isoformat(),
                "day_of_week": target_date.strftime("%A"),
                "total_employees_analyzed": len(employees),
                "employees_at_risk": len(predictions),
                "high_risk_count": len([p for p in predictions if p["early_departure_probability"] > 70]),
                "top_early_departure_predictions": predictions[:limit],
                "all_predictions": predictions if len(predictions) <= 20 else predictions[:20],
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting future early departures: {str(e)}")
            return {"error": str(e)}

    async def predict_today_early_departures(self, db: Session, limit: int = 10, 
                                           department_filter: str = None) -> Dict[str, Any]:
        """🕐 Predict who is likely to leave early today"""
        
        try:
            today = date.today()
            current_time = datetime.now().time()
            
            logger.info(f"🕐 Predicting early departures for {today} at {current_time}")
            
            # Only make predictions if it's still work hours (before 3 PM)
            if current_time.hour >= 15:
                return {
                    "message": "Too late in the day for early departure predictions",
                    "current_time": current_time.strftime("%H:%M"),
                    "suggestion": "Early departure predictions are most accurate when made before 3:00 PM"
                }
            
            # Get employees who are present today
            present_query = db.query(
                ProcessedAttendance.employee_id,
                ProcessedAttendance.check_in_time,
                ProcessedAttendance.is_present,
                ProcessedAttendance.is_late,
                Employee.first_name,
                Employee.last_name,
                Employee.employee_id.label('emp_id'),
                Employee.position,
                Department.name.label('department_name')
            ).join(Employee, ProcessedAttendance.employee_id == Employee.id
            ).join(Department, Employee.department_id == Department.id, isouter=True
            ).filter(
                ProcessedAttendance.date == today,
                ProcessedAttendance.is_present == True,
                Employee.status == "ACTIVE"
            )
            
            if department_filter:
                present_query = present_query.filter(
                    Department.name.ilike(f"%{department_filter}%")
                )
            
            present_today = present_query.all()
            logger.info(f"👥 Found {len(present_today)} employees present today")
            
            predictions = []
            
            for record in present_today:
                # Get historical pattern for early departures
                early_departure_pattern = await self._analyze_early_departure_pattern(
                    db, record.employee_id, days=30
                )
                
                # Calculate risk factors
                risk_factors = await self._calculate_early_departure_risk(
                    db, record.employee_id, today, record.check_in_time, 
                    record.is_late, early_departure_pattern
                )
                
                if risk_factors["total_risk"] > 0.3:  # Only include significant risks
                    predictions.append({
                        "employee_id": record.emp_id,
                        "name": f"{record.first_name} {record.last_name}",
                        "department": record.department_name or "Unknown",
                        "position": record.position,
                        "check_in_time": record.check_in_time.strftime("%H:%M") if record.check_in_time else "Unknown",
                        "was_late_today": record.is_late,
                        "early_departure_probability": round(risk_factors["total_risk"] * 100, 1),
                        "risk_factors": risk_factors["factors"],
                        "historical_pattern": early_departure_pattern,
                        "risk_level": self._categorize_risk(risk_factors["total_risk"])
                    })
            
            # Sort by probability
            predictions.sort(key=lambda x: x["early_departure_probability"], reverse=True)
            
            # Generate insights
            insights = self._generate_early_departure_insights(predictions, today)
            
            return {
                "prediction_date": today.isoformat(),
                "prediction_time": current_time.strftime("%H:%M"),
                "day_of_week": today.strftime("%A"),
                "employees_present_today": len(present_today),
                "employees_at_risk": len(predictions),
                "high_risk_count": len([p for p in predictions if p["early_departure_probability"] > 70]),
                "top_early_departure_predictions": predictions[:limit],
                "all_at_risk": predictions if len(predictions) <= 15 else predictions[:15],
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting early departures: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    # Helper methods
    async def _get_employee_recent_data(self, db: Session, employee_id: int, days: int = 30) -> List[Dict]:
        """Get employee's recent attendance data"""
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        records = db.query(ProcessedAttendance).filter(
            ProcessedAttendance.employee_id == employee_id,
            ProcessedAttendance.date >= start_date,
            ProcessedAttendance.date <= end_date
        ).order_by(ProcessedAttendance.date.desc()).all()
        
        return [{
            "date": record.date,
            "is_present": record.is_present,
            "is_late": record.is_late,
            "late_minutes": record.late_minutes or 0,
            "total_hours": float(record.total_working_hours) if record.total_working_hours else 0,
            "check_in_time": record.check_in_time,
            "check_out_time": record.check_out_time,
            "day_of_week": record.date.weekday()
        } for record in records]

    async def _engineer_tomorrow_features(self, db: Session, employee: Employee, 
                                        tomorrow: date, recent_data: List[Dict]) -> Optional[List[float]]:
        """Engineer features for tomorrow's prediction"""
        
        try:
            features = []
            
            # Basic date features for tomorrow
            features.extend([
                tomorrow.weekday(),  # day_of_week
                tomorrow.month,      # month
                1 if tomorrow.weekday() == 0 else 0,  # is_monday
                1 if tomorrow.weekday() == 4 else 0,  # is_friday
                1 if tomorrow.weekday() >= 5 else 0,  # is_weekend
                1 if tomorrow.day <= 3 else 0,        # is_month_start
                1 if tomorrow.day >= 28 else 0,       # is_month_end
            ])
            
            # Employee features
            today = date.today()
            tenure_days = (tomorrow - employee.hire_date.date()).days if employee.hire_date else 365
            age = (tomorrow - employee.date_of_birth).days / 365.25 if employee.date_of_birth else 30
            
            features.extend([tenure_days, age])
            
            # Calculate rolling statistics from recent data
            if len(recent_data) >= 7:
                last_7_late = sum(1 for r in recent_data[:7] if r["is_late"]) / 7
                last_7_absent = sum(1 for r in recent_data[:7] if not r["is_present"]) / 7
                last_7_hours = sum(r["total_hours"] for r in recent_data[:7] if r["is_present"]) / max(1, sum(1 for r in recent_data[:7] if r["is_present"]))
            else:
                last_7_late = last_7_absent = 0
                last_7_hours = 8
            
            if len(recent_data) >= 14:
                last_14_late = sum(1 for r in recent_data[:14] if r["is_late"]) / 14
                last_14_absent = sum(1 for r in recent_data[:14] if not r["is_present"]) / 14
                last_14_hours = sum(r["total_hours"] for r in recent_data[:14] if r["is_present"]) / max(1, sum(1 for r in recent_data[:14] if r["is_present"]))
            else:
                last_14_late = last_7_late
                last_14_absent = last_7_absent
                last_14_hours = last_7_hours
            
            if len(recent_data) >= 30:
                last_30_late = sum(1 for r in recent_data[:30] if r["is_late"]) / 30
                last_30_absent = sum(1 for r in recent_data[:30] if not r["is_present"]) / 30
                last_30_hours = sum(r["total_hours"] for r in recent_data[:30] if r["is_present"]) / max(1, sum(1 for r in recent_data[:30] if r["is_present"]))
            else:
                last_30_late = last_14_late
                last_30_absent = last_14_absent  
                last_30_hours = last_14_hours
            
            features.extend([
                last_7_late, last_14_late, last_30_late,
                last_7_absent, last_14_absent, last_30_absent,
                last_7_hours, last_14_hours, last_30_hours
            ])
            
            # Previous day features
            if recent_data:
                prev_day = recent_data[0]
                features.extend([
                    1 if prev_day["is_late"] else 0,
                    1 if not prev_day["is_present"] else 0,
                    prev_day["total_hours"]
                ])
            else:
                features.extend([0, 0, 8])
            
            # Season features
            season_features = [0, 0, 0, 0]  # spring, summer, fall, winter
            if tomorrow.month in [3, 4, 5]:
                season_features[0] = 1
            elif tomorrow.month in [6, 7, 8]:
                season_features[1] = 1
            elif tomorrow.month in [9, 10, 11]:
                season_features[2] = 1
            else:
                season_features[3] = 1
            
            features.extend(season_features)
            
            # Department features (one-hot encoded)
            dept = db.query(Department).filter(Department.id == employee.department_id).first()
            dept_name = dept.name if dept else "Unknown"
            
            # Get all department names for one-hot encoding
            all_depts = db.query(Department.name).filter(Department.is_active == True).all()
            dept_names = [d[0] for d in all_depts]
            
            dept_features = [1 if f"dept_{dept_name}" == f"dept_{d}" else 0 for d in dept_names]
            features.extend(dept_features)
            
            # Position features
            position = (employee.position or "").lower()
            position_features = [
                1 if "manager" in position else 0,
                1 if "senior" in position else 0,
                1 if "junior" in position else 0,
                1 if "intern" in position else 0,
                1 if "lead" in position else 0,
                1 if "director" in position else 0,
                1 if not any(cat in position for cat in ["manager", "senior", "junior", "intern", "lead", "director"]) else 0
            ]
            features.extend(position_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error engineering tomorrow features: {str(e)}")
            return None

    async def _analyze_late_risk_factors(self, recent_data: List[Dict], tomorrow: date) -> List[str]:
        """Analyze risk factors for being late"""
        factors = []
        
        if not recent_data:
            return factors
        
        # Recent late pattern
        recent_late_count = sum(1 for r in recent_data[:7] if r["is_late"])
        if recent_late_count >= 2:
            factors.append(f"Late {recent_late_count} times in past week")
        
        # Day of week pattern
        tomorrow_dow = tomorrow.weekday()
        same_dow_records = [r for r in recent_data if r["day_of_week"] == tomorrow_dow]
        same_dow_late = sum(1 for r in same_dow_records if r["is_late"])
        
        if len(same_dow_records) >= 3 and same_dow_late / len(same_dow_records) > 0.5:
            day_name = tomorrow.strftime("%A")
            factors.append(f"Often late on {day_name}s")
        
        # Monday effect
        if tomorrow_dow == 0:  # Monday
            factors.append("Monday return-to-work effect")
        
        # Friday effect  
        if tomorrow_dow == 4:  # Friday
            friday_lates = sum(1 for r in recent_data if r["day_of_week"] == 4 and r["is_late"])
            friday_total = sum(1 for r in recent_data if r["day_of_week"] == 4)
            if friday_total > 0 and friday_lates / friday_total > 0.3:
                factors.append("Friday lateness pattern")
        
        return factors

    async def _get_recent_late_pattern(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """Get recent lateness pattern summary"""
        
        if not recent_data:
            return {"late_days": 0, "total_days": 0, "pattern": "No data"}
        
        present_days = [r for r in recent_data if r["is_present"]]
        late_days = [r for r in present_days if r["is_late"]]
        
        return {
            "late_days": len(late_days),
            "total_days": len(present_days),
            "late_rate": round(len(late_days) / max(1, len(present_days)) * 100, 1),
            "avg_late_minutes": round(sum(r["late_minutes"] for r in late_days) / max(1, len(late_days)), 1),
            "pattern": "High" if len(late_days) / max(1, len(present_days)) > 0.3 else "Low"
        }

    async def _analyze_early_departure_pattern(self, db: Session, employee_id: int, days: int = 30) -> Dict[str, Any]:
        """Analyze employee's early departure pattern"""
        
        recent_data = await self._get_employee_recent_data(db, employee_id, days)
        
        early_departures = 0
        total_present_days = 0
        avg_hours = []
        
        for record in recent_data:
            if record["is_present"]:
                total_present_days += 1
                hours = record["total_hours"]
                avg_hours.append(hours)
                
                # Consider < 7.5 hours as early departure
                if hours < 7.5:
                    early_departures += 1
        
        early_departure_rate = (early_departures / total_present_days) if total_present_days > 0 else 0
        average_hours = sum(avg_hours) / len(avg_hours) if avg_hours else 8
        
        return {
            "early_departure_rate": round(early_departure_rate * 100, 1),
            "average_daily_hours": round(average_hours, 1),
            "days_analyzed": total_present_days,
            "early_departures_count": early_departures,
            "pattern_strength": "High" if early_departure_rate > 0.3 else "Medium" if early_departure_rate > 0.1 else "Low"
        }

    async def _calculate_early_departure_risk(self, db: Session, employee_id: int, 
                                           today: date, check_in_time, was_late_today: bool,
                                           historical_pattern: Dict) -> Dict[str, Any]:
        """Calculate risk factors for early departure today"""
        
        risk_factors = []
        total_risk = 0.0
        
        # Factor 1: Historical pattern
        historical_rate = historical_pattern.get("early_departure_rate", 0) / 100
        if historical_rate > 0.3:
            risk_factors.append("High historical early departure rate")
            total_risk += 0.4
        elif historical_rate > 0.1:
            risk_factors.append("Moderate early departure tendency")
            total_risk += 0.2
        
        # Factor 2: Day of week (Friday effect)
        if today.weekday() == 4:  # Friday
            risk_factors.append("Friday departure tendency")
            total_risk += 0.25
        
        # Factor 3: Late arrival today (might compensate by leaving early)
        if was_late_today:
            risk_factors.append("Arrived late today")
            total_risk += 0.3
        
        # Factor 4: Very late check-in (stress indicator)
        if check_in_time and check_in_time.hour >= 11:
            risk_factors.append("Very late check-in today")
            total_risk += 0.2
        
        # Factor 5: Low average hours pattern
        avg_hours = historical_pattern.get("average_daily_hours", 8)
        if avg_hours < 7.5:
            risk_factors.append("Consistently works shorter hours")
            total_risk += 0.3
        
        # Factor 6: Month-end effect
        if today.day >= 28:
            risk_factors.append("End of month workload completion")
            total_risk += 0.1
        
        return {
            "total_risk": min(total_risk, 1.0),  # Cap at 100%
            "factors": risk_factors
        }

    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level based on probability"""
        if probability > 0.7:
            return "High"
        elif probability > 0.4:
            return "Medium"
        else:
            return "Low"

    def _generate_late_prediction_insights(self, predictions: List[Dict], tomorrow: date) -> List[str]:
        """Generate insights for late predictions"""
        insights = []
        
        if not predictions:
            insights.append("✅ No high-risk late arrivals predicted for tomorrow")
            return insights
        
        high_risk = [p for p in predictions if p["late_probability"] > 70]
        medium_risk = [p for p in predictions if 40 <= p["late_probability"] <= 70]
        
        # Day of week context
        day_name = tomorrow.strftime("%A")
        insights.append(f"📅 Predictions for {day_name}, {tomorrow.strftime('%B %d, %Y')}")
        
        if high_risk:
            insights.append(f"🚨 {len(high_risk)} employees at high risk of being late")
            insights.append(f"📊 Highest risk: {high_risk[0]['name']} ({high_risk[0]['late_probability']}%)")
        
        if medium_risk:
            insights.append(f"⚠️ {len(medium_risk)} employees at medium risk")
        
        # Department patterns
        dept_risks = {}
        for p in predictions[:10]:
            dept = p["department"]
            if dept not in dept_risks:
                dept_risks[dept] = []
            dept_risks[dept].append(p["late_probability"])
        
        if dept_risks:
            dept_avg = {dept: sum(risks)/len(risks) for dept, risks in dept_risks.items()}
            highest_risk_dept = max(dept_avg.items(), key=lambda x: x[1])
            insights.append(f"🏢 Department with highest avg risk: {highest_risk_dept[0]} ({highest_risk_dept[1]:.1f}%)")
        
        # Day-specific insights
        if tomorrow.weekday() == 0:  # Monday
            monday_predictions = len(predictions)
            insights.append(f"📈 Monday effect: {monday_predictions} employees flagged for potential lateness")
        elif tomorrow.weekday() == 4:  # Friday
            insights.append("🎯 Friday factor included in risk assessment")
        
        return insights

    def _generate_early_departure_insights(self, predictions: List[Dict], today: date) -> List[str]:
        """Generate insights for early departure predictions"""
        insights = []
        
        if not predictions:
            insights.append("✅ No significant early departure risks detected")
            return insights
        
        high_risk = [p for p in predictions if p["early_departure_probability"] > 70]
        medium_risk = [p for p in predictions if 40 <= p["early_departure_probability"] <= 70]
        
        day_name = today.strftime("%A")
        insights.append(f"📅 Early departure analysis for {day_name}")
        
        if high_risk:
            insights.append(f"🕐 {len(high_risk)} employees at high risk of leaving early")
            insights.append(f"📊 Highest risk: {high_risk[0]['name']} ({high_risk[0]['early_departure_probability']}%)")
        
        if medium_risk:
            insights.append(f"⚠️ {len(medium_risk)} employees at medium risk")
        
        # Analyze common risk factors
        all_factors = []
        for p in predictions:
            all_factors.extend(p["risk_factors"])
        
        if all_factors:
            factor_counts = {}
            for factor in all_factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
            most_common = max(factor_counts.items(), key=lambda x: x[1])
            insights.append(f"📈 Most common risk factor: {most_common[0]} ({most_common[1]} employees)")
        
        # Friday-specific insights
        if today.weekday() == 4:
            friday_risks = len(predictions)
            insights.append(f"🎯 Friday effect: {friday_risks} employees flagged for early departure risk")
        
        # Late arrival correlation
        late_today_count = sum(1 for p in predictions if p.get("was_late_today", False))
        if late_today_count > 0:
            insights.append(f"⏰ {late_today_count} at-risk employees arrived late today")
        
        return insights

# Initialize the prediction engine
prediction_engine = AttendancePredictionEngine()

logger.info("🔮 ML Prediction Engine initialized")