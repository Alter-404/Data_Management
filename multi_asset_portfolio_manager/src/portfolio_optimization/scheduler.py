"""
Portfolio scheduling module for enforcing trading rules.
"""

from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

def normalize_datetime(dt):
    """Convert datetime to timezone-naive."""
    if pd.api.types.is_datetime64_dtype(dt):
        # For pandas datetime objects
        return pd.Timestamp(dt).to_pydatetime().replace(tzinfo=None)
    elif isinstance(dt, pd.Timestamp):
        # For pandas Timestamp objects
        return dt.to_pydatetime().replace(tzinfo=None)
    elif hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        # For timezone-aware datetime objects
        return dt.replace(tzinfo=None)
    return dt

class PortfolioScheduler:
    """Portfolio scheduler that enforces trading rules."""
    
    def __init__(self, 
                start_date: datetime = datetime(2015, 1, 1),
                end_date: datetime = datetime(2024, 12, 31),
                training_end_date: datetime = datetime(2022, 12, 31)):
        """Initialize the scheduler with the evaluation period."""
        # Normalize dates to avoid timezone issues
        self.start_date = normalize_datetime(start_date)
        self.end_date = normalize_datetime(end_date)
        self.training_end_date = normalize_datetime(training_end_date)
        
        self.logger = logging.getLogger(__name__)
        self.trading_days = self._generate_trading_days()
        
        # Filter days using normalized dates
        self.training_days = [d for d in self.trading_days if d <= self.training_end_date]
        self.evaluation_days = [d for d in self.trading_days if d > self.training_end_date]
        self.next_trading_day_idx = 0
        
    def _generate_trading_days(self) -> List[datetime]:
        """Generate a list of trading days (Mondays only) within the evaluation period."""
        trading_days = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # If Monday (weekday 0)
            if current_date.weekday() == 0:
                trading_days.append(current_date)
            current_date += timedelta(days=1)
            
        self.logger.info(f"Generated {len(trading_days)} trading days between {self.start_date} and {self.end_date}")
        return trading_days
    
    def get_next_trading_day(self) -> Optional[datetime]:
        """Get the next trading day in the schedule."""
        if self.next_trading_day_idx >= len(self.trading_days):
            return None
        
        next_day = self.trading_days[self.next_trading_day_idx]
        self.next_trading_day_idx += 1
        return next_day
    
    def reset(self):
        """Reset the scheduler to the beginning of the evaluation period."""
        self.next_trading_day_idx = 0
    
    def is_trading_day(self, date: datetime) -> bool:
        """Check if a given date is a trading day."""
        # Normalize input date for comparison
        date_normalized = normalize_datetime(date)
        return date_normalized.weekday() == 0 and self.start_date <= date_normalized <= self.end_date
    
    def is_training_day(self, date: datetime) -> bool:
        """Check if a given date is in the training period."""
        # Normalize input date for comparison
        date_normalized = normalize_datetime(date)
        return self.is_trading_day(date) and date_normalized <= self.training_end_date
    
    def is_evaluation_day(self, date: datetime) -> bool:
        """Check if a given date is in the evaluation period."""
        # Normalize input date for comparison
        date_normalized = normalize_datetime(date)
        return self.is_trading_day(date) and date_normalized > self.training_end_date
    
    def get_remaining_trading_days(self) -> List[datetime]:
        """Get all remaining trading days in the schedule."""
        return self.trading_days[self.next_trading_day_idx:]
    
    def get_evaluation_period(self) -> Tuple[datetime, datetime]:
        """Get the evaluation period start and end dates."""
        return (self.training_end_date + timedelta(days=1), self.end_date)
    
    def get_training_period(self) -> Tuple[datetime, datetime]:
        """Get the training period start and end dates."""
        return (self.start_date, self.training_end_date)
        
    def get_training_days(self) -> List[datetime]:
        """Get all trading days in the training period."""
        return self.training_days
        
    def get_evaluation_days(self) -> List[datetime]:
        """Get all trading days in the evaluation period."""
        return self.evaluation_days