import pandas as pd
import numpy as np
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_and_clean_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Loads, cleans, and pre-processes the corporate credit data.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found at: {filepath}")
        return None
        
    try:
        df = pd.read_csv(filepath)
        
        # --- Target Engineering ---
        # 1=Safe (AAA), 0=Risky (Junk). We want 1=Default/Risky.
        if 'Binary Rating' in df.columns:
            df['Default'] = df['Binary Rating'].apply(lambda x: 1 if x == 0 else 0)
        else:
            raise ValueError("Target column 'Binary Rating' not found.")
            
        # --- ID Engineering ---
        # Prioritize Ticker, fall back to Corporation name
        if 'Ticker' in df.columns:
            df['Company_ID'] = df['Ticker']
        elif 'Corporation' in df.columns:
            df['Company_ID'] = df['Corporation']
        
        # Display Name
        df['Display_Name'] = df.apply(
            lambda x: f"{x['Corporation']} ({x['Company_ID']})" if 'Corporation' in df.columns else x['Company_ID'],
            axis=1
        )
        
        # --- Feature Engineering ---
        # Ensure strict numeric types and handle infinite values
        from src.config import FEATURES  # Import here to avoid circular dependency if structure changes
        for col in FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0 # Handle missing columns gracefully
                
        return df
        
    except Exception as e:
        logger.exception(f"Critical error loading data: {e}")
        return None
