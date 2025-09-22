import pandas as pd
import requests
from urllib.parse import urlparse
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_url_availability(url, timeout=10):
    """Check if URL is accessible"""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code < 400
    except:
        return False

def check_urls_in_cell(url_cell):
    """Check multiple URLs separated by semicolons"""
    if pd.isna(url_cell) or not url_cell.strip():
        return False
    
    urls = [url.strip() for url in str(url_cell).split(';') if url.strip()]
    return any(check_url_availability(url) for url in urls)

def clean_retraction_data(input_file='papers.csv', output_file='cleaned.csv'):
    """
    Remove entries that are:
    - Not paywalled (Paywalled != 'Yes')
    - Don't have 'Issues About Data' in Reason
    - Have accessible URLs
    """
    
    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} records")
    
    initial_count = len(df)
    
    # Filter 1: Keep only paywalled papers
    df = df[df['Paywalled'].str.strip().str.lower() == 'no']
    logger.info(f"After paywall filter: {len(df)} records ({initial_count - len(df)} removed)")
    
    # Filter 2: Keep only papers with "Issues About Data" in Reason
    df = df[df['Reason'].str.contains('Issues About Data', case=False, na=False)]
    logger.info(f"After data issues filter: {len(df)} records")
    
    # Filter 3: Remove papers with accessible URLs
    logger.info("Checking URL accessibility...")
    
    urls_accessible = []
    for idx, row in df.iterrows():
        accessible = check_urls_in_cell(row['URLS'])
        urls_accessible.append(accessible)
        
        if idx % 10 == 0:  # Progress indicator
            logger.info(f"Checked {idx + 1}/{len(df)} URLs")
        
        time.sleep(0.1)  # Be nice to servers
    
    # Keep only entries where URLs are NOT accessible
    df = df[~pd.Series(urls_accessible, index=df.index)]
    logger.info(f"Final count after URL filter: {len(df)} records")
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    logger.info(f"Cleaned data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_retraction_data()
    print(f"Cleaning complete. Final dataset has {len(cleaned_df)} records.")