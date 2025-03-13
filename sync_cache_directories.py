#!/usr/bin/env python
import os
import shutil
import pandas as pd

def sync_cache_directories():
    """Sync data files between main and alternative cache directories"""
    main_cache_dir = '/home/panal/Documents/dashboard-trading/data/cache'
    alt_cache_dir = '/home/panal/Documents/bot-machine-learning-main/data/cache'
    
    # Create directories if they don't exist
    os.makedirs(main_cache_dir, exist_ok=True)
    os.makedirs(alt_cache_dir, exist_ok=True)
    
    # Get files from both directories
    main_files = set([f for f in os.listdir(main_cache_dir) if f.endswith('.pkl')])
    alt_files = set([f for f in os.listdir(alt_cache_dir) if f.endswith('.pkl')])
    
    # Find files that are in one directory but not the other
    main_only = main_files - alt_files
    alt_only = alt_files - main_files
    
    print(f"Found {len(main_only)} files only in main cache and {len(alt_only)} files only in alternative cache")
    
    # Copy main-only files to alt directory
    for file in main_only:
        src = os.path.join(main_cache_dir, file)
        dst = os.path.join(alt_cache_dir, file)
        try:
            shutil.copy2(src, dst)
            print(f"Copied from main to alt: {file}")
        except Exception as e:
            print(f"Error copying {file}: {str(e)}")
    
    # Copy alt-only files to main directory
    for file in alt_only:
        src = os.path.join(alt_cache_dir, file)
        dst = os.path.join(main_cache_dir, file)
        try:
            shutil.copy2(src, dst)
            print(f"Copied from alt to main: {file}")
        except Exception as e:
            print(f"Error copying {file}: {str(e)}")
    
    print("Cache directories synchronized successfully!")
    
    # Verify data integrity by loading a few files
    print("\nVerifying data integrity...")
    all_files = list(main_files.union(alt_files))
    
    # Sample up to 5 files for verification
    import random
    sample_files = random.sample(all_files, min(5, len(all_files)))
    
    for file in sample_files:
        main_path = os.path.join(main_cache_dir, file)
        alt_path = os.path.join(alt_cache_dir, file)
        
        try:
            if os.path.exists(main_path):
                df = pd.read_pickle(main_path)
                print(f"✓ Main: {file} ({len(df)} rows)")
                
            if os.path.exists(alt_path):
                df = pd.read_pickle(alt_path)
                print(f"✓ Alt: {file} ({len(df)} rows)")
                
        except Exception as e:
            print(f"✗ Error verifying {file}: {str(e)}")

if __name__ == "__main__":
    print("=== SYNCHRONIZING CACHE DIRECTORIES ===")
    sync_cache_directories()
    print("\nDone! The data should now be accessible to all analysis scripts.")
