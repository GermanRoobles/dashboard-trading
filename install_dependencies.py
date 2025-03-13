#!/usr/bin/env python
import subprocess
import sys
import importlib

def check_and_install_packages():
    """Check for and install missing dependencies"""
    print("Checking for missing dependencies...")
    
    # List of required packages with their installation names
    required_packages = [
        ('pandas', 'pandas>=1.5.0'),
        ('numpy', 'numpy>=1.24.0'),
        ('matplotlib', 'matplotlib>=3.5.2'),
        ('ta', 'ta>=0.10.2'),            # Technical analysis package
        ('ccxt', 'ccxt>=2.6.23'),        # Crypto exchange API
        ('dash', 'dash>=2.6.0'),         # Dashboard
        ('plotly', 'plotly>=5.9.0'),     # Plotting for dash
        ('seaborn', 'seaborn>=0.11.2'),  # Enhanced visualization
    ]
    
    missing_packages = []
    
    # Check for each package
    for package_name, install_name in required_packages:
        try:
            importlib.import_module(package_name)
            # If import succeeds, check version
            try:
                module = importlib.import_module(package_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {package_name}: {version}")
            except (ImportError, AttributeError):
                print(f"✓ {package_name}: installed (version unknown)")
        except ImportError:
            print(f"✗ {package_name}: not found")
            missing_packages.append(install_name)
    
    # Install missing packages
    if missing_packages:
        print("\nInstalling missing packages...")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}")
    else:
        print("\nAll required packages are installed!")
    
    # Add post-installation verification
    if 'ta' in [pkg[0] for pkg in required_packages]:
        try:
            import ta
            print("\nVerifying TA-lib alternative:")
            # Fix version detection for TA module
            try:
                version = ta.__version__
            except AttributeError:
                # Use a more modern approach instead of pkg_resources
                try:
                    import importlib.metadata
                    version = importlib.metadata.version('ta')
                except ImportError:
                    # Fall back to pkg_resources only if importlib.metadata is not available
                    try:
                        import pkg_resources
                        version = pkg_resources.get_distribution("ta").version
                    except:
                        version = "installed (version unknown)"
            
            print(f"TA version: {version}")
            
            # Simple test of functionality
            import pandas as pd
            import numpy as np
            
            # Create a simple DataFrame
            df = pd.DataFrame({
                'close': np.random.random(100) * 100
            })
            
            # Try to calculate RSI
            rsi = ta.momentum.RSIIndicator(df['close']).rsi()
            print(f"RSI calculation test: {'Successful' if not rsi.empty else 'Failed'}")
            
        except Exception as e:
            print(f"Error verifying TA installation: {e}")

if __name__ == "__main__":
    check_and_install_packages()
