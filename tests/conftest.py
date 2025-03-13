import pytest
import sys
import os
from typing import Set
import warnings

# Silence warnings
warnings.simplefilter('ignore')

# Add root dir to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Preserve specific modules
PRESERVE_MODULES = {
    'os', 'sys', 'pytest', 'pathlib',
    '_pytest', 'importlib', 'typing',
    'collections', 'abc', 'warnings'
}

def should_skip_rewrite(name: str, preserve: Set[str] = PRESERVE_MODULES) -> bool:
    """Check if module should skip rewrite"""
    return (
        any(name.startswith(p) for p in preserve) or
        'site-packages' in name or
        'importlib' in name
    )

@pytest.hookimpl(hookwrapper=True)
def pytest_collect_file(path, parent):
    """Custom hook to prevent rewriting certain modules"""
    outcome = yield
    if should_skip_rewrite(str(path), PRESERVE_MODULES):
        item = outcome.get_result()
        if item is not None:
            item.session._fixturemanager._arg2fixturedefs = {}

@pytest.fixture(autouse=True)
def setup_test():
    """Setup test environment with clean imports"""
    # Store original modules
    orig_modules = set(sys.modules.keys())
    
    yield
    
    # Remove any new modules
    current_modules = set(sys.modules.keys())
    for mod in current_modules - orig_modules:
        if not mod.startswith(('pytest', '_pytest', 'pluggy', 'py')):
            sys.modules.pop(mod, None)
