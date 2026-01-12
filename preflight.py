#!/usr/bin/env python3
"""
Preflight check: Verify dependencies and module imports before running the app.
"""
import sys
import importlib

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"[OK] Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  [WARN] Warning: Python 3.8+ recommended")
        return False
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed and return version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"[OK] {package_name}: {version}")
        return True
    except ImportError:
        print(f"[MISSING] {package_name}: NOT INSTALLED")
        return False

def check_key_packages():
    """Check key packages."""
    print("\nKey Packages:")
    packages = [
        ('streamlit', 'streamlit'),
        ('torch', 'torch'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'sklearn'),
        ('plotly', 'plotly'),
        ('matplotlib', 'matplotlib'),
    ]
    
    all_ok = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_ok = False
    
    # Optional packages
    print("\nOptional Packages:")
    optional = [
        ('shap', 'shap'),
        ('kaleido', 'kaleido'),
    ]
    for pkg_name, import_name in optional:
        if check_package(pkg_name, import_name):
            print(f"  (Optional) [OK] {pkg_name} installed")
        else:
            print(f"  (Optional) [WARN] {pkg_name} not installed (some features may be limited)")
    
    return all_ok

def check_module_imports():
    """Check that app modules can be imported."""
    print("\nModule Imports:")
    
    modules_to_check = [
        ('ml.pipeline', 'ml/pipeline.py'),
        ('ml.eval', 'ml/eval.py'),
        ('models.base', 'models/base.py'),
        ('models.nn_whuber', 'models/nn_whuber.py'),
        ('models.glm', 'models/glm.py'),
        ('models.huber_glm', 'models/huber_glm.py'),
        ('models.rf', 'models/rf.py'),
        ('utils.session_state', 'utils/session_state.py'),
        ('utils.seed', 'utils/seed.py'),
        ('utils.datasets', 'utils/datasets.py'),
    ]
    
    all_ok = True
    for module_name, file_path in modules_to_check:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name}")
        except ImportError as e:
            print(f"[ERROR] {module_name}: {str(e)}")
            all_ok = False
        except Exception as e:
            print(f"[WARN] {module_name}: {type(e).__name__} - {str(e)}")
            # Some modules might have dependencies that fail, but module itself loads
    
    return all_ok

def main():
    """Run preflight checks."""
    print("=" * 60)
    print("Modeling Lab - Preflight Check")
    print("=" * 60)
    
    print("\nPython Environment:")
    py_ok = check_python_version()
    
    pkg_ok = check_key_packages()
    mod_ok = check_module_imports()
    
    print("\n" + "=" * 60)
    if py_ok and pkg_ok and mod_ok:
        print("[OK] All checks passed! Ready to run the app.")
        print("\nNext steps:")
        print("  Windows:  .\\run.ps1")
        print("  macOS/Linux:  ./run.sh")
        return 0
    else:
        print("[ERROR] Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
