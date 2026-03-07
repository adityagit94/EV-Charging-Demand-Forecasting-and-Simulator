# Protobuf Compatibility Fix

The project had a compatibility issue between Streamlit and the protobuf package.
This is now fixed by pinning `protobuf==3.20.3` in both `requirements.txt` and `pyproject.toml`.

## If you still see the error in a deployment environment:

### Option 1: Clean reinstall (Recommended)
In your venv, run:
```bash
pip uninstall protobuf -y
pip install protobuf==3.20.3
pip install --upgrade streamlit
```

### Option 2: Use the startup script
Instead of running `streamlit run app.py` directly, use:
```bash
./run_streamlit.sh dashboard/app.py
```

This script sets the necessary environment variable as a workaround.

### Option 3: Set environment variable (if reinstall doesn't work)
If all else fails, set this before running Streamlit:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run dashboard/app.py
```

Note: Option 3 is slower but will work as a last resort.

## For Codespaces/Cloud deployments:
1. Check that your `requirements.txt` includes `protobuf==3.20.3`
2. Rebuild/restart your environment to force a fresh dependency installation
3. The pinned version should then be installed automatically
