#!/bin/bash
# Streamlit startup script with protobuf workaround
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

# Run the dashboard
python -m streamlit run "$@"
