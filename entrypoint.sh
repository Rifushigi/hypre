#!/bin/bash

# Start FastAPI backend (internal only)
uvicorn main:app --host 0.0.0.0 --port 8000 &
 
# Start Streamlit UI (public)
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 