# gpu_test.py
import torch
import streamlit as st

st.title("🧠 GPU Status Check")

if torch.cuda.is_available():
    st.success(f"✅ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    st.error("❌ CUDA is NOT available. Running on CPU.")
