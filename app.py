"""Run from the project root: python -m streamlit run app.py."""

import streamlit as st

from medcnn.management_ui import render


st.set_page_config(page_title="MedCNN Inspector", page_icon="🔬", layout="wide")
render()
