import streamlit as st

from common import timeout_popup


timeout_popup()
st.set_page_config(
    page_title='The Lyrical Miracle',
    page_icon='🔊',
)

st.title('The Lyrical Miracle')
st.subheader('_Your music listening history, with AI powered lyrical analysis_')
st.sidebar.success('Check out one of these pages!')


BODY_TEXT = """
Check out the sidebar to see month-by-month breakdowns or to see your whole timeline.
"""
st.markdown(BODY_TEXT)
