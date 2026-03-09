import streamlit as st

st.set_page_config(
    page_title='The Lyrical Miracle',
    page_icon='👋',
)

st.title('The Lyrical Miracle')
st.subheader('_Your music listening history, with AI powered lyrical analysis_')
st.sidebar.success('Click here!')


BODY_TEXT = """
Check out the sidebar to see month-by-month breakdowns or to see your whole timeline.
"""
st.markdown(BODY_TEXT)
