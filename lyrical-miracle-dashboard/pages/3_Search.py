import random

import streamlit as st

from vector_search import text_search, transform_search_res
from common import timeout_popup

MAX_SEARCH_RESULTS = 5
SAMPLE_SEARCHES = [
    'The one that goes beep boo boo bop boo boo beep',
    'The one where Drake drunk dials his ex',
    'J. Cole writes a letter to his daughter',
    'Bad Bunny reminisces about growing up in Puerto Rico',
    'Sad song about someone you used to love',
    "I don't know if I can trust anyone",
    'Life is really stressing me out right now... I hope the struggle is worth it in the end',
    "I feel like I'm meant for something more... like I have a greater purpose",
    "I have a crush on someone but I'm too scared to approach",
    'Inspiring song about finally reaching success',
    "Let's have fun tonight!",
]


timeout_popup()
st.set_page_config(page_title="What's that song...?", page_icon='🤔')
st.header("What's that song...? 🤔")
st.sidebar.header('The Lyrical Miracle')
st.markdown("""Search for that one song that's on the tip of your tongue.
            Or find songs that match your mood right now.""")

text_input = st.text_input(
    label='',
    value=random.sample(SAMPLE_SEARCHES, 1)[0],
    max_chars=1000,
    key='lyric_search_text',
)

if text_input:
    with st.spinner('Searching...'):
        search_res = text_search(text_input, MAX_SEARCH_RESULTS)
        st.dataframe(
            transform_search_res(search_res).rename(
                lambda s: s.replace('_', ' ').title()
            ),
            column_config={
                'Url': st.column_config.LinkColumn('Lyrics', display_text='genius.com')
            },
        )
