"""AppTest tests for the Timelines page"""


def test_timelines_page_renders(app_harness):
    at = app_harness.run_page('pages/2_Timelines.py')

    assert not at.exception, [e.value for e in at.exception]
    app_harness.timeout_popup.assert_called_once()
    assert 'Your Listening History' in [h.value for h in at.header]
    assert at.selectbox[0].value == 0
    assert len(at.get('plotly_chart')) == 3


def test_timelines_diversity_selection(app_harness):
    at = app_harness.run_page('pages/2_Timelines.py')

    at.selectbox[0].select_index(2)
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state['diversity_type_ix'] == 2
