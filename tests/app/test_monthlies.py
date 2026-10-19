"""AppTest tests for the Monthlies page"""


def test_monthlies_page_renders(app_harness):
    at = app_harness.run_page('pages/1_Monthlies.py')

    assert not at.exception, [e.value for e in at.exception]
    app_harness.timeout_popup.assert_called_once()
    # no month selected by default
    assert at.sidebar.selectbox[0].value is None
    assert 'Clustered listening sessions' in [h.value for h in at.header]
    assert len(at.get('plotly_chart')) == 3
    assert len(at.dataframe) == 1


def test_monthlies_select_month(app_harness):
    at = app_harness.run_page('pages/1_Monthlies.py')

    # options are newest-first, so index 1 is January 2024
    at.sidebar.selectbox[0].select_index(1)
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state['selected_date'].month == 1
    assert len(at.dataframe) == 1
