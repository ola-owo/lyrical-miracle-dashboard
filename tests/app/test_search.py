"""AppTest tests for the lyric search page"""


def test_search_page_renders_results(app_harness):
    at = app_harness.run_page('pages/3_Search.py')

    assert not at.exception, [e.value for e in at.exception]
    app_harness.timeout_popup.assert_called_once()
    assert app_harness.text_search.called
    assert at.dataframe[0].value.columns.tolist() == [
        'Song',
        'Artist',
        'Album',
        'Release Date',
        'Url',
    ]


def test_search_page_uses_new_query(app_harness):
    at = app_harness.run_page('pages/3_Search.py')

    at.text_input[0].set_value('beep boop').run()

    assert not at.exception, [e.value for e in at.exception]
    assert app_harness.text_search.call_args.args[0] == 'beep boop'
    assert app_harness.text_search.call_args.args[1] == 5
