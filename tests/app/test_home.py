"""AppTest smoke tests for the Home page"""


def test_home_page_renders(app_harness):
    at = app_harness.run_page('Home.py')

    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == ['The Lyrical Miracle']
    assert len(at.markdown) == 1


def test_home_page_includes_timeout_popup(app_harness):
    app_harness.run_page('Home.py')

    app_harness.timeout_popup.assert_called_once()
