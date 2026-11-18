from pathlib import Path

import pytest


def _widget(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_generic_workbench_pages_and_real_actions():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[2]
    if not (root / "medcnn_runs/resnet18_full/model.pt").exists():
        pytest.skip("Local trained example checkpoint unavailable")
    script = root / "medcnn_app.py"
    if not script.exists():
        script = root / "app.py"
    app = AppTest.from_file(script, default_timeout=120).run()
    selector = _widget(app.selectbox, "Model")
    selector.select_index(next(i for i, name in enumerate(selector.options)
                               if "resnet18" in name.lower().replace("-", "").replace(" ", ""))).run()
    assert not app.exception

    for page, action in (("Feature Discovery", "Extract activations"),
                         ("Activation Patching", "Patch and compare"),
                         ("Feature Interaction Lab", "Measure individual and paired effects")):
        _widget(app.sidebar.radio, "Analysis").set_value(page).run()
        assert not app.exception, page
        _widget(app.button, action).click().run()
        assert not app.exception, (page, action)

    _widget(app.sidebar.radio, "Analysis").set_value("Controlled Dataset Experiments").run()
    _widget(app.slider, "Stratified test images").set_value(16).run()
    _widget(app.multiselect, "Channel budgets").set_value([1]).run()
    _widget(app.multiselect, "Replacement methods").set_value(["zero"]).run()
    _widget(app.button, "Run controlled experiment").click().run()
    assert not app.exception
    assert len(app.dataframe) >= 1
