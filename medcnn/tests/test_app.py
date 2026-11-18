from pathlib import Path

import pytest


def _widget(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_unified_workspace_and_both_examples():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[2]
    script = root / "medcnn_app.py"
    if not script.exists():
        script = root / "app.py"
    app = AppTest.from_file(script, default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "MedCNN Inspector"
    assert [tab.label for tab in app.tabs] == ["Workspace", "Add Model", "Import Dataset", "Manage"]
    assert _widget(app.selectbox, "2D CNN architecture or trusted local factory")
    assert _widget(app.selectbox, "Dataset format")

    for architecture in ("resnet18", "densenet121"):
        if not (root / "medcnn_runs" / f"{architecture}_full" / "model.pt").exists():
            pytest.skip("Full local checkpoints are not available")
        model_selector = _widget(app.selectbox, "Model")
        index = next(i for i, name in enumerate(model_selector.options)
                     if architecture in name.lower().replace("-", "").replace(" ", ""))
        model_selector.select_index(index).run()
        assert not app.exception, architecture
        assert any("Ready:" in item.value for item in app.success)
        assert _widget(app.sidebar.radio, "Analysis")
        for page in ("Model Inspector", "Feature Discovery", "Activation Patching",
                     "Concept Explorer", "Feature Interaction Lab", "Dataset Experiments",
                     "Controlled Dataset Experiments"):
            _widget(app.sidebar.radio, "Analysis").set_value(page).run()
            assert not app.exception, (architecture, page)
