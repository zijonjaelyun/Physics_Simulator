"""Actual Streamlit AppTest tests; never replace Streamlit with a mock here.

These tests run after `pip install -r requirements-dev.txt`. If Streamlit is
unavailable, skip rather than claiming the UI was integration-tested.
"""
from pathlib import Path
import importlib.util
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
pytestmark=pytest.mark.skipif(importlib.util.find_spec("streamlit") is None,
    reason="Streamlit package unavailable; actual AppTest cannot run")

def make_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT/"app.py"),default_timeout=60).run()

def test_initial_screen_equation_and_input():
    app=make_app()
    assert not app.exception
    assert len(app.latex)>=2
    app.slider(key="cu0").set_value(.01).run()
    assert not app.exception
    assert any("1.07042" in str(x.value) for x in app.latex)

def test_preset_and_temperature():
    app=make_app()
    app.selectbox(key="preset").select("구리 이온 농도 10배").run()
    assert not app.exception
    assert app.session_state["cu0"]==1.0
    app.slider(key="temp").set_value(45.0).run()
    assert not app.exception

@pytest.mark.parametrize("page",[
    "02 · 방전과 이온 이동","03 · 심화 탐구","04 · 검증과 모델"])
def test_each_page(page):
    app=make_app()
    app.radio(key="page").set_value(page).run()
    assert not app.exception

@pytest.mark.parametrize("question",[
    "B. 같은 초기 전압이면 반응 가능한 양도 같을까?",
    "C. 이온 공급과 부하가 결과를 얼마나 바꿀까?",
    "D. 개별 입자의 무작위 운동은 확산과 어떻게 연결될까?",
])
def test_research_branches(question):
    app=make_app()
    app.radio(key="page").set_value("03 · 심화 탐구").run()
    app.selectbox(key="research_question").select(question).run()
    assert not app.exception

def test_sweep_action():
    app=make_app()
    app.radio(key="page").set_value("03 · 심화 탐구").run()
    app.selectbox(key="research_question").select(
        "C. 이온 공급과 부하가 결과를 얼마나 바꿀까?").run()
    app.button(key="sweep_run").click().run()
    assert not app.exception

def test_convergence_action():
    app=make_app()
    app.radio(key="page").set_value("04 · 검증과 모델").run()
    app.button(key="converge").click().run()
    assert not app.exception
