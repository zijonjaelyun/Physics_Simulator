"""Browser validation of the real HTML/JS replay (not the Streamlit server)."""
from pathlib import Path
import json, math, sys, shutil
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from physics import Settings, simulate
from visuals import replay_html,snapshot_html

def main():
    report={"scope":"Actual Chromium HTML components; not Streamlit server", "checks":[]}
    with sync_playwright() as pw:
        executable=shutil.which("chromium") or shutil.which("chromium-browser")
        browser=pw.chromium.launch(executable_path=executable,
                                   headless=True,args=["--no-sandbox"])
        page=browser.new_page(viewport={"width":1150,"height":1000},device_scale_factor=1)
        errors=[]
        page.on("pageerror",lambda e:errors.append(str(e)))
        for name,config in [
            ("same",Settings()),("cu10",Settings(cu_M=1.0)),
            ("zn10",Settings(zn_M=1.0)),("open",Settings(protocol="open")),
            ("temp",Settings(cu_M=.01,temperature_C=45)),
        ]:
            page.goto("about:blank")
            page.set_content(snapshot_html(config))
            page.wait_for_function("!!window.__labTest")
            s=page.evaluate("window.__labTest.getState()")
            assert math.isfinite(s["surface_nernst_V"])
            assert s["current_A"]==0 if name=="open" else s["current_A"]>0
            assert not page.locator("#formula").is_visible()
            report["checks"].append("snapshot:"+name)
        page.goto("about:blank")
        page.set_content(snapshot_html(Settings(cu_M=.1,zn_M=1.0)))
        page.wait_for_timeout(150)
        page.screenshot(path=str(ROOT/"preview_initial.png"),full_page=True)
        result=simulate(Settings(protocol="rest"))
        page.goto("about:blank")
        page.set_content(replay_html(result))
        page.wait_for_function("!!window.__labTest")
        assert page.locator("#formula").is_visible()
        page.evaluate("window.__labTest.seek(600)")
        s=page.evaluate("window.__labTest.getState()")
        assert abs(float(page.locator("#es").inner_text())-s["surface_nernst_V"])<6e-6
        assert 594 <= s["time_s"] <= 600
        report["checks"].append("equation-sync")
        page.screenshot(path=str(ROOT/"preview_replay.png"),full_page=True)
        page.evaluate("window.__labTest.seek(1800)")
        s=page.evaluate("window.__labTest.getState()")
        assert s["current_A"]==0
        report["checks"].append("rest-zero-current")
        page.select_option("#model","mixed")
        s=page.evaluate("window.__labTest.getState()")
        assert 1794 <= s["time_s"] <= 1800
        assert abs(s["cu_average_M"]-s["cu_surface_M"])<1e-13
        report["checks"].append("model-switch-at-same-time")
        page.click("#reset")
        assert page.evaluate("window.__labTest.getState().time_s")==0
        page.click("#play")
        page.wait_for_timeout(500)
        assert page.evaluate("window.__labTest.getState().time_s")>0
        page.click("#play")
        t=page.evaluate("window.__labTest.getState().time_s")
        page.wait_for_timeout(200)
        assert page.evaluate("window.__labTest.getState().time_s")==t
        report["checks"].append("play-pause-reset")
        page.locator("#seek").evaluate("(e)=>{e.value=2400;e.dispatchEvent(new Event('input'))}")
        assert page.evaluate("window.__labTest.getState().time_s")>2300
        report["checks"].append("slider-seek")
        page.set_viewport_size({"width":390,"height":844})
        page.wait_for_timeout(100)
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth+1")
        page.screenshot(path=str(ROOT/"preview_mobile.png"),full_page=True)
        report["checks"].append("mobile-no-horizontal-overflow")
        assert not errors, errors
        report["js_errors"]=errors
        browser.close()
    (ROOT/"browser_validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
