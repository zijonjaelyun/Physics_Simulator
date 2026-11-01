"""Nernst Lab — Streamlit entry point.

Run from this folder: python -m streamlit run app.py
Only four files are required to deploy: app.py, physics.py, visuals.py,
requirements.txt. The remaining files provide documentation and tests.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from physics import (
    Settings, nernst, nernst_terms, capacity_mAh, simulate, NF, F, RGAS,
    E_STANDARD, brownian_experiment, convergence_experiment,
)
from visuals import snapshot_html, replay_html, timeseries_frame, profile_frame

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Nernst Lab | 네른스트 전지 실험실",
                   page_icon="🔋", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container{padding-top:1.55rem;padding-bottom:3rem;max-width:1480px}
h1,h2,h3{letter-spacing:-.045em}
h1{font-size:2.15rem!important}h2{font-size:1.5rem!important}

[data-testid="stSidebar"] h2{font-size:1.18rem!important}
[data-testid="stMetric"]{background:var(--secondary-background-color);border:1px solid #dce5ef;border-radius:11px;padding:12px 15px}
[data-testid="stMetricLabel"]{font-size:.8rem}
[data-testid="stMetricValue"]{font-size:1.65rem}
.eyebrow{font-size:11px;font-weight:750;letter-spacing:2.4px;color:#287496;margin:0 0 4px}
.hero-copy{font-size:15px;line-height:1.8;color:#597187;margin:0 0 18px}
.kicker{display:inline-block;padding:5px 10px;border-radius:20px;background:#e7f2f7;color:#28647d;font-size:12px;margin-bottom:9px}
.formula-note{padding:12px 16px;background:#eff6fb;border-left:3px solid #257b9d;border-radius:5px;color:#3b5870;line-height:1.8;font-size:13px}
[data-testid="stLatex"]{overflow-x:auto}
.stDownloadButton button{width:100%}
</style>""", unsafe_allow_html=True)

PAGES = ["01 · 네른스트 식 조절", "02 · 방전과 이온 이동", "03 · 심화 탐구", "04 · 검증과 모델"]
PROTOCOLS = {"연속 방전": "continuous", "중간 휴지 후 재방전": "rest", "개방 회로": "open"}
DEFAULT = dict(cu0=0.100, zn0=0.100, temp=25.0, ext_r=1000.0,
               protocol_label="연속 방전", minutes=60.0, vcu=0.50, vzn=0.50,
               area=1.0, film=0.50, d9=0.70, internal_r=50.0, dt=2.0, cells=48)
PRESETS = {
    "기본 · 같은 농도": {},
    "구리 이온 농도 10배": {"cu0":1.0, "zn0":0.1},
    "아연 이온 농도 10배": {"cu0":0.1, "zn0":1.0},
    "확산 제한 관찰": {"ext_r":100.0, "minutes":10.0, "dt":0.5},
    "휴지 · 전압 회복": {"protocol_label":"중간 휴지 후 재방전"},
}
for key, val in DEFAULT.items():
    st.session_state.setdefault(key, val)

def apply_preset():
    for key, val in (DEFAULT | PRESETS[st.session_state.preset]).items():
        st.session_state[key] = val

@st.cache_data(show_spinner=False, max_entries=12)
def run_cached(config: dict):
    return simulate(Settings(**config))

@st.cache_data(show_spinner=False, max_entries=8)
def random_walk_cached(d: float, seed: int):
    return brownian_experiment(d, duration=60, particles=2400, steps=300, seed=seed)

def figure(title: str, x: str, y: str, height: int = 340):
    f = go.Figure()
    f.update_layout(title=dict(text=title, font=dict(size=16)), xaxis_title=x, yaxis_title=y,
        height=height, template="plotly_white", margin=dict(l=54,r=20,t=58,b=50),
        font=dict(family="Arial, Malgun Gothic, sans-serif",color="#29465f"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.15, x=0), hovermode="x unified")
    f.update_xaxes(gridcolor="#edf1f6")
    f.update_yaxes(gridcolor="#edf1f6")
    return f

def show_plot(f, key):
    st.plotly_chart(f, use_container_width=True, key=key,
                    config={"displaylogo":False, "responsive":True})

def show_sources():
    st.markdown("""
**수식과 모델의 근거**
- [IUPAC · 네른스트 방정식](https://goldbook.iupac.org/terms/view/09068): 활동도, 평형 전위, 온도별 로그 계수.
- [OpenStax · 전극과 전지 전위](https://openstax.org/books/chemistry-2e/pages/17-3-electrode-and-cell-potentials): 표준 전위와 전지 반응.
- [OpenStax · 전하량과 반응량](https://openstax.org/books/chemistry-2e/pages/17-7-electrolysis): 패러데이 법칙의 화학량론.
- [IUPAC · 평형 전극 전위](https://goldbook.iupac.org/terms/view/09064): 전류가 없는 평형 상태와 전극 전위.
- [COMSOL · Nernst–Planck 수송](https://doc.comsol.com/6.3/doc/com.comsol.help.fce/fce_ug_electrochem.07.086.html): 확산·전기장 이동·대류의 구분.
- [COMSOL · 전기화학 모델의 가정](https://doc.comsol.com/6.3/doc/com.comsol.help.echem/echem_introduction.02.02.html): 지지전해질과 확산 중심 모델.
- [Streamlit · 배포 파일 구조](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).
""")

with st.sidebar:
    st.markdown("## 🔋 Nernst Lab")
    st.caption("식의 값 → 전지 상태 → 이온 이동")
    page = st.radio("실험 화면", PAGES, key="page")
    st.divider()
    st.selectbox("빠른 시작 조건", list(PRESETS), key="preset", on_change=apply_preset)
    st.markdown("### 네른스트 식의 입력값")
    st.slider("초기 Cu²⁺ 농도 (mol/L)", 0.001, 1.000, step=0.001, format="%.3f", key="cu0")
    st.slider("초기 Zn²⁺ 농도 (mol/L)", 0.001, 1.000, step=0.001, format="%.3f", key="zn0")
    st.slider("온도 (°C)", 5.0, 45.0, step=1.0, key="temp")
    st.caption("입력을 바꾸면 식과 초기 전지 그림이 즉시 갱신됩니다.")
    st.markdown("### 부하와 시간")
    st.number_input("외부 저항 R_ext (Ω)", 10.0, 100000.0, step=100.0, key="ext_r")
    st.selectbox("회로 연결 방식", list(PROTOCOLS), key="protocol_label")
    st.number_input("관찰 시간 (분)", 1.0, 180.0, step=5.0, key="minutes")
    with st.expander("부피 · 전극 · 확산 · 수치 설정"):
        st.number_input("구리 용액 총부피 (mL)", 0.20, 50.0, step=0.10, format="%.2f", key="vcu")
        st.number_input("아연 용액 부피 (mL)", 0.20, 50.0, step=0.10, format="%.2f", key="vzn")
        st.number_input("구리 전극 면적 (cm²)", 0.20, 5.0, step=0.10, key="area")
        st.number_input("확산층 두께 δ (mm)", 0.10, 2.0, step=0.10, key="film")
        st.number_input("확산계수 D (×10⁻⁹ m²/s)", 0.10, 3.0, step=0.10, format="%.2f", key="d9")
        st.number_input("집중 내부 저항 r (Ω)", 0.0, 1000.0, step=10.0, key="internal_r")
        st.select_slider("시간 간격 Δt (s)", [0.5, 1.0, 2.0, 5.0], key="dt")
        st.selectbox("확산층 격자 수", [24, 48, 96], key="cells")
        st.caption("D·r·기하 값은 교육용 가정값입니다. 표면 농도 하한은 10⁻⁸ M입니다.")
    st.caption("E°=1.10 V, n=2 고정 · 활동도≈농도. 실제 전지의 정량 예측·안전 판정용이 아닙니다.")

s = Settings(cu_M=st.session_state.cu0, zn_M=st.session_state.zn0,
    temperature_C=st.session_state.temp, resistance_ohm=st.session_state.ext_r,
    protocol=PROTOCOLS[st.session_state.protocol_label],
    duration_s=st.session_state.minutes*60, volume_cu_ml=st.session_state.vcu,
    volume_zn_ml=st.session_state.vzn, area_cm2=st.session_state.area,
    film_mm=st.session_state.film, diffusion_m2_s=st.session_state.d9*1e-9,
    internal_ohm=st.session_state.internal_r, dt_s=st.session_state.dt,
    cells=st.session_state.cells)
try:
    s.validate()
except ValueError as exc:
    st.error("조건을 확인해 주세요. 확산층 부피(A×δ)는 구리 용액 총부피보다 작아야 합니다. 계산 단계는 24,000개 이하여야 합니다.")
    st.code(str(exc))
    st.stop()

st.markdown('<div class="eyebrow">PHYSICS SIMULATION · ELECTROCHEMISTRY</div>', unsafe_allow_html=True)
st.title("네른스트 전지 실험실")
st.markdown('<div class="hero-copy">농도비가 전압을 결정한다면, 이온이 이동하는 속도는 무엇을 바꿀까?<br>'
            '네른스트 식의 각 항을 직접 조절하고, 다니엘 전지의 평형 전압과 실제 부하 전압을 비교합니다.</div>',
            unsafe_allow_html=True)
terms = nernst_terms(s.cu_M, s.zn_M, s.temperature_C)
if s.temperature_C != 25:
    st.info("온도 실험의 범위: T는 RT/(2F)에만 반영합니다. E°=1.10 V, 확산계수 D, 내부 저항 r은 고정합니다. 실제 전지의 전체 온도 의존성을 예측하는 모델이 아닙니다.")

def equation_panel():
    with st.container(border=True):
        st.markdown("#### 네른스트 방정식 · 현재 초기 조건")
        st.latex(r"E_{\mathrm{eq}}=E^\circ-\frac{R_{\mathrm{g}}T}{nF}\ln Q"
                 r"\quad Q=\frac{a_{\mathrm{Zn}^{2+}}}{a_{\mathrm{Cu}^{2+}}}"
                 r"\approx\frac{[\mathrm{Zn}^{2+}]}{[\mathrm{Cu}^{2+}]},\quad n=2")
        st.latex(fr"E_{{\mathrm{{eq}}}}=1.10000-\underbrace{{{terms['k_V_per_decade']:.5f}}}_{{2.303R_{{\mathrm{{g}}}}T/(2F)}}"
                 fr"\log_{{10}}\!\left(\frac{{\color{{#7661a6}}{{{s.zn_M:.4f}}}}}{{\color{{#1389ac}}{{{s.cu_M:.4f}}}}}\right)"
                 fr"=\boxed{{{terms['E_V']:.5f}\ \mathrm{{V}}}}")
        a,b,c,d = st.columns(4)
        a.metric("반응 지수 Q · 무차원", f"{terms['Q']:.4g}")
        b.metric("log₁₀ Q", f"{terms['log10_Q']:+.5f}")
        c.metric("농도에 의한 보정 ΔE", f"{terms['concentration_shift_V']*1000:+.3f} mV")
        d.metric("평형 전압 E_eq", f"{terms['E_V']:.5f} V")
        st.caption(f"T={terms['temperature_K']:.2f} K · 기체상수 R_g={RGAS:.6f} J/(mol·K) · F={F:.5f} C/mol · "
                   "순수 고체 Zn·Cu의 활동도는 1. 외부 저항 R_ext는 기체상수 R_g와 다른 양입니다.")
        st.caption("25 °C에서는 0.05916/2≈0.02958 V/decade입니다. 0.0592/2는 반올림한 교과용 계수입니다.")

def obtain_result():
    try:
        with st.spinner("농도·회로·확산을 연결하여 계산 중입니다."):
            return run_cached(asdict(s))
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        st.error(f"계산하지 못했습니다: {exc}")
        st.stop()

def status_notice(result):
    if result["status"] != "duration":
        st.warning(f"{result['actual_duration_s']:.4g}초에서 표면 농도 하한에 도달해 두 모델을 함께 종료했습니다. "
                   "이는 0 V 도달이나 모든 구리 이온의 소진이 아닙니다. 더 큰 외부 저항 또는 더 빠른 이온 공급 조건으로 비교해 보세요.")
    if s.protocol == "rest":
        st.caption(f"휴지 구간: {s.duration_s*.35:.0f}–{s.duration_s*.65:.0f} s. 이때 I=0이며, 농도는 확산으로 계속 변할 수 있습니다.")

if page == PAGES[0]:
    equation_panel()
    if terms["Q"] > 1:
        explanation = "Q > 1 → log₁₀Q > 0 → 농도 보정이 음수 → E_eq < E°"
    elif terms["Q"] < 1:
        explanation = "Q < 1 → log₁₀Q < 0 → 농도 보정이 양수 → E_eq > E°"
    else:
        explanation = "Q = 1 → log₁₀Q = 0 → E_eq = E°. 두 농도가 꼭 1 M일 필요는 없습니다."
    st.markdown(f'<div class="formula-note">{explanation}</div>', unsafe_allow_html=True)
    st.write("")
    components.html(snapshot_html(s), height=630, scrolling=True)

    left,right = st.columns([1.25,1])
    with left:
        f=figure("농도비를 로그로 보면 직선이 됩니다", "log₁₀ Q", "평형 전압 E_eq (V)")
        x=np.linspace(-3,3,181)
        for temp in sorted(set([5.0,25.0,45.0,s.temperature_C])):
            k=nernst_terms(.1,.1,temp)["k_V_per_decade"]
            f.add_trace(go.Scatter(x=x,y=E_STANDARD-k*x,name=f"{temp:g} °C",
                mode="lines",line=dict(width=2.6 if temp==s.temperature_C else 1.2,
                                      dash="solid" if temp==s.temperature_C else "dot")))
        f.add_trace(go.Scatter(x=[terms["log10_Q"]],y=[terms["E_V"]],
            name="현재 입력",mode="markers",marker=dict(size=12,symbol="diamond",color="#176b92")))
        show_plot(f,"nernst_curve")
        st.caption("이 그래프에서 온도 효과는 네른스트 항의 기울기 변화만을 뜻합니다.")
    with right:
        st.markdown("#### 무엇을 바꾸면 어떻게 변할까?")
        st.dataframe(pd.DataFrame([
            ["Cu²⁺ 농도만 10배",f"+{terms['k_V_per_decade']*1000:.3f} mV"],
            ["Zn²⁺ 농도만 10배",f"−{terms['k_V_per_decade']*1000:.3f} mV"],
            ["두 농도를 함께 10배","평형 전압 변화 없음"],
            ["외부 저항만 변경","초기 평형 전압 불변 · 전류는 변함"],
        ],columns=["조작", "이상 모델의 예측"]),hide_index=True,use_container_width=True)
        st.info("식의 E_eq는 평형 전압입니다. 부하가 연결되면 단자 전압은 내부 저항과 전극 표면의 농도 변화에도 영향을 받습니다.")
        st.latex(r"I=\frac{E_{\mathrm{eq}}}{R_{\mathrm{ext}}+r},\qquad V_{\mathrm{term}}=IR_{\mathrm{ext}}")
        st.caption("위 회로식은 초기의 균일 농도 상태에 대한 단순화입니다. 방전 중에는 표면 농도로 네른스트 값을 다시 계산합니다.")

    with st.expander("왜 n=2와 E°는 자유 슬라이더가 아닌가?"):
        st.latex(r"\mathrm{Zn(s)\rightarrow Zn^{2+}(aq)+2e^-}")
        st.latex(r"\mathrm{Cu^{2+}(aq)+2e^-\rightarrow Cu(s)}")
        st.write("이 앱은 Zn–Cu 반응이므로 전자 수 n=2가 고정됩니다. n을 임의로 바꾸면 다른 반응을 계산하게 됩니다. "
                 "E°=1.10 V는 교육용 기준값입니다. 농도·외부 저항과 달리 마음대로 정하는 상태 변수가 아닙니다.")
        st.write("온도가 바뀌면 실제 E°도 달라질 수 있지만, 여기서는 네른스트 항만 분리해서 탐구하도록 E°를 고정했습니다.")
    with st.expander("근거와 적용 조건"):
        st.write("활동도를 농도/c°로 근사합니다(c°=1 mol/L). 농도비에서는 c°가 약분됩니다. "
                 "특히 높은 농도나 지지전해질이 많은 실제 용액에서는 이 근사가 부정확할 수 있습니다.")
        show_sources()

elif page == PAGES[1]:
    st.markdown("### 네른스트 식을 시간에 따라 다시 계산하기")
    st.latex(r"E_s(t)=E^\circ-\frac{R_{\mathrm{g}}T}{2F}\ln\frac{[\mathrm{Zn}^{2+}](t)}{[\mathrm{Cu}^{2+}]_s(t)}"
             r"\quad\longrightarrow\quad V_{\mathrm{term}}=E_s-Ir")
    st.caption("재생 화면 안의 값 대입 식은 현재 시각과 모델 선택에 맞춰 바뀝니다. "
               "E_s는 국소 가역 계면 가정으로 얻은 전위차이며, 농도 구배가 있는 전지 전체의 열역학적 평형 전압이라는 뜻은 아닙니다.")
    result=obtain_result()
    status_notice(result)
    components.html(replay_html(result), height=1200, scrolling=True)
    d=result["diffusion"]
    st.markdown("### 전압 감소를 분리해서 보기")
    st.latex(r"V_{\mathrm{term}}=E_{\mathrm{avg}}"
             r"-\underbrace{(E_{\mathrm{avg}}-E_s)}_{\text{농도 불균일 효과}}"
             r"-\underbrace{Ir}_{\text{저항 손실}}")
    l,r=st.columns(2)
    with l:
        f=figure("평균 · 표면 · 단자: 세 값은 다릅니다","시간 (s)","전압 (V)")
        for field,label,color in [("bulk_nernst_V","평균 농도 E_avg","#8498ad"),
                                  ("surface_nernst_V","표면 농도 E_s","#1687aa"),
                                  ("terminal_V","단자 전압 V_term","#c78646")]:
            f.add_trace(go.Scatter(x=d["time_s"],y=d[field],name=label,mode="lines",line=dict(color=color)))
        show_plot(f,"voltage_breakdown")
    with r:
        f=figure("이온 공급을 무시한 경우와 고려한 경우","시간 (s)","전류 (mA)")
        for key,label,dash in [("mixed","균일 혼합","dash"),("diffusion","확산 고려","solid")]:
            f.add_trace(go.Scatter(x=result[key]["time_s"],y=result[key]["current_A"]*1000,
                       name=label,mode="lines",line=dict(dash=dash)))
        show_plot(f,"current_models")
    summary=[]
    for key,label in [("mixed","균일 혼합"),("diffusion","확산 고려")]:
        z=result[key]
        summary.append({"모델":label, "비교 종료 (s)":result["actual_duration_s"],
            "누적 전하 (mAh)":z["charge_C"][-1]/3.6, "최종 E_avg (V)":z["bulk_nernst_V"][-1],
            "최종 E_s (V)":z["surface_nernst_V"][-1], "최종 단자 전압 (V)":z["terminal_V"][-1],
            "표면 Cu²⁺ (M)":z["cu_surface_M"][-1]})
    st.dataframe(pd.DataFrame(summary),hide_index=True,use_container_width=True)
    st.caption(f"초기 구리 이온의 화학량론적 전하량 상한: {result['theoretical_capacity_mAh']:.4f} mAh. "
               "이는 실제로 사용할 수 있는 전지 용량이나 방전 완료 기준이 아닙니다.")
    with st.expander("현재 실험의 전체 데이터를 확인 · 저장"):
        frame=timeseries_frame(result)
        st.dataframe(frame.iloc[::max(1,len(frame)//150)],use_container_width=True,hide_index=True)
        a,b,c,dcol=st.columns(4)
        a.download_button("시간별 CSV",frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="nernst_timeseries.csv",mime="text/csv")
        b.download_button("농도 분포 CSV",profile_frame(result).to_csv(index=False).encode("utf-8-sig"),
            file_name="nernst_profiles.csv",mime="text/csv")
        c.download_button("설정 JSON",json.dumps(asdict(s),ensure_ascii=False,indent=2),
            file_name="nernst_settings.json",mime="application/json")
        dcol.download_button("현재 실험 HTML 재생본",replay_html(result),
            file_name="nernst_replay.html",mime="text/html")
        st.caption("CSV의 step_current_A는 직전 시간 구간의 적분용 전류입니다. 전하 보존 검증은 이 값으로 합니다. "
                   "current_A는 해당 기록 시각의 회로 상태에 대한 전류이므로, 이상적인 스위치 전환에서 서로 다를 수 있습니다.")

elif page == PAGES[2]:
    st.markdown("### 심화 탐구 · 식의 예측과 운동 모델을 함께 검증")
    question=st.selectbox("탐구 질문",[
        "A. 로그 법칙을 직접 확인하기",
        "B. 같은 초기 전압이면 반응 가능한 양도 같을까?",
        "C. 이온 공급과 부하가 결과를 얼마나 바꿀까?",
        "D. 개별 입자의 무작위 운동은 확산과 어떻게 연결될까?",
    ],key="research_question")
    if question.startswith("A"):
        equation_panel()
        st.write("왼쪽 농도를 바꾸며 현재 계산값을 기록하세요. 서로 다른 Q에서 얻은 값을 모아 E–log₁₀Q의 기울기를 확인합니다.")
        st.caption("아래는 가상 모델의 관측 기록이며, 실제 전지를 측정한 실험 데이터가 아닙니다.")
        st.session_state.setdefault("observations",[])
        a,b=st.columns(2)
        if a.button("현재 조건 기록",type="primary",use_container_width=True):
            row={"Cu²⁺ (M)":s.cu_M,"Zn²⁺ (M)":s.zn_M,"T (°C)":s.temperature_C,
                 "Q":terms["Q"],"log10 Q":terms["log10_Q"],"E (V)":terms["E_V"]}
            st.session_state.observations.append(row)
        if b.button("기록 지우기",use_container_width=True):
            st.session_state.observations=[]
        obs=pd.DataFrame(st.session_state.observations)
        if not obs.empty:
            st.dataframe(obs,hide_index=True,use_container_width=True)
            f=figure("기록한 네른스트 응답","log₁₀ Q","E (V)")
            for temp,g in obs.groupby("T (°C)"):
                f.add_trace(go.Scatter(x=g["log10 Q"],y=g["E (V)"],mode="markers",name=f"{temp:g} °C"))
                if g["log10 Q"].nunique()>=2:
                    slope,intercept=np.polyfit(g["log10 Q"],g["E (V)"],1)
                    expected=-nernst_terms(.1,.1,float(temp))["k_V_per_decade"]
                    st.write(f"**{temp:g} °C**: 기록에서 구한 기울기 {slope:.6f} V/decade · 이론값 {expected:.6f} V/decade")
            show_plot(f,"observation_chart")
            st.download_button("가상 관측 기록 CSV",obs.to_csv(index=False).encode("utf-8-sig"),
                               "nernst_observations.csv","text/csv")
        else:
            st.info("서로 다른 농도비를 3개 이상 기록해 보세요. 온도가 다른 점들은 별도 묶음으로 분석합니다.")

    elif question.startswith("B"):
        st.markdown("#### 농도비는 같게, 절대 농도는 다르게")
        factors=[.5,1.,2.]
        table=[]
        for factor in factors:
            cu,zn=s.cu_M*factor,s.zn_M*factor
            table.append({"두 농도의 배율":factor, "Cu²⁺ (M)":cu,"Zn²⁺ (M)":zn,
                "Q":zn/cu,"초기 E_eq (V)":nernst(cu,zn,s.temperature_C),
                "구리 반응량 기준 상한 (mAh)":capacity_mAh(cu,s.volume_cu_ml)})
        st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
        st.latex(r"Q=\frac{k[\mathrm{Zn}^{2+}]}{k[\mathrm{Cu}^{2+}]}=\frac{[\mathrm{Zn}^{2+}]}{[\mathrm{Cu}^{2+}]}"
                 r"\quad\text{그러나}\quad q_{\max}=2F[\mathrm{Cu}^{2+}]_0V_{\mathrm{Cu}}")
        st.info("전압은 에너지/전하, mAh는 전하량입니다. 같은 초기 전압이 같은 반응 가능량을 뜻하지는 않습니다. "
                "여기서는 금속 아연이 충분하고 농도를 활동도로 근사한다고 가정합니다.")
        st.caption("농도가 높아질수록 이상 용액 가정의 실제 정확도는 낮아질 수 있습니다. 상한은 실제 사용 가능한 용량이 아닙니다.")
        f=figure("초기 전압이 같아도 총 반응량은 달라집니다","농도 배율","화학량론적 전하량 상한 (mAh)")
        f.add_trace(go.Bar(x=[str(x)+"배" for x in factors],
                          y=[x["구리 반응량 기준 상한 (mAh)"] for x in table],name="전하량 상한"))
        show_plot(f,"capacity_scale")

    elif question.startswith("C"):
        st.markdown("#### 한 변수만 바꾸는 공정한 비교")
        variable=st.radio("변경 변수",["외부 저항","확산계수"],horizontal=True,key="sweep_var")
        st.write("다른 조건을 고정하고 다섯 후보를 계산합니다. 표면 고갈 시점이 서로 다르면 "
                 "가장 짧은 유효 시간 안에서 공통 관찰 시간을 다시 정해 비교합니다.")
        st.caption("자동 비교는 연속 방전·최대 600 s로 고정합니다. 모든 가능한 조건에 대한 최적화는 아닙니다.")
        signature=json.dumps(asdict(s),sort_keys=True)+variable
        if st.button("다섯 조건 계산",type="primary",key="sweep_run"):
            with st.spinner("동일한 유효 시간으로 다섯 조건을 비교합니다."):
                configs=[]
                factors=[.25,.5,1.,2.,4.] if variable=="외부 저항" else [.5,.75,1.,1.5,2.]
                for factor in factors:
                    c=replace(s,duration_s=min(600.,s.duration_s),protocol="continuous",
                              dt_s=min(2.,s.dt_s),cells=min(48,s.cells))
                    c=replace(c,**({"resistance_ohm":s.resistance_ohm*factor} if variable=="외부 저항"
                                   else {"diffusion_m2_s":s.diffusion_m2_s*factor}))
                    configs.append(c)
                prelim=[run_cached(asdict(c)) for c in configs]
                common=min(z["actual_duration_s"] for z in prelim)*.98
                if common<.01:
                    st.warning("어떤 후보는 거의 즉시 표면 농도 하한에 도달합니다. 외부 저항을 높이거나 공급 조건을 개선하세요.")
                else:
                    outcomes=[run_cached(asdict(replace(c,duration_s=common,
                                          dt_s=min(c.dt_s,common/120)))) for c in configs]
                    st.session_state.sweep=(signature,common,factors,outcomes)
        stored=st.session_state.get("sweep")
        if stored and stored[0]==signature:
            _,common,factors,outcomes=stored
            rows=[]
            f=figure(f"공통 관찰 시간 {common:.3g} s","시간 (s)","단자 전압 (V)")
            for factor,z in zip(factors,outcomes):
                a=z["diffusion"];cfg=z["settings"]
                label=(f"{cfg['resistance_ohm']:g} Ω" if variable=="외부 저항"
                       else f"D={cfg['diffusion_m2_s']/1e-9:.3g}×10⁻⁹")
                rows.append({"후보":label,"관찰 (s)":z["actual_duration_s"],
                    "최종 I (mA)":a["current_A"][-1]*1e3,
                    "최종 V_term (V)":a["terminal_V"][-1],
                    "표면 Cu²⁺ (M)":a["cu_surface_M"][-1],
                    "누적 전하 (mAh)":a["charge_C"][-1]/3.6})
                f.add_trace(go.Scatter(x=a["time_s"],y=a["terminal_V"],mode="lines",name=label))
            show_plot(f,"sweep_chart")
            rows=pd.DataFrame(rows);st.dataframe(rows,hide_index=True,use_container_width=True)
            st.download_button("조건 비교 CSV",rows.to_csv(index=False).encode("utf-8-sig"),
                               "controlled_comparison.csv","text/csv")
        elif stored:
            st.caption("입력 조건이 바뀌었습니다. 이전 결과 대신 현재 조건으로 다시 계산하세요.")
        st.markdown("**회복 실험:** 왼쪽 ‘회로 연결 방식’을 ‘중간 휴지 후 재방전’으로 바꾸고 "
                    "‘02 · 방전과 이온 이동’을 열면, 전류가 0인 동안 표면 농도와 전압이 회복되는지를 확인할 수 있습니다.")

    else:
        st.markdown("#### 무작위 보행 → 평균제곱변위 → 확산")
        st.latex(r"\Delta x=\sqrt{2D\Delta t}\,\xi,\quad\xi\sim\mathcal{N}(0,1)"
                 r"\qquad \langle[x(t)-x(0)]^2\rangle=2Dt")
        st.caption("전지와 별개의 자유 1차원 확산 실험입니다. 전기장·전극·경계가 없으며, y축은 입자 번호일 뿐 공간 좌표가 아닙니다.")
        seed=st.number_input("난수 시드",0,9999,42,1,key="brownian_seed")
        walk=random_walk_cached(s.diffusion_m2_s,int(seed))
        k=st.slider("관찰 시점 · 시간 단계",0,len(walk["time_s"])-1,150,key="brownian_step")
        st.write(f"현재 시각 **{walk['time_s'][k]:.1f} s** · D = **{s.diffusion_m2_s/1e-9:.2f}×10⁻⁹ m²/s**")
        left,right=st.columns(2)
        with left:
            f=figure("대표 입자 12개의 실제 계산 위치","변위 x (mm)","입자 번호")
            x=walk["trajectories_m"][:,k]*1000
            f.add_trace(go.Scatter(x=x,y=np.arange(1,13),mode="markers",
                name="입자 위치",marker=dict(size=11,color="#1386a9")))
            limit=max(float(np.max(np.abs(walk["trajectories_m"]*1000)))*1.1,.1)
            f.update_xaxes(range=[-limit,limit]);f.update_yaxes(dtick=1)
            show_plot(f,"walk_positions")
        with right:
            f=figure("입자 2,400개의 통계와 이론 비교","시간 (s)","평균제곱변위 (mm²)")
            f.add_trace(go.Scatter(x=walk["time_s"],y=walk["msd_m2"]*1e6,name="무작위 보행",mode="lines"))
            f.add_trace(go.Scatter(x=walk["time_s"],y=walk["theory_m2"]*1e6,name="이론 2Dt",
                                  mode="lines",line=dict(dash="dash")))
            f.add_vline(x=float(walk["time_s"][k]),line_dash="dot")
            show_plot(f,"walk_msd")
        st.write("시드를 바꾸면 각각의 궤적은 달라지지만, 많은 입자의 평균적인 퍼짐은 2Dt에 가까워집니다. "
                 "개별 입자가 반드시 농도가 낮은 쪽으로만 움직인다는 뜻은 아닙니다.")

else:
    st.markdown("### 검증 가능한 모델로 만들기")
    st.markdown("#### 계산을 연결하는 네 가지 관계")
    st.latex(r"E_s=E^\circ-\frac{R_{\mathrm g}T}{2F}\ln\frac{c_{\mathrm{Zn}}}{c_{\mathrm{Cu},s}}")
    st.latex(r"I(R_{\mathrm{ext}}+r)=E_s,\qquad\Delta n_{\mathrm{Cu}}=-\frac{I\Delta t}{2F}")
    st.latex(r"\frac{\partial c}{\partial t}=D\frac{\partial^2c}{\partial x^2},"
             r"\qquad D\left.\frac{\partial c}{\partial x}\right|_{x=0}=\frac{I}{2FA}")
    st.caption("x는 전극에서 용액 쪽으로 증가합니다. Cu²⁺는 표면에서 소모되므로 소비 플럭스의 크기는 I/(2FA)입니다.")
    st.markdown("""
**모델 A — 균일 혼합:** 두 반쪽 전지의 농도가 공간적으로 균일합니다.  
**모델 B — 확산:** 구리 전극 앞의 1차원 확산층과 유한한 저장액을 함께 계산합니다. 아연 측은 균일 혼합으로 둡니다.  

두 모델은 같은 초기 조건과 같은 유효 시간에서 비교합니다. 구리 소모량과 아연 생성량은 누적 전하와 연결됩니다.
""")
    with st.expander("가정과 계산하지 않는 것",expanded=True):
        st.markdown("""
- **활동도≈농도**, 순수 고체의 활동도=1, 금속 아연은 충분함.
- **충분한 지지전해질**을 가정하여 반응 이온의 전기장 이동을 생략함. 층 내부의 대류도 생략함.
- 전극 반응은 빠르고 국소적으로 가역적이라고 가정함. 활성화 과전압·이중층 충전은 계산하지 않음.
- **온도는 네른스트 항에만 반영함.** E°=1.10 V, D, r의 온도 의존성은 생략함.
- 구리 용액 총부피는 확산층+유한 저장액의 합. 염다리·상대 이온의 상세 수송은 개념도에만 표시함.
- 전지 그림의 점은 농도 대표 표식. 실제 개별 입자 궤적은 별도의 무작위 보행 실험에서만 계산함.
- 표면 농도 10⁻⁸ M 하한은 수치 모델의 유효 범위 기준. 전압을 인위적으로 0 V로 만들지 않음.
- 실측 데이터로 보정하지 않았으므로 실제 전지 수명·출력을 정확히 예측한다고 주장하지 않음.
""")
    st.markdown("#### 왜 ‘E=0이 되면 정지’로 만들지 않았을까?")
    st.latex(fr"E=0\Rightarrow\log_{{10}}Q=\frac{{E^\circ}}{{2.303R_gT/(2F)}}"
             fr"={terms['log10_K_ideal']:.2f}")
    st.write("이 이상 모델에서는 0 V에 필요한 농도비가 매우 큽니다. 표면 농도가 모델의 유효 범위를 벗어나면 "
             "로그를 억지로 계산하거나 농도를 0으로 잘라내지 않고, 종료 이유를 표시합니다.")
    result=obtain_result()
    rows=[]
    for model,label in [("mixed","균일 혼합"),("diffusion","확산 고려")]:
        z=result["checks"][model]
        rows.append({"모델":label, "물질수지 상대 오차":z["relative_mass_error"],
            "전하 적분 오차 (C)":z["charge_error_C"], "회로식 잔차 (V)":z["circuit_error_V"],
            "검증": "통과" if z["valid"] else "확인 필요"})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption("작은 보존 오차는 계산이 일관적이라는 뜻이지, 생략한 물리 현상까지 정확하다는 뜻은 아닙니다.")
    if st.button("시간·공간 간격 수렴 검사",key="converge"):
        with st.spinner("세 해상도를 같은 요청 시간에서 비교합니다."):
            cv=convergence_experiment(s)
        st.dataframe(pd.DataFrame(cv),hide_index=True,use_container_width=True)
        durations=[r["duration_s"] for r in cv]
        if max(durations)-min(durations)>1e-6:
            st.warning("종료 시각이 달라 최종값 차이를 직접 수렴 오차로 해석할 수 없습니다. 부하를 줄이거나 관찰 시간을 짧게 설정하세요.")
        else:
            qref=cv[-1]["charge_C"]
            if qref:
                st.success(f"가장 거친 조건과 가장 세밀한 조건의 누적 전하 상대 차이: {abs(cv[0]['charge_C']-qref)/abs(qref)*100:.6g}%")
    with st.expander("출처",expanded=True):
        show_sources()
    with st.expander("프로젝트·배포 안내"):
        st.code("python -m pip install -r requirements.txt\npython -m streamlit run app.py",language="bash")
        st.write("GitHub 저장소 최상위에 app.py, physics.py, visuals.py, requirements.txt를 함께 올립니다. "
                 "Streamlit Community Cloud의 Main file path는 app.py입니다.")
        st.caption("프런트엔드 빌드, API 키, 비밀키, 외부 이미지 서버가 필요 없습니다.")

st.divider()
st.caption("NERNST LAB v2 · 네른스트 식을 중심으로 한 다니엘 전지·이온 확산 탐구 | "
           "모든 수치는 교육용 모델의 계산값이며 실제 측정값이 아닙니다.")
