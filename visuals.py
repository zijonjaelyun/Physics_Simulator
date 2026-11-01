"""Self-contained browser views. Python computes states; JS only renders them.

No CDN, external JavaScript, image assets, API keys, or frontend build are needed.
Representative dots are an illustration of concentration, NOT tracked ions.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from physics import Settings, nernst, nernst_terms

def timeseries_frame(result: dict) -> pd.DataFrame:
    frames = []
    for name in ("mixed", "diffusion"):
        d = pd.DataFrame(result[name])
        d.insert(0, "model", name)
        d["concentration_polarization_V"] = d.bulk_nernst_V - d.surface_nernst_V
        d["ohmic_loss_V"] = d.current_A * result["settings"]["internal_ohm"]
        d["log10_Q_surface"] = np.log10(d.zn_M / d.cu_surface_M)
        d["log10_Q_average"] = np.log10(d.zn_M / d.cu_average_M)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

def profile_frame(result: dict) -> pd.DataFrame:
    cols = [f"cu_M_x_{x:.7f}_mm" for x in result["x_mm"]] + ["cu_bulk_M"]
    d = pd.DataFrame(result["profiles_M"], columns=cols)
    d.insert(0, "time_s", result["diffusion"]["time_s"])
    return d

def result_payload(result: dict, max_frames: int = 800) -> dict:
    count = len(result["diffusion"]["time_s"])
    idx = set(np.linspace(0, count - 1, min(count, max_frames)).astype(int))
    changes = np.where(np.diff(result["diffusion"]["closed"]) != 0)[0]
    for i in changes:
        idx.update((int(i), int(i + 1)))
    ix = np.asarray(sorted(idx), dtype=int)
    data = {k: result[k] for k in ("settings", "status", "actual_duration_s")}
    data["mode"] = "replay"
    for name in ("mixed", "diffusion"):
        data[name] = {k: np.asarray(v)[ix].tolist() for k, v in result[name].items()}
    data["profiles_M"] = result["profiles_M"][ix].tolist()
    data["x_mm"] = result["x_mm"].tolist()
    return data

def snapshot_html(settings: Settings) -> str:
    e = nernst(settings.cu_M, settings.zn_M, settings.temperature_C)
    closed = settings.protocol != "open"
    current = max(0.0, e) / settings.total_r if closed else 0.0
    row = dict(time_s=[0.0], current_A=[current],
               terminal_V=[current*settings.resistance_ohm if closed else e],
               bulk_nernst_V=[e], surface_nernst_V=[e],
               cu_average_M=[settings.cu_M], cu_bulk_M=[settings.cu_M],
               cu_surface_M=[settings.cu_M], zn_M=[settings.zn_M],
               charge_C=[0.0], closed=[int(closed)])
    from dataclasses import asdict
    return html_from_payload(dict(mode="snapshot", settings=asdict(settings),
        status="snapshot", actual_duration_s=0, mixed=row, diffusion=row,
        profiles_M=[[settings.cu_M]*settings.cells+[settings.cu_M]],
        x_mm=((np.arange(settings.cells)+.5)*settings.film_mm/settings.cells).tolist()))

def replay_html(result: dict) -> str:
    return html_from_payload(result_payload(result))

def html_from_payload(data: dict) -> str:
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).replace("<", "\\u003c")
    return TEMPLATE.replace("__PAYLOAD__", encoded)

TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nernst Lab · 다니엘 전지</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f6f8fc;color:#1a304b;font-family:system-ui,-apple-system,"Malgun Gothic",sans-serif;font-size:14px}
.lab{max-width:1280px;margin:auto;padding:16px}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}
.eyebrow{font-size:10px;letter-spacing:2px;color:#55748f;font-weight:750}
h2{margin:3px 0 0;font-size:19px;letter-spacing:-.6px}
.badge{font-size:11px;padding:6px 11px;border-radius:20px;background:#e3eef5;color:#235471;white-space:nowrap}
.formula{background:#132d46;color:#f4f8fd;border-radius:14px;padding:17px 22px;display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center;margin-bottom:12px}
.formula small{color:#b3c4d8;font-size:11px;display:block;margin-bottom:8px}
.symbol{font-family:Georgia,serif;font-size:24px;line-height:1.6;letter-spacing:.2px}
.num{color:#c2d9e8;font-size:13px;margin-top:6px;line-height:1.7;font-variant-numeric:tabular-nums}
.big{font-size:37px;font-weight:780;letter-spacing:-1.5px;text-align:right}.big span{font-size:15px;color:#b3c4d8}
.cu{color:#7ce1fb}.zn{color:#c1baff}.gold{color:#ffd186}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
button,select{font:inherit;border:1px solid #cbd6e2;background:#fff;color:#203d58;padding:8px 12px;border-radius:8px;cursor:pointer}
button.primary{background:#176a8f;color:white;border-color:#176a8f;min-width:94px}
button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #d68c30;outline-offset:2px}
.range{display:flex;align-items:center;gap:12px;background:#fff;border:1px solid #e0e7f0;border-radius:10px;padding:10px 12px;margin-bottom:10px}
input[type=range]{flex:1;width:100%;accent-color:#176a8f}
.time{min-width:135px;font-variant-numeric:tabular-nums;font-weight:650;font-size:12px;text-align:right}
.scene{background:white;border:1px solid #dce5ef;border-radius:14px;overflow:hidden}
canvas{display:block;width:100%;height:auto}
#cell{aspect-ratio:1080/382}
.meters{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}
.meter{border:1px solid #dce5ef;background:#fff;border-radius:10px;padding:11px 13px;min-width:0}
.meter label{font-size:11px;color:#60788e;display:block;line-height:1.5}
.meter strong{display:block;font-size:20px;letter-spacing:-.6px;margin-top:4px;font-variant-numeric:tabular-nums;word-break:break-word}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.plot{border:1px solid #dce5ef;border-radius:12px;background:#fff;padding:12px 10px 5px;min-width:0}
.plot h3{font-size:12px;margin:0 0 6px 6px}
.plot canvas{aspect-ratio:520/216}
.legend{font-size:10px;color:#6b7c8f;margin:5px 6px;line-height:1.6}
.note{font-size:11px;color:#61758b;line-height:1.8;margin:9px 2px 0}
.status{margin-top:8px;padding:8px 12px;border-radius:8px;background:#fff3d8;color:#805c1c;font-size:12px;line-height:1.6}
[hidden]{display:none!important}
@media(max-width:620px){.lab{padding:8px}.formula{grid-template-columns:1fr;padding:14px}.big{text-align:left;font-size:29px}
.symbol{font-size:20px}.num{font-size:11px}.meters{grid-template-columns:repeat(2,1fr)}.plots{grid-template-columns:1fr}
h2{font-size:16px}.badge{font-size:10px}.time{min-width:110px}.controls{font-size:12px}}
</style></head><body>
<div class="lab">
<div class="top"><div><div class="eyebrow">NERNST LAB / Zn–Cu</div><h2 id="viewTitle">현재 조건의 전지</h2></div><div class="badge" id="badge"></div></div>
<div class="formula" id="formula">
 <div><small id="formulaTitle">현재 시각 · 표면 농도를 네른스트 식에 대입</small>
 <div class="symbol">E<sub>s</sub> = E° − <span class="gold">RT / 2F</span> · ln(<span class="zn">[Zn²⁺]</span> / <span class="cu">[Cu²⁺]<sub>s</sub></span>)</div>
 <div class="num" id="substitution"></div></div>
 <div class="big"><b id="es"></b><span> V</span></div>
</div>
<div id="transport">
 <div class="controls">
 <button id="play" class="primary">▶ 재생</button><button id="reset">처음으로</button>
 <select id="speed" aria-label="재생 속도"><option value="10">10배속</option><option value="30">30배속</option><option value="60">60배속</option><option value="120">120배속</option><option value="300">300배속</option></select>
 <select id="model" aria-label="계산 모델"><option value="diffusion">확산 고려 모델</option><option value="mixed">균일 혼합 모델</option></select>
 </div>
 <div class="range"><input type="range" id="seek" min="0" max="100" step="0.1" value="0" aria-label="물리 시간"><div class="time" id="clock"></div></div>
</div>
<div class="scene"><canvas id="cell" width="1080" height="382" aria-label="다니엘 전지와 대표 이온"></canvas></div>
<div class="meters">
 <div class="meter"><label>평균 농도 기준 E<sub>avg</sub></label><strong id="eb"></strong></div>
 <div class="meter"><label>단자 전압 V<sub>term</sub></label><strong id="vt"></strong></div>
 <div class="meter"><label>외부 회로 전류 I</label><strong id="current"></strong></div>
 <div class="meter"><label id="lastLabel">구리 석출량 · 패러데이 법칙</label><strong id="deposit"></strong></div>
</div>
<div class="plots" id="plots">
 <div class="plot"><h3>Cu²⁺ 농도 분포 · 전극 표면 → 저장액</h3><canvas id="profile" width="520" height="216"></canvas>
 <div class="legend">파랑: 현재 농도　점선: 초기 농도　●: 표면 농도(회로식에 사용)</div></div>
 <div class="plot"><h3>네른스트 전압과 단자 전압 · 같은 시간</h3><canvas id="timeline" width="520" height="216"></canvas>
 <div class="legend">회색: E<sub>avg</sub>　파랑: E<sub>s</sub>　주황: V<sub>term</sub>　세로선: 현재 시각</div></div>
</div>
<div class="status" id="stopNote" hidden></div>
<div class="note" id="note"></div>
</div>
<script>
"use strict";
const DATA=__PAYLOAD__;
const S=DATA.settings, LIVE=DATA.mode==="snapshot", TMAX=DATA.actual_duration_s;
const F=96485.33212, RG=8.31446261815324, TEMP=S.temperature_C+273.15;
const K=RG*TEMP/(2*F)*Math.log(10);
const $=id=>document.getElementById(id);
let model=LIVE?"mixed":"diffusion", idx=0, cursor=0, playing=false, last=performance.now(), anim=0;
const times=DATA.diffusion.time_s, maxDots=120;
const dots=Array.from({length:220},(_,i)=>({x:fract(Math.sin(i*127.1+13)*43758.5453),
y:fract(Math.sin(i*311.7+77)*25311.343),p:i*1.77}));
function fract(x){return x-Math.floor(x)}
function fmt(x,n=4){return Math.abs(x)>0&&Math.abs(x)<Math.pow(10,-n)?x.toExponential(2):x.toFixed(n)}
function state(){const d=DATA[model];let r={};for(const key of Object.keys(d))r[key]=d[key][idx];return r}
function text(ctx,str,x,y,size=15,color="#304c66",align="left",weight="500"){
 ctx.font=weight+" "+size+'px system-ui,"Malgun Gothic",sans-serif';ctx.fillStyle=color;ctx.textAlign=align;ctx.fillText(str,x,y);
}
function rr(ctx,x,y,w,h,r,fill,stroke=null){ctx.beginPath();ctx.roundRect(x,y,w,h,r);if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.stroke()}}
function line(ctx,pts,color,width=2,dash=[]){ctx.beginPath();ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=width;pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();ctx.setLineDash([])}
function arrow(ctx,x1,y1,x2,y2,color="#607d95",width=2){line(ctx,[[x1,y1],[x2,y2]],color,width);let a=Math.atan2(y2-y1,x2-x1);ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-8*Math.cos(a-.45),y2-8*Math.sin(a-.45));ctx.lineTo(x2-8*Math.cos(a+.45),y2-8*Math.sin(a+.45));ctx.closePath();ctx.fillStyle=color;ctx.fill()}
function circle(ctx,x,y,r,fill,stroke){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.stroke()}}
function renderCell(r){
 const c=$("cell"),ctx=c.getContext("2d");ctx.clearRect(0,0,1080,382);ctx.fillStyle="#fff";ctx.fillRect(0,0,1080,382);
 // Electron conductor stays outside the electrolyte.
 line(ctx,[[155,138],[155,40],[925,40],[925,138]],r.closed?"#526d83":"#aebecb",3);
 rr(ctx,447,23,184,34,7,"#f1f5f9","#b8c7d5");text(ctx,r.closed?("R_ext = "+S.resistance_ohm.toLocaleString()+" Ω"):"개방 회로 · I = 0",539,46,14,"#39546e","center");
 if(r.closed){arrow(ctx,660,40,719,40,"#d59a2f",2.3);text(ctx,"e⁻  Zn → Cu",730,31,12,"#927128");}
 else {ctx.fillStyle="white";ctx.fillRect(681,34,38,12);line(ctx,[[681,40],[714,21]],"#8298ab",2)}
 text(ctx,"아연 전극 · 산화 · (−)",105,106,18,"#4d4b7f","left","700");
 text(ctx,"구리 전극 · 환원 · (+)",678,106,18,"#167a98","left","700");
 // Salt bridge, schematic only.
 const bridge=[[347,169],[347,128],[735,128],[735,169]];
 line(ctx,bridge,"#cbd7e1",19);line(ctx,bridge,"#eef3f7",12);
 text(ctx,"염다리 · 전하 균형의 개념도",540,99,12,"#718499","center");
 arrow(ctx,493,129,420,129,"#8c79aa");arrow(ctx,583,129,654,129,"#539887");
 text(ctx,"NO₃⁻ → 아연 쪽",390,155,11,"#7d6c98");
 text(ctx,"K⁺ → 구리 쪽",599,155,11,"#477e73");
 // Beakers; color/points show concentration, not a chemistry-specific optical law.
 let ref=Math.max(S.cu_M,S.zn_M,r.cu_average_M,r.zn_M);
 for(const spec of [{x:98,c:r.zn_M,col:"#8378ba",name:"Zn²⁺"},{x:677,c:r.cu_average_M,col:"#168db4",name:"Cu²⁺"}]){
  rr(ctx,spec.x,163,306,168,0,spec.name==="Cu²⁺"?"#e7f5fb":"#f2f0fa");
  ctx.save();ctx.globalAlpha=Math.min(.22,.04+.18*spec.c/ref);ctx.fillStyle=spec.col;ctx.fillRect(spec.x,163,306,168);ctx.restore();
  line(ctx,[[spec.x,142],[spec.x,333],[spec.x+306,333],[spec.x+306,142]],"#94aabc",2.5);
  const count=Math.max(0,Math.round(maxDots*spec.c/ref));
  for(let j=0;j<count;j++){
   const q=dots[j],xx=spec.x+19+((q.x+.012*Math.sin(anim*.7+q.p)+1)%1)*265;
   const yy=180+((q.y+.02*Math.cos(anim*.9+q.p)+1)%1)*128;
   circle(ctx,xx,yy,3.3,spec.col);
  }
 }
 rr(ctx,137,139,36,173,3,"#9a9cb2");rr(ctx,907,139,36,173,3,"#b57349");
 text(ctx,"Zn",155,202,18,"white","center","750");text(ctx,"Cu",925,202,18,"white","center","750");
 text(ctx,"Zn(s) → Zn²⁺ + 2e⁻",252,357,16,"#5f588c","center","650");
 text(ctx,"Cu²⁺ + 2e⁻ → Cu(s)",830,357,16,"#177d9c","center","650");
 // Numeric concentrations avoid ambiguity at dot counts below resolution.
 rr(ctx,193,268,190,43,8,"rgba(255,255,255,.95)","#e4e8ef");
 text(ctx,"[Zn²⁺] = "+fmt(r.zn_M)+" M",288,294,16,"#5f588c","center","700");
 rr(ctx,698,268,195,43,8,"rgba(255,255,255,.95)","#dcecf3");
 text(ctx,"[Cu²⁺]avg = "+fmt(r.cu_average_M)+" M",795,294,15,"#177d9c","center","700");
 rr(ctx,431,186,218,107,12,"#f6f9fc","#e1e8ef");
 text(ctx,"Zn + Cu²⁺ → Zn²⁺ + Cu",540,214,14,"#36516c","center","650");
 text(ctx,(r.closed?"전류가 흐르는 상태":"전류가 없는 상태"),540,241,13,"#718499","center");
 text(ctx,((LIVE?"초기 ":"")+"Q"+(model==="diffusion"?"_s":"")+" = "+fmt(r.zn_M/r.cu_surface_M,3)),540,269,17,"#183f60","center","750");
 if(r.closed && r.current_A>0){
  const rate=Math.max(.1,Math.min(3,r.current_A*1000/1));
  for(let j=0;j<7;j++){
   const t=(anim*.075*rate+j/7)%1,px=160+t*760;
   if(px<444||px>637){circle(ctx,px,40,5,"#d0a64c");}
  }
 }
}
function axes(ctx,w,h,xname,yname,ylim,xmax){
 const b={l:57,r:w-18,t:19,b:h-43};ctx.clearRect(0,0,w,h);
 for(let j=0;j<4;j++){let y=b.b-(b.b-b.t)*j/3;line(ctx,[[b.l,y],[b.r,y]],"#e8edf4",1);
 text(ctx,fmt(ylim[0]+(ylim[1]-ylim[0])*j/3,3),b.l-7,y+4,10,"#78899b","right")}
 line(ctx,[[b.l,b.t],[b.l,b.b],[b.r,b.b]],"#bbc9d6",1);
 for(let j=0;j<3;j++){let x=b.l+(b.r-b.l)*j/2;text(ctx,(xmax*j/2).toFixed(xmax<10?2:0),x,b.b+17,10,"#78899b","center")}
 text(ctx,xname,(b.l+b.r)/2,h-7,10,"#647c91","center");text(ctx,yname,9,12,10,"#647c91");
 return {X:x=>b.l+(b.r-b.l)*x/xmax,Y:y=>b.b-(b.b-b.t)*(y-ylim[0])/(ylim[1]-ylim[0]),b:b};
}
function renderPlots(r){
 if(LIVE)return;
 const c=$("profile"),ctx=c.getContext("2d"),profile=DATA.profiles_M[idx];
 const ymax=Math.max(S.cu_M,r.cu_average_M,...profile)*1.1;
 const a=axes(ctx,520,216,"전극에서 거리 x (mm)","c (M)",[0,ymax],S.film_mm);
 line(ctx,[[a.X(0),a.Y(S.cu_M)],[a.X(S.film_mm),a.Y(S.cu_M)]],"#9aa9b9",1,[5,4]);
 let points=[[0,r.cu_surface_M]];
 DATA.x_mm.forEach((x,i)=>points.push([x,model==="diffusion"?profile[i]:r.cu_average_M]));
 points.push([S.film_mm,model==="diffusion"?profile[profile.length-1]:r.cu_average_M]);
 line(ctx,points.map(p=>[a.X(p[0]),a.Y(p[1])]),"#158cb0",2.4);
 circle(ctx,a.X(0),a.Y(r.cu_surface_M),4,"#157da1");
 const tc=$("timeline"),tctx=tc.getContext("2d"),d=DATA[model];
 const fields=["bulk_nernst_V","surface_nernst_V","terminal_V"];
 let lo=Math.min(...fields.flatMap(f=>d[f])),hi=Math.max(...fields.flatMap(f=>d[f]));
 let span=Math.max(.03,hi-lo);lo=Math.max(0,lo-span*.18);hi=hi+span*.15;
 const q=axes(tctx,520,216,"시간 t (s)","V",[lo,hi],Math.max(TMAX,1));
 const cols=["#97a5b7","#1785aa","#cc853e"];
 fields.forEach((f,j)=>line(tctx,times.map((t,i)=>[q.X(t),q.Y(d[f][i])]),cols[j],j===2?2.3:1.5,j===0?[4,4]:[]));
 line(tctx,[[q.X(times[idx]),q.b.t],[q.X(times[idx]),q.b.b]],"#344e67",1.2,[3,3]);
 fields.forEach((f,j)=>circle(tctx,q.X(times[idx]),q.Y(d[f][idx]),3.4,cols[j]));
}
function renderNumbers(r){
 $("badge").textContent=(LIVE?"초기 상태 미리보기":model==="diffusion"?"확산 · 유한 저장액":"균일 혼합")+" / "+S.temperature_C+" °C";
 $("es").textContent=r.surface_nernst_V.toFixed(5);
 const q=r.zn_M/r.cu_surface_M;
 $("substitution").innerHTML="1.10000 − "+K.toFixed(5)+" × log₁₀(<span class='zn'>"+fmt(r.zn_M,6)+"</span> / <span class='cu'>"+fmt(r.cu_surface_M,6)+"</span>) = "+r.surface_nernst_V.toFixed(5)+" V"+
 "<br>Q = "+fmt(q,4)+"　|　log₁₀Q = "+Math.log10(q).toFixed(5)+"　|　T = "+TEMP.toFixed(2)+" K";
 $("formulaTitle").textContent=model==="diffusion"?"현재 시각 · Cu 전극 표면 농도를 대입 (국소 가역 계면 가정)":"현재 시각 · 균일 농도를 대입";
 $("eb").textContent=r.bulk_nernst_V.toFixed(5)+" V";
 $("vt").textContent=r.terminal_V.toFixed(5)+" V";
 $("current").textContent=(r.current_A*1000).toFixed(4)+" mA";
 $("deposit").textContent=(r.charge_C/(2*F)*63.546*1000).toFixed(4)+" mg";
 if(LIVE){$("lastLabel").textContent="농도에 의한 전압 보정 ΔE";$("deposit").textContent=((r.bulk_nernst_V-1.1)*1000).toFixed(3)+" mV";}
 $("clock").textContent=times[idx].toFixed(1)+" / "+TMAX.toFixed(1)+" s";
}
function render(){const r=state();renderNumbers(r);renderCell(r);renderPlots(r)}
function indexAt(t){let l=0,h=times.length-1;while(l<h){let m=Math.ceil((l+h)/2);if(times[m]<=t)l=m;else h=m-1}return l}
function seek(t){cursor=Math.min(TMAX,Math.max(0,t));idx=indexAt(cursor);$("seek").value=cursor;render()}
function stop(){playing=false;$("play").textContent="▶ 재생"}
$("play").onclick=()=>{if(cursor>=TMAX)seek(0);playing=!playing;$("play").textContent=playing?"❚❚ 일시정지":"▶ 재생";last=performance.now()};
$("reset").onclick=()=>{stop();anim=0;seek(0)};
$("seek").oninput=e=>{stop();seek(Number(e.target.value))};
$("model").onchange=e=>{model=e.target.value;render()};
$("speed").value=TMAX>1800?"120":TMAX>600?"60":"30";
$("seek").max=TMAX;$("seek").step=Math.max(.001,TMAX/3000);
$("viewTitle").textContent=LIVE?"농도에 반응하는 전지 그림":"시간에 따라 식과 전지가 함께 변합니다";
if(LIVE){
 $("transport").hidden=true;$("plots").hidden=true;$("formula").hidden=true;
 $("note").textContent="초기 상태 그림입니다. 입력을 바꾸면 농도·전압·전류가 바뀌며, 이 화면의 장식적 움직임 자체가 농도를 소모하지는 않습니다. 점 수는 같은 시각의 농도비를 근사한 대표점(최대 120개)이며 개별 이온 궤적이 아닙니다. 색은 이온 구분용이며 실제 용액 색이 아닙니다. 실제 방전은 ‘방전·이온 이동’에서 확인하세요.";
}else{
 $("note").textContent="일시정지·시간 이동·모델 선택 모두 같은 계산 시각을 표시합니다. 점은 농도의 대표 표식이며 실제 이온 궤적이 아닙니다. 색은 구분용, 염다리는 개념도이며 정량 수송 계산은 Cu 전극 앞 1차원 확산층에 한정합니다. E°·D·r의 온도 의존성은 생략했습니다.";
 if(DATA.status!=="duration"){
  $("stopNote").hidden=false;$("stopNote").textContent="표면 농도가 모델 하한(10⁻⁸ M)에 도달하여 "+TMAX.toFixed(3)+" s에서 비교를 종료했습니다. 이는 전압 0 V, 전체 이온 소진 또는 실제 전지 수명을 뜻하지 않습니다.";
 }
}
document.addEventListener("visibilitychange",()=>{if(document.hidden)stop()});
function loop(now){
 const elapsed=Math.min((now-last)/1000,.2);last=now;
 if(LIVE||playing)anim+=elapsed;
 if(playing){cursor+=elapsed*Number($("speed").value);if(cursor>=TMAX){cursor=TMAX;stop()}idx=indexAt(cursor);$("seek").value=cursor;renderNumbers(state());renderPlots(state())}
 if(LIVE||playing)renderCell(state());
 requestAnimationFrame(loop);
}
render();requestAnimationFrame(loop);
window.__labTest={seek,getState:state,getIndex:()=>idx,getModel:()=>model,isPlaying:()=>playing,data:DATA};
</script></body></html>"""
