"""Create a dependency-free interactive 3D scatter view of computed atlas nodes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"


def main():
    path=OUT/"atlas_retention_physical_summary.csv"
    with path.open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    points=[]
    for r in rows:
        points.append([float(r["alpha"]),float(r["beta"]),float(r["kappa"]),
                       float(r["P_ret"]),r["model"],float(r["eta"]),
                       float(r["tau0_deg"]),str(r["within_250V"]).lower()=="true"])
    payload=json.dumps(points,separators=(",",":"),ensure_ascii=False)
    html='''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V13.3 — интерактивный 3D-атлас</title><style>
body{font:15px system-ui,sans-serif;margin:18px;background:#f4f6fa;color:#132033}h1{font-size:22px;margin:0 0 12px}
.bar{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:10px 0 14px}.bar label{display:flex;gap:6px;align-items:center;background:white;padding:8px 10px;border-radius:7px;box-shadow:0 1px 4px #0001}
select{font-size:14px;padding:4px}canvas{width:min(100%,1100px);height:auto;background:white;border:1px solid #ccd3df;border-radius:8px;touch-action:none}
.note{max-width:1100px;line-height:1.45;color:#42526a}.legend{display:flex;gap:15px;margin:8px 0}.swatch{display:inline-block;width:13px;height:13px;border-radius:50%;margin-right:5px;vertical-align:-2px}
</style></head><body><h1>V13.3: вычисленные узлы нелинейного удержания</h1>
<div class="bar"><label>η / срез <select id="eta"><option value="all">Все срезы (3D)</option></select></label>
<label>RF-фаза τ₀ <select id="phase"><option>0</option><option>90</option></select>°</label>
<label>Модель <select id="model"><option value="all">Q и Q+RO</option><option value="Q">Q</option><option value="RO">Q+RO</option></select></label>
<label><input id="physical" type="checkbox"> Только узлы ≤250 В</label></div>
<canvas id="plot" width="1100" height="700"></canvas><div class="legend"><span><b>○</b> Q</span><span><b>▲</b> Q+RO</span><span>Цвет: P_ret (синий низкий, зелёный средний, красный высокий)</span></div>
<p class="note" id="info"></p><p class="note">Поворачивайте 3D-вид мышью/касанием. Показаны только фактически интегрированные узлы; поверхность между точками не реконструируется. Q и Q+RO используют общие начальные состояния. Это конечновременное поперечное удержание при фиксированной апертуре, а не передача прибора конечной длины.</p>
<script>const points=__POINTS__;
const etaSelect=document.getElementById('eta'),phaseSelect=document.getElementById('phase'),modelSelect=document.getElementById('model'),physicalSelect=document.getElementById('physical'),canvas=document.getElementById('plot'),ctx=canvas.getContext('2d'),info=document.getElementById('info');
const etas=[...new Set(points.map(p=>p[5]))].sort((a,b)=>a-b);for(const e of etas){const o=document.createElement('option');o.value=String(e);o.textContent='η='+Number(e).toFixed(3);etaSelect.appendChild(o)}
let yaw=-0.65,pitch=0.43,drag=null;
function color(v){let r,g,b;if(v<.5){const q=v*2;r=Math.round(40+40*q);g=Math.round(95+140*q);b=Math.round(210-115*q)}else{const q=(v-.5)*2;r=Math.round(80+165*q);g=Math.round(235-145*q);b=Math.round(95-80*q)}return `rgb(${r},${g},${b})`}
function draw(){const W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);const selected=points.filter(p=>(etaSelect.value==='all'||p[4]==='Q'||Math.abs(p[5]-Number(etaSelect.value))<1e-9)&&p[6]===Number(phaseSelect.value)&&(modelSelect.value==='all'||p[4]===modelSelect.value)&&(!physicalSelect.checked||p[7]));
const projected=[];for(const p of selected){let x=(p[0]+.05)/.1-.5,y=(p[1]-.05)/.25-.5,z=(p[2]+.06)/.12-.5;let xx=x*Math.cos(yaw)+z*Math.sin(yaw),zz=-x*Math.sin(yaw)+z*Math.cos(yaw);let yy=y*Math.cos(pitch)-zz*Math.sin(pitch);zz=y*Math.sin(pitch)+zz*Math.cos(pitch);const perspective=3.6/(3.6+zz);projected.push({x:W/2+xx*W*.68*perspective,y:H/2-yy*H*.68*perspective,z:zz,p})}projected.sort((a,b)=>a.z-b.z);
ctx.strokeStyle='#8290a5';ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(W*.18,H*.82);ctx.lineTo(W*.78,H*.82);ctx.moveTo(W*.18,H*.82);ctx.lineTo(W*.18,H*.17);ctx.moveTo(W*.18,H*.82);ctx.lineTo(W*.35,H*.61);ctx.stroke();ctx.fillStyle='#27364c';ctx.font='16px system-ui';ctx.fillText('β',W*.79,H*.83);ctx.fillText('α',W*.15,H*.14);ctx.fillText('κ',W*.35,H*.59);
for(const q of projected){ctx.fillStyle=color(q.p[3]);ctx.globalAlpha=q.p[7]?0.87:0.28;if(q.p[4]==='Q'){ctx.beginPath();ctx.arc(q.x,q.y,3.3,0,2*Math.PI);ctx.fill()}else{ctx.beginPath();ctx.moveTo(q.x,q.y-4);ctx.lineTo(q.x-3.8,q.y+3.4);ctx.lineTo(q.x+3.8,q.y+3.4);ctx.closePath();ctx.fill()}}ctx.globalAlpha=1;info.textContent=`Узлов показано: ${selected.length} из ${points.length}. τ₀=${phaseSelect.value}°. Математические точки выше лимита при отключённом фильтре затемнены.`}
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];yaw+=dx*.008;pitch=Math.max(-1.45,Math.min(1.45,pitch+dy*.008));drag=[e.clientX,e.clientY];draw()});canvas.addEventListener('pointerup',()=>drag=null);for(const e of [etaSelect,phaseSelect,modelSelect,physicalSelect])e.addEventListener('change',draw);draw();</script></body></html>'''.replace("__POINTS__",payload)
    (OUT/"interactive_three_parameter_atlas.html").write_text(html,encoding="utf-8")
    print(f"wrote {OUT/'interactive_three_parameter_atlas.html'} points={len(points)}")


if __name__=="__main__":main()
