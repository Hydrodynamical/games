"""Generate a self-contained HTML inspector for human-game training and search.

No chess, neural, browser, or plotting dependencies are needed. Partial runs are
readable; malformed records produce visible warnings instead of hiding results.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _read(path, default, warnings):
    if not path.exists():
        return default
    try:
        result = json.loads(path.read_text(), parse_constant=_reject_constant)
        if not isinstance(result, type(default)):
            raise ValueError(f"Expected {type(default).__name__}")
        return result
    except (OSError, ValueError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return default


def _jsonl(path, warnings):
    if not path.exists():
        return []
    records = []
    try:
        with path.open() as stream:
            for index, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line, parse_constant=_reject_constant)
                    if not isinstance(record, dict):
                        raise ValueError("Expected object")
                    records.append(record)
                except ValueError:
                    warnings.append(f"Skipped invalid {path.name} line {index}")
    except (OSError, UnicodeError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
    return records


def _match_totals(records):
    # A cutoff is never scored, even if an inconsistent old row supplies zero.
    completed = [row for row in records if row.get("termination") != "move_limit"
                 and not row.get("truncated") and row.get("mcts_score") in (0, .5, 1)]
    return {"games": len(records), "completed_games": len(completed),
            "mcts_wins": sum(row["mcts_score"] == 1 for row in completed),
            "policy_wins": sum(row["mcts_score"] == 0 for row in completed),
            "actual_draws": sum(row["mcts_score"] == .5 for row in completed),
            "unresolved_cutoffs": sum(row.get("termination") == "move_limit" or bool(row.get("truncated"))
                                      for row in records),
            "mcts_score_on_completed_games": (sum(row["mcts_score"] for row in completed) / len(completed)
                                              if completed else None)}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _learning_view(data):
    """Keep frozen-checkpoint evaluations separate from within-epoch averages."""
    config, summary = data["config"], data["summary"]
    by_epoch = {}
    summary_epochs = summary.get("epochs") if isinstance(summary.get("epochs"), list) else []
    # History is authoritative; summary epoch records supply fields omitted by
    # older/intermediate history writers. Never substitute averaged train loss.
    for record in summary_epochs + data["validation"]:
        if not isinstance(record, dict) or not _finite(record.get("epoch")):
            continue
        merged = by_epoch.setdefault(record["epoch"], {})
        for key, value in record.items():
            if value is not None or key not in merged:
                merged[key] = value
    rows = []
    start = config.get("start_epoch", 0)
    for epoch, record in sorted(by_epoch.items()):
        validation = record.get("validation") or {}
        training = record.get("training_evaluation") or {}
        validation = validation if isinstance(validation, dict) else {}
        training = training if isinstance(training, dict) else {}
        train_nll, val_nll = training.get("policy_nll_full_action"), validation.get("policy_nll_full_action")
        row = {"epoch": epoch, "step": record.get("step"), "is_start": epoch == start,
               "training_evaluation": training, "validation": validation,
               "train_nll": train_nll, "validation_nll": val_nll,
               "train_top1": training.get("policy_top1_full_action"),
               "validation_top1": validation.get("policy_top1_full_action"),
               "train_value_mse": training.get("value_mse"), "validation_value_mse": validation.get("value_mse"),
               "train_zero_mse": training.get("zero_value_baseline_mse"),
               "validation_zero_mse": validation.get("zero_value_baseline_mse"),
               "policy_nll_gap": val_nll - train_nll if _finite(val_nll) and _finite(train_nll) else None,
               "within_epoch_policy_loss": record.get("policy_loss"),
               "within_epoch_value_loss": record.get("value_loss")}
        rows.append(row)
    measured = [row for row in rows if _finite(row["validation_nll"])]
    observed_best = min(measured, key=lambda row: row["validation_nll"]) if measured else {}
    best_epoch = summary.get("best_epoch", observed_best.get("epoch"))
    best_nll = summary.get("best_validation_nll", observed_best.get("validation_nll"))
    comparable = [row for row in rows if _finite(row["train_nll"]) and _finite(row["validation_nll"])]
    divergence_epochs = [later["epoch"] for earlier, later in zip(comparable, comparable[1:])
                         if later["train_nll"] < earlier["train_nll"]
                         and later["validation_nll"] > earlier["validation_nll"]]
    return {"epochs": rows, "start_epoch": start, "best_epoch": best_epoch, "best_validation_nll": best_nll,
            "planned_final_epoch": config.get("planned_final_epoch",
                start + config["epochs"] if _finite(start) and _finite(config.get("epochs")) else None),
            "completed_epoch": summary.get("completed_epoch"),
            "epochs_completed": summary.get("epochs_completed", len(summary_epochs)),
            "stopped_early": summary.get("stopped_early"), "stop_reason": summary.get("stop_reason"),
            "epochs_without_improvement": summary.get("epochs_without_improvement"),
            "patience": config.get("patience"), "min_delta": config.get("min_delta"),
            "divergence_epochs": divergence_epochs,
            "train_evaluation_available": bool(comparable)}


def generate_report(run_dir, output=None):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    warnings = []
    data = {"run_name": run_dir.name, "config": _read(run_dir / "config.json", {}, warnings),
            "dataset": _read(run_dir / "dataset_manifest.json", {}, warnings),
            "summary": _read(run_dir / "summary.json", {}, warnings),
            "updates": _jsonl(run_dir / "updates.jsonl", warnings),
            "validation": _read(run_dir / "validation.json", [], warnings), "evaluations": {}, "warnings": warnings}
    data["learning"] = _learning_view(data)
    for phase in ("initial", "final"):
        directory = run_dir / f"eval-{phase}"
        evaluation = {"summary": _read(directory / "summary.json", {}, warnings),
                      "positions": _jsonl(directory / "positions.jsonl", warnings),
                      "matches": _jsonl(directory / "matches.jsonl", warnings)}
        evaluation["match_totals"] = (_match_totals(evaluation["matches"]) if evaluation["matches"]
                                      else evaluation["summary"].get("matches", {}))
        data["evaluations"][phase] = evaluation
    initial, final = (data["evaluations"][phase]["positions"] for phase in ("initial", "final"))
    identity = lambda rows: [(row.get("game_id"), row.get("ply"), row.get("fen"), row.get("human_move")) for row in rows]
    data["initial_final_samples_match"] = identity(initial) == identity(final) if initial and final else None
    if data["initial_final_samples_match"] is False:
        warnings.append("Initial and final evaluation samples differ. Each raw/MCTS comparison still uses identical positions within its own evaluation.")
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    document = TEMPLATE.replace("__HUMAN_DATA__", payload)
    destination = Path(output).resolve() if output else run_dir / "report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(document)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human chess training inspector</title><style>
:root{color-scheme:light;--ink:#20323e;--muted:#5a6e7b;--line:#dce5e9;--blue:#247a9b;--orange:#ba7028}
*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:var(--ink);font:15px/1.55 system-ui,sans-serif}main{max-width:1220px;margin:auto;padding:35px 24px 70px}h1{font-size:34px;line-height:1.2;margin:6px 0 12px}h2{font-size:22px;margin:0 0 8px}h3{font-size:17px;margin:0 0 8px}p{margin:6px 0 15px}.muted,.caption{color:var(--muted)}.caption{font-size:13px}.eyebrow{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--blue)}.section{margin-top:24px}.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.note{border-left:4px solid var(--orange);background:#fff6e9;padding:12px 16px;white-space:pre-wrap}.hidden{display:none}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 10px;white-space:nowrap}th{font-size:12px;color:var(--muted)}td:first-child,th:first-child{padding-left:0}.chart svg{width:100%;height:auto;display:block}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px}.legend span::before{content:'●';color:var(--series);margin-right:5px}.empty{padding:25px;color:var(--muted);text-align:center}.explorer{display:grid;grid-template-columns:minmax(240px,370px) 1fr;gap:24px}.board{display:grid;grid-template-columns:repeat(8,1fr);aspect-ratio:1;border:1px solid #80939d}.square{aspect-ratio:1;display:grid;place-items:center;font-size:clamp(24px,4.2vw,43px);line-height:1;color:#151c20;text-shadow:0 1px white}.light{background:#e8edf0}.dark{background:#97aeb7}.mark{box-shadow:inset 0 0 0 3px #cc922b}.controls{display:flex;gap:8px;align-items:center;margin-bottom:16px}button,select{font:inherit;border:1px solid #b8c8cf;border-radius:6px;background:white;color:var(--ink);padding:7px 11px}select{min-width:0;max-width:100%;flex:1}.mono{font:12px ui-monospace,monospace;overflow-wrap:anywhere}.badge{display:inline-block;padding:3px 8px;border-radius:5px;background:#eaf2f5}.facts{display:grid;grid-template-columns:1fr 1fr;gap:12px}.fact{background:#f4f7f9;border-radius:7px;padding:12px}.fact span{display:block;font-size:12px;color:var(--muted)}.fact strong{font-size:19px}header{display:flex;justify-content:space-between;gap:20px;align-items:start}.move-human{color:#94611e}.move-raw{color:var(--blue)}.move-search{color:var(--orange)}@media(max-width:760px){.grid,.explorer{grid-template-columns:1fr}header{display:block}h1{font-size:28px}.board{max-width:370px}.square{font-size:38px}main{padding:24px 14px}}
</style></head><body><main>
<header><div><div class="eyebrow">Human games / policy learning / MCTS</div><h1>Learning Hikaru's moves, then searching</h1><p id="run-label" class="muted"></p><p id="status"></p></div><button id="download">Download report data</button></header>
<div id="warnings" class="note hidden section"></div>
<section class="section panel"><h2>Which checkpoint was selected?</h2><div id="selection-facts" class="facts"></div><p id="continuation-note" class="caption"></p><p id="stopping-note" class="caption"></p></section>
<section class="section panel"><h2>What the model was taught</h2><p id="supervision"></p><p class="caption">The policy learns only the selected player's own moves. The value head learns observed outcomes from both sides; these include human mistakes, resignations, and clock effects, and are not perfect-play evaluations.</p><div id="dataset-table" class="table-wrap"></div><p id="split-note" class="caption"></p></section>
<section class="section"><h2>Is the model learning or overfitting?</h2><p class="muted">Training and full validation metrics below use all 4,480 action outputs. They are separate from the legal-move-conditioned sampled test metrics farther down.</p><p class="caption">Each checkpoint is evaluated on the full training and validation splits with its weights held fixed. A decreasing training NLL alongside a rising validation NLL can indicate overfitting. A train/validation gap alone is not proof: the chronological splits also contain different games.</p><div id="overfit-note" class="note"></div><div class="grid section"><div class="panel"><h3>Move prediction at each checkpoint</h3><p class="caption">Full-action negative log likelihood on target-player turns. Lower is better.</p><div id="validation-chart" class="chart"></div></div><div class="panel"><h3>Checkpoint move agreement</h3><p class="caption">Top-one accuracy across all network actions; higher is better.</p><div id="agreement-chart" class="chart"></div></div><div class="panel"><h3>Validation minus training NLL</h3><p class="caption">Only frozen-checkpoint evaluations enter this gap.</p><div id="gap-chart" class="chart"></div></div><div class="panel"><h3>Outcome prediction at each checkpoint</h3><p class="caption">Both players' observed outcomes; compare each split with its own always-zero baseline.</p><div id="value-fit-chart" class="chart"></div></div></div><div id="validation-table" class="panel table-wrap section"></div><div id="value-fit-table" class="panel table-wrap section"></div><details class="section panel"><summary>Inspect within-epoch optimizer losses</summary><p class="caption">These measurements and epoch averages use changing weights as training proceeds. They are not the full training-set evaluation of the final checkpoint and must not be used to calculate the generalization gap.</p><div id="training-chart" class="chart"></div><div id="epoch-average-table" class="table-wrap"></div></details></section>
<section class="section panel"><h2>Same model, with and without MCTS</h2><p class="muted">Both modes use the same checkpoint and positions within each evaluation. MCTS gets additional computation; this is not an equal-time comparison.</p><div id="comparison-table" class="table-wrap"></div><p id="sample-note" class="caption"></p><p class="caption">Move agreement measures imitation, not move quality. Raw NLL here is conditioned on legal moves. Search uses deterministic move selection without exploration noise. Search top-five ties use action value then prior.</p></section>
<section class="section grid"><div class="panel"><h2>Fixed tactical checks</h2><div id="tactical-table" class="table-wrap"></div><p class="caption">Immediate mates and hanging-queen captures; a tiny diagnostic set, not an Elo estimate.</p></div><div class="panel"><h2>Color-paired matches</h2><div id="matches-table" class="table-wrap"></div><p class="caption">MCTS and raw policy swap colors from each fixed opening. Cutoffs are unresolved and excluded from the completed-game score. A small deterministic match set cannot establish a rating.</p></div></section>
<section class="section panel"><h2>Inspect a held-out decision</h2><p class="caption">Position before Hikaru's recorded move. Gold squares mark that move's start and destination. Every sampled position preserves the preceding game's repetition history.</p><div class="controls"><button id="previous" aria-label="Previous position">←</button><select id="position-select" aria-label="Held-out position"></select><button id="next" aria-label="Next position">→</button></div><div id="position-empty" class="empty hidden">No held-out decisions recorded yet.</div><div id="explorer" class="explorer"><div><div id="board" class="board" role="img" aria-label="Chess position"></div><p id="fen" class="mono"></p></div><div><p id="position-label"></p><div id="position-facts" class="facts"></div><div id="move-table" class="table-wrap section"></div></div></div></section>
<section class="section panel"><h2>Run settings</h2><pre id="config" class="mono"></pre></section>
</main><script id="human-data" type="application/json">__HUMAN_DATA__</script><script>
"use strict";
const data=JSON.parse(document.getElementById("human-data").textContent),$=id=>document.getElementById(id),finite=x=>typeof x==="number"&&Number.isFinite(x);
const number=(x,d=3)=>finite(x)?x.toFixed(d):"—",pct=x=>finite(x)?(100*x).toFixed(1)+"%":"—",count=x=>finite(x)?x.toLocaleString():"—",date=x=>finite(x)?new Date(x*1000).toISOString().slice(0,10):"—";
const text=(tag,value,cls)=>{const e=document.createElement(tag);e.textContent=String(value);if(cls)e.className=cls;return e;};
function table(id,headers,rows){const host=$(id);host.replaceChildren();if(!rows.length){host.append(text("div","No measurements yet.","empty"));return;}const t=document.createElement("table"),head=document.createElement("thead"),tr=document.createElement("tr");for(const label of headers)tr.append(text("th",label));head.append(tr);t.append(head);const body=document.createElement("tbody");for(const row of rows){const line=document.createElement("tr");for(const cell of row)line.append(text("td",cell??"—"));body.append(line);}t.append(body);host.append(t);}
$("run-label").textContent=data.run_name+" · "+(data.config.started_at||"Start time not recorded");$("status").append(text("span","Training: "+(data.summary.status||"not recorded"),"badge"),document.createTextNode(" · "+count(data.summary.optimizer_steps)+" optimizer updates"));
const learning=data.learning,selection=$("selection-facts");for(const[name,value]of[["Starting checkpoint epoch",count(learning.start_epoch)],["Completed epoch / planned limit",count(learning.completed_epoch)+" / "+count(learning.planned_final_epoch)],["Selected best epoch",count(learning.best_epoch)],["Best validation NLL",number(learning.best_validation_nll,4)]]){const cell=text("div","","fact");cell.append(text("span",name),text("strong",value));selection.append(cell);}
$("continuation-note").textContent=(finite(learning.start_epoch)&&learning.start_epoch>0?"Continued from epoch "+learning.start_epoch+"; epoch labels are absolute across runs. ":"Started from a new initialization. ")+count(learning.epochs_completed)+" additional epochs completed in this run. The selected best checkpoint can precede the last completed epoch.";
const stoppingState=learning.stopped_early===true?"Stopped early":data.summary.status==="running"?"Monitoring":learning.stopped_early===false?"Did not stop early":"Early-stop status not recorded";
$("stopping-note").textContent=stoppingState+(learning.stop_reason?": "+String(learning.stop_reason).replaceAll("_"," "):".")+" Patience: "+count(learning.patience)+" epochs; minimum progress: "+number(learning.min_delta,4)+" NLL; epochs without that progress: "+count(learning.epochs_without_improvement)+". Best-checkpoint selection uses the lowest recorded validation NLL; the patience threshold controls when to stop, not which model to retain.";
$("overfit-note").textContent=!learning.train_evaluation_available?"Full training-set checkpoint evaluations were not recorded. Validation is shown, but a train/validation gap cannot be inferred from within-epoch average losses.":learning.divergence_epochs.length?"At epoch(s) "+learning.divergence_epochs.join(", ")+", training NLL fell while validation NLL rose relative to the preceding evaluated checkpoint. This is a possible overfitting signal; use validation to select the checkpoint and keep the test set for the final comparison.":"No adjacent recorded checkpoints show training NLL falling while validation NLL rises. This does not establish the absence of overfitting; inspect the curves and keep the final test comparison separate from checkpoint selection.";
if(data.warnings.length){$("warnings").classList.remove("hidden");$("warnings").textContent=data.warnings.join("\n");}
$("supervision").textContent=count(data.dataset.accepted_games)+" accepted games · "+(data.config.policy_supervision||data.dataset.policy_supervision||"target-player policy supervision")+".";
table("dataset-table",["Split","Games","All value positions","Player policy positions","UTC date range"],Object.entries(data.dataset.splits||{}).map(([name,s])=>[name,count(s.games),count(s.positions),count(s.policy_positions),date(s.first_end_time)+" → "+date(s.last_end_time)]));
$("split-note").textContent="Split method: "+(data.dataset.split?.strategy||"not recorded")+". Whole games and UTC dates stay together. Rejected input records: "+JSON.stringify(data.dataset.rejected_counts||{});
function chart(id,rows,series,{xKey="step",xLabel="Optimizer step",signed=false,percentage=false}={}){const host=$(id);host.replaceChildren();const points=rows.filter(r=>finite(r[xKey])&&series.some(s=>finite(r[s.key])));if(!points.length){host.append(text("div","No measurements yet.","empty"));return;}const width=520,height=240,pad={l:50,r:14,t:15,b:32};let xmin=Infinity,xmax=-Infinity,ymin=0,ymax=.01;for(const row of points){xmin=Math.min(xmin,row[xKey]);xmax=Math.max(xmax,row[xKey]);for(const s of series)if(finite(row[s.key])){if(signed)ymin=Math.min(ymin,row[s.key]);ymax=Math.max(ymax,row[s.key]);}}xmax=Math.max(xmin+1,xmax);ymin*=1.08;ymax*=1.08;const active=series.filter(s=>points.some(r=>finite(r[s.key])));
const xp=x=>pad.l+(x-xmin)/(xmax-xmin)*(width-pad.l-pad.r),yp=y=>height-pad.b-(y-ymin)/(ymax-ymin)*(height-pad.t-pad.b),ns="http://www.w3.org/2000/svg",svg=document.createElementNS(ns,"svg");svg.setAttribute("viewBox",`0 0 ${width} ${height}`);svg.setAttribute("role","img");svg.setAttribute("aria-label",active.map(s=>s.label).join(" and ")+" by "+xLabel.toLowerCase());
function el(tag,attrs,label){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(label!==undefined)e.textContent=label;svg.append(e);return e;}
for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4;el("line",{x1:pad.l,x2:width-pad.r,y1:yp(y),y2:yp(y),stroke:"#dce5e9"});el("text",{x:pad.l-8,y:yp(y)+4,"text-anchor":"end","font-size":11,fill:"#5a6e7b"},percentage?pct(y):y.toFixed(signed?2:1));}
for(const x of [xmin,Math.round((xmin+xmax)/2),xmax])el("text",{x:xp(x),y:height-10,"text-anchor":"middle","font-size":11,fill:"#5a6e7b"},String(x));
active.forEach(s=>{const valid=points.filter(r=>finite(r[s.key]));el("polyline",{points:valid.map(r=>xp(r[xKey])+","+yp(r[s.key])).join(" "),fill:"none",stroke:s.color,"stroke-width":2,"stroke-dasharray":s.dash||"none"});if(valid.length<50)valid.forEach(r=>el("circle",{cx:xp(r[xKey]),cy:yp(r[s.key]),r:3,fill:s.color}));});host.append(svg);const legend=text("div","","legend");for(const s of active){const item=text("span",s.label);item.style.setProperty("--series",s.color);legend.append(item);}host.append(legend,text("p","Horizontal axis: "+xLabel.toLowerCase(),"caption"));}
chart("training-chart",data.updates,[{key:"policy_loss",label:"Policy cross-entropy",color:"#247a9b"},{key:"value_loss",label:"Value MSE",color:"#ba7028"}]);
const epochs=learning.epochs,epochLabel=r=>String(r.epoch)+(r.is_start?" · start":"")+(r.epoch===learning.best_epoch?" · best":""),epochAxis={xKey:"epoch",xLabel:"Absolute epoch"};
chart("validation-chart",epochs,[{key:"train_nll",label:"Train NLL",color:"#247a9b"},{key:"validation_nll",label:"Validation NLL",color:"#ba7028",dash:"6 3"}],epochAxis);
chart("agreement-chart",epochs,[{key:"train_top1",label:"Train top-one",color:"#247a9b"},{key:"validation_top1",label:"Validation top-one",color:"#ba7028",dash:"6 3"}],{...epochAxis,percentage:true});
chart("gap-chart",epochs,[{key:"policy_nll_gap",label:"Validation − train NLL",color:"#795a95"}],{...epochAxis,signed:true});
chart("value-fit-chart",epochs,[{key:"train_value_mse",label:"Train value MSE",color:"#247a9b"},{key:"validation_value_mse",label:"Validation value MSE",color:"#ba7028",dash:"6 3"},{key:"train_zero_mse",label:"Train always-zero",color:"#708899",dash:"2 4"},{key:"validation_zero_mse",label:"Validation always-zero",color:"#9a8571",dash:"2 4"}],epochAxis);
table("validation-table",["Absolute epoch","Step","Train policy examples","Validation policy examples","Train NLL","Validation NLL","NLL gap","Train top-one","Validation top-one","Train top-five","Validation top-five"],epochs.map(r=>[epochLabel(r),r.step,count(r.training_evaluation.policy_sample_count),count(r.validation.policy_sample_count),number(r.train_nll),number(r.validation_nll),number(r.policy_nll_gap),pct(r.train_top1),pct(r.validation_top1),pct(r.training_evaluation.policy_top5_full_action),pct(r.validation.policy_top5_full_action)]));
table("value-fit-table",["Absolute epoch","Train value examples","Validation value examples","Train value MSE","Validation value MSE","Train always-zero MSE","Validation always-zero MSE"],epochs.map(r=>[epochLabel(r),count(r.training_evaluation.value_sample_count),count(r.validation.value_sample_count),number(r.train_value_mse),number(r.validation_value_mse),number(r.train_zero_mse),number(r.validation_zero_mse)]));
table("epoch-average-table",["Epoch","Within-epoch mean policy loss","Within-epoch mean value loss","Checkpoint train NLL","Checkpoint train value MSE"],epochs.filter(r=>finite(r.within_epoch_policy_loss)||finite(r.within_epoch_value_loss)).map(r=>[r.epoch,number(r.within_epoch_policy_loss),number(r.within_epoch_value_loss),number(r.train_nll),number(r.train_value_mse)]));
const phases=["initial","final"],label=p=>p==="initial"?(learning.start_epoch>0?"Before continuation":"Initial")+" · epoch "+count(learning.start_epoch):"Selected best · epoch "+count(learning.best_epoch),comparisons=[];
for(const phase of phases){const summary=data.evaluations[phase].summary;for(const subset of ["all","opening","later"]){const h=summary.heldout?.[subset];if(h?.positions)comparisons.push([label(phase)+" · "+subset,count(h.positions),pct(h.raw_top1),pct(h.search_top1),pct(h.raw_top5),pct(h.search_top5),number(h.raw_legal_nll),pct(h.move_change_fraction)]);}}
table("comparison-table",["Checkpoint / sample","Positions","Raw match","MCTS match","Raw top-five","MCTS top-five","Raw legal NLL","Move changed"],comparisons);
$("sample-note").textContent="Initial/final sampled positions match: "+(data.initial_final_samples_match===null?"awaiting both evaluations":data.initial_final_samples_match?"yes":"no")+". Opening = first 20 plies. Final sample spans "+count(data.evaluations.final.summary.sampling?.sampled_games)+" games; excluded terminal positions: "+JSON.stringify(data.evaluations.final.summary.sampling?.engine_terminal_exclusions||{})+".";
table("tactical-table",["Checkpoint","Raw policy","MCTS"],phases.filter(p=>data.evaluations[p].summary.tactical_probes).map(p=>{const t=data.evaluations[p].summary.tactical_probes;return[label(p),count(t.raw_passes)+" / "+count(t.scored_positions),count(t.search_passes)+" / "+count(t.scored_positions)];}));
table("matches-table",["Checkpoint","MCTS wins","Policy wins","Draws","Cutoffs","MCTS score, completed only"],phases.filter(p=>data.evaluations[p].match_totals.games).map(p=>{const m=data.evaluations[p].match_totals;return[label(p),m.mcts_wins,m.policy_wins,m.actual_draws,m.unresolved_cutoffs,pct(m.mcts_score_on_completed_games)+" ("+m.completed_games+" games)"];}));
const rows=data.evaluations.final.positions.length?data.evaluations.final.positions:data.evaluations.initial.positions,select=$("position-select");let selected=0;
for(let i=0;i<rows.length;i++){const option=text("option",`${i+1}. Game ${rows[i].game_id} · ply ${rows[i].ply}`);option.value=String(i);select.append(option);}
const pieces={K:"♔",Q:"♕",R:"♖",B:"♗",N:"♘",P:"♙",k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟"};
function show(){if(!rows.length){$("position-empty").classList.remove("hidden");$("explorer").classList.add("hidden");return;}selected=Math.max(0,Math.min(rows.length-1,selected));select.value=String(selected);const row=rows[selected],board=$("board");board.replaceChildren();const squares=[];for(const symbol of String(row.fen||"").split(" ")[0]){if(symbol==="/")continue;if(/[1-8]/.test(symbol))for(let n=0;n<Number(symbol);n++)squares.push("");else squares.push(symbol);}for(let i=0;i<64;i++){const r=Math.floor(i/8),c=i%8,square=String.fromCharCode(97+c)+(8-r),mark=row.human_move?.slice(0,2)===square||row.human_move?.slice(2,4)===square,cell=text("div",pieces[squares[i]]||"","square "+((r+c)%2?"dark":"light")+(mark?" mark":""));cell.setAttribute("title",square);board.append(cell);}$("fen").textContent=row.fen;$("position-label").textContent=`Game ${row.game_id} · ply ${row.ply} · ${row.phase} · recorded result ${row.result||"—"}`;const facts=$("position-facts");facts.replaceChildren();for(const[name,value]of[["Hikaru played",row.human_move],["Raw policy chose",row.raw_move],["MCTS chose",row.search_move],["Raw value / observed target",number(row.value)+" / "+number(row.target_value,0)],["Raw decision time",number(1000*row.raw_seconds,1)+" ms"],["MCTS decision time",number(1000*row.search_seconds,1)+" ms"]]){const cell=text("div","","fact");cell.append(text("span",name),text("strong",value));facts.append(cell);}table("move-table",["Mode","Top-five moves","Probability on Hikaru's move"],[["Raw policy",(row.raw_top5||[]).join(", "),pct(row.human_move_probability)],["MCTS visits",(row.search_top5||[]).join(", "),pct(row.human_move_search_probability)]]);$("previous").disabled=selected===0;$("next").disabled=selected===rows.length-1;}
select.addEventListener("change",()=>{selected=Number(select.value);show();});$("previous").addEventListener("click",()=>{selected--;show();});$("next").addEventListener("click",()=>{selected++;show();});show();
$("config").textContent=JSON.stringify({training:data.config,evaluation:data.evaluations.final.summary.config||data.evaluations.initial.summary.config||{}},null,2);
$("download").addEventListener("click",()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:"application/json"})),a=document.createElement("a");a.href=url;a.download=data.run_name+"-report.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
</script></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(generate_report(args.run_dir, args.output))


if __name__ == "__main__":
    main()
