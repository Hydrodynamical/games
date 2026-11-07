"""Offline comparison of a supervised baseline and its self-play continuation.

Training losses, held-out outcome prediction and paired playing strength remain
separate. Missing or incomplete logs are visible without requiring torch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

try:
    from .chess_engine import GameState
    from .human_report import _read, _jsonl, _finite
except ImportError:
    from chess_engine import GameState
    from human_report import _read, _jsonl, _finite


def _match_totals(records):
    # Unknown cutoffs cannot become scored draws/wins through a stale score.
    completed = [row for row in records if not row.get("truncated")
                 and row.get("termination") != "move_limit" and row.get("outcome") != "*"
                 and row.get("candidate_score") in (0, .5, 1)]
    points = sum(row["candidate_score"] for row in completed)
    unresolved = len(records) - len(completed)
    return {"games": len(records), "completed_games": len(completed), "unresolved_cutoffs": unresolved,
            "candidate_wins": sum(row["candidate_score"] == 1 for row in completed),
            "baseline_wins": sum(row["candidate_score"] == 0 for row in completed),
            "actual_draws": sum(row["candidate_score"] == .5 for row in completed),
            "candidate_score_on_completed_games": points / len(completed) if completed else None,
            "candidate_score_all_games_bounds": [points / len(records), (points + unresolved) / len(records)]
                                                  if records else None,
            "termination_counts": dict(Counter(row.get("termination", "unknown") for row in records))}


def _combined_metrics(groups):
    """Combine disjoint result groups with their original position weights."""
    groups = [group for group in groups if isinstance(group, dict) and _finite(group.get("positions"))
              and group["positions"] > 0]
    count = sum(group["positions"] for group in groups)
    result = {"positions": count, "games": sum(group.get("games", 0) for group in groups)}
    for model in ("baseline", "candidate", "zero"):
        result[model] = {}
        for metric in ("mse", "mae"):
            values = [group.get(model, {}).get(metric) for group in groups]
            result[model][metric] = (sum(value * group["positions"] for value, group in zip(values, groups)) / count
                                     if count and all(_finite(value) for value in values) else None)
    return result


def _match_boards(records, warnings):
    boards = []
    for record in records:
        game = dict(record)
        try:
            state = GameState.from_fen(game["initial_fen"])
            frames = [state.to_fen()]
            for move in game["moves"]:
                state.push_uci(move)
                frames.append(state.to_fen())
            if game.get("final_fen") is not None and game["final_fen"] != frames[-1]:
                raise ValueError("final FEN differs from move history")
            game["frames"] = frames
        except (KeyError, TypeError, ValueError, IndexError, AssertionError) as exc:
            warnings.append(f"Cannot replay match {game.get('game_id', '?')}: {exc}")
            game["frames"] = []
        boards.append(game)
    return boards


def generate_report(run_dir, output=None):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    destination = Path(output).resolve() if output else run_dir / "selfplay_report.html"
    warnings = []
    metadata = _read(run_dir / "config.json", {}, warnings)
    config = metadata.get("config", metadata)
    config = config if isinstance(config, dict) else {}
    comparison_dir = run_dir / "comparison"
    comparison = _read(comparison_dir / "summary.json", {}, warnings)
    comparison_config = _read(comparison_dir / "config.json", {}, warnings) or comparison.get("config", {})
    values = comparison.get("values", {})
    values = values if isinstance(values, dict) else {}
    matches = _jsonl(comparison_dir / "matches.jsonl", warnings)
    results = values.get("by_game_result", {})
    results = results if isinstance(results, dict) else {}
    outcomes = values.get("by_game_outcome", {})
    outcomes = outcomes if isinstance(outcomes, dict) else {}
    breakdowns = [dict(label="All completed games", metrics=values.get("all", {})),
                  dict(label="Decisive games", metrics=outcomes.get("decisive") or _combined_metrics([results.get("1-0"), results.get("0-1")])),
                  dict(label="Actual draws", metrics=outcomes.get("draw") or results.get("1/2-1/2", {}))]
    curve = _read(run_dir / "value_curve.json", [], warnings)
    curve = sorted([row for row in curve if isinstance(row, dict) and _finite(row.get("game"))
                    and isinstance(row.get("metrics"), dict)], key=lambda row: row["game"])
    retention = _read(run_dir / "human_retention.json", {}, warnings)
    data = {"run_name": run_dir.name, "metadata": metadata, "config": config,
            "summary": _read(run_dir / "summary.json", {}, warnings),
            "updates": _jsonl(run_dir / "updates.jsonl", warnings),
            "probes": _jsonl(run_dir / "probes.jsonl", warnings),
            "comparison": comparison, "comparison_config": comparison_config,
            "values": values, "value_breakdowns": breakdowns, "value_curve": curve,
            "human_retention": retention,
            "matches": _match_boards(matches, warnings),
            "match_totals": _match_totals(matches) if matches else comparison.get("matches", {}),
            "warnings": warnings,
            "training_report_href": "./" + quote(os.path.relpath(run_dir / "report.html", destination.parent), safe="/")}
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    document = TEMPLATE.replace("__SELFPLAY_DATA__", payload)
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
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chess self-play experiment</title><style>
:root{color-scheme:light;--ink:#20323e;--muted:#5a6e7b;--line:#dce5e9;--blue:#247a9b;--orange:#ba7028}*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:var(--ink);font:15px/1.55 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:32px 24px 70px}h1{font-size:34px;line-height:1.2;margin:6px 0 12px}h2{font-size:22px;margin:0 0 8px}h3{font-size:17px;margin:0 0 8px}p{margin:6px 0 14px}a{color:var(--blue)}.caption,.muted{color:var(--muted)}.caption{font-size:13px}.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--blue)}.section{margin-top:24px}.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.note{border-left:4px solid var(--orange);background:#fff6e9;padding:12px 16px;white-space:pre-wrap}.hidden{display:none}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 10px;white-space:nowrap}th{font-size:12px;color:var(--muted)}td:first-child,th:first-child{padding-left:0}.chart svg{width:100%;height:auto;display:block}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px}.legend span::before{content:'●';color:var(--series);margin-right:5px}.empty{padding:25px;color:var(--muted);text-align:center}.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.fact{background:#f4f7f9;border-radius:7px;padding:12px}.fact span{display:block;font-size:12px;color:var(--muted)}.fact strong{font-size:20px}.controls{display:flex;gap:8px;align-items:center;margin-bottom:16px;flex-wrap:wrap}button,select{font:inherit;border:1px solid #b8c8cf;border-radius:6px;background:white;color:var(--ink);padding:7px 11px}select{min-width:0;max-width:100%;flex:1}.explorer{display:grid;grid-template-columns:minmax(240px,370px) 1fr;gap:24px}.board{display:grid;grid-template-columns:repeat(8,1fr);aspect-ratio:1;border:1px solid #80939d}.square{aspect-ratio:1;display:grid;place-items:center;font-size:clamp(24px,4.2vw,43px);line-height:1;color:#151c20;text-shadow:0 1px white}.light{background:#e8edf0}.dark{background:#97aeb7}.mark{box-shadow:inset 0 0 0 3px #cc922b}.mono{font:12px ui-monospace,monospace;overflow-wrap:anywhere;white-space:pre-wrap}header{display:flex;justify-content:space-between;gap:20px;align-items:start}@media(max-width:760px){.grid,.explorer{grid-template-columns:1fr}.facts{grid-template-columns:1fr 1fr}header{display:block}h1{font-size:28px}.board{max-width:370px}.square{font-size:38px}main{padding:24px 14px}}
</style></head><body><main>
<header><div><div class="eyebrow">Supervised checkpoint → MCTS self-play</div><h1>Did self-play improve the model?</h1><p id="run-label" class="muted"></p><p id="status"></p></div><button id="download">Download report data</button></header>
<div id="warnings" class="note hidden section"></div>
<section class="section panel"><h2>Experiment and checkpoint choice</h2><div id="training-facts" class="facts"></div><p id="selection-note" class="caption"></p><p id="training-note" class="caption"></p><p><a id="training-report" href="report.html">Open detailed training inspector</a></p></section>
<section class="section panel"><h2>Held-out outcome prediction</h2><p class="muted">Both frozen networks see the same positions in independent games generated by the supervised baseline. Lower MSE and MAE are better.</p><div id="value-table" class="table-wrap"></div><p id="value-note" class="caption"></p><div id="value-breakdown" class="table-wrap section"></div><p id="uncertainty-note" class="caption"></p><p class="caption">Targets are observed outcomes under the frozen benchmark policy, from the side to move. They are not perfect-play values. Unknown move-limit outcomes are excluded from this evaluation; genuine draws remain included. Training continues to use the separately recorded cutoff-label policy.</p></section>
<section class="section panel"><h2>Value estimates across self-play</h2><div class="grid"><div><h3>Mean squared error</h3><div id="value-chart" class="chart"></div></div><div><h3>Mean absolute error</h3><div id="mae-chart" class="chart"></div></div></div><div id="curve-table" class="table-wrap"></div><p class="caption">Each point evaluates a frozen checkpoint on the same held-out benchmark. Intermediate checkpoints diagnose the learning trajectory; the candidate is chosen by the fixed training budget, not by this curve. Training-batch loss alone does not establish improved prediction or playing strength.</p></section>
<section class="section panel"><h2>Equal-search playing comparison</h2><p id="search-note" class="caption"></p><div id="match-table" class="table-wrap"></div><p id="bounds-note" class="caption"></p><p class="caption">Colors are swapped for each fixed opening. Completed-game score excludes unresolved cutoffs. The all-game bounds show the extremes if every unresolved game were a loss or a win; cutoffs receive no draw score. This small paired experiment does not estimate Elo.</p></section>
<section class="section grid"><div class="panel"><h2>Human-game retention</h2><div id="retention-table" class="table-wrap"></div><p class="caption">Full-action move prediction on the same held-out human split; value MSE uses observed human outcomes. This measures retained imitation, separately from self-play outcome prediction and playing strength.</p></div><div class="panel"><h2>Fixed tactical diagnostics</h2><div id="probe-table" class="table-wrap"></div><p class="caption">Tiny mate/capture checks can be solved by search itself. They are observations, not a strength rating.</p></div></section>
<section class="section panel"><h2>Inspect a paired match</h2><div class="controls"><select id="match-select" aria-label="Match"></select><button id="previous" aria-label="Previous ply">←</button><button id="next" aria-label="Next ply">→</button><button id="last">Final position</button></div><p id="match-empty" class="empty hidden">No replayable match histories yet.</p><div id="explorer" class="explorer"><div><div id="board" class="board" role="img" aria-label="Chess position"></div><p id="fen" class="mono"></p></div><div><p id="match-label"></p><p id="ply-label"></p><p id="moves" class="mono"></p></div></div></section>
<details class="section panel"><summary>Settings and provenance</summary><pre id="config" class="mono"></pre></details>
</main><script id="selfplay-data" type="application/json">__SELFPLAY_DATA__</script><script>
"use strict";
const data=JSON.parse(document.getElementById("selfplay-data").textContent),$=id=>document.getElementById(id),finite=x=>typeof x==="number"&&Number.isFinite(x);
const num=(x,d=3)=>finite(x)?x.toFixed(d):"—",pct=x=>finite(x)?(100*x).toFixed(1)+"%":"—",count=x=>finite(x)?x.toLocaleString():"—";
const text=(tag,value,cls)=>{const e=document.createElement(tag);e.textContent=String(value);if(cls)e.className=cls;return e;};
function table(id,headers,rows){const host=$(id);host.replaceChildren();if(!rows.length){host.append(text("div","No measurements yet.","empty"));return;}const t=document.createElement("table"),head=document.createElement("thead"),tr=document.createElement("tr");headers.forEach(h=>tr.append(text("th",h)));head.append(tr);t.append(head);const body=document.createElement("tbody");for(const row of rows){const line=document.createElement("tr");row.forEach(cell=>line.append(text("td",cell??"—")));body.append(line);}t.append(body);host.append(t);}
$("run-label").textContent=data.run_name;$("status").textContent="Training: "+(data.summary.status||"not recorded")+" · comparison: "+(data.comparison.status||"not recorded");$("training-report").href=data.training_report_href;
if(data.warnings.length){$("warnings").classList.remove("hidden");$("warnings").textContent=data.warnings.join("\n");}
for(const[name,value]of[["Self-play games",count(data.summary.games_completed)+" / "+count(data.config.iterations)],["Optimizer updates",count(data.summary.optimizer_steps)],["Move-limit games",count(data.summary.move_limit_games)+" ("+pct(data.summary.move_limit_fraction)+")"],["Actual draws",count(data.summary.actual_draws)],["Decisive games",count(data.summary.decisive_games)],["Cutoff positions in training",pct(data.summary.cutoff_target_fraction)]]){const cell=text("div","","fact");cell.append(text("span",name),text("strong",value));$("training-facts").append(cell);}
$("selection-note").textContent="Baseline: frozen supervised checkpoint. Candidate: the preselected final checkpoint after "+count(data.config.iterations)+" self-play games. Intermediate benchmark results do not select the candidate. "+(data.summary.status==="completed"?"Training is complete.":"The run is incomplete; final evidence may not yet be available.");
$("training-note").textContent="Fresh optimizer and empty replay at the start of the RL stage. Both colors learn normalized MCTS visits and observed outcomes. Training cutoff policy: "+(data.config.cutoff_value_policy||data.summary.cutoff_value_policy||"not recorded")+". "+count(data.config.sims)+" simulations/ply, up to "+count(data.config.max_moves)+" plies/game, "+count(data.config.updates)+" updates/game.";
const values=data.values,all=values.all||{},models=[["baseline","Supervised baseline"],["candidate","After self-play"],["zero","Always zero"]];
table("value-table",["Model","MSE","MAE","Mean |prediction|","Positions","Games"],finite(all.positions)?models.map(([key,label])=>[label,num(all[key]?.mse,4),num(all[key]?.mae,4),num(key==="zero"?0:all[key]?.mean_absolute_prediction,4),count(all.positions),count(all.games)]):[]);
$("value-note").textContent="Scored "+count(values.completed_games)+" of "+count(values.games)+" benchmark games; excluded "+count(values.excluded_unknown_outcome_games)+" unfinished games / "+count(values.excluded_unknown_outcome_positions)+" positions. Candidate minus baseline MSE: "+num(all.candidate_minus_baseline_mse,4)+"; MAE: "+num(all.candidate_minus_baseline_mae,4)+" (negative favors self-play).";
table("value-breakdown",["Outcome population","Games","Positions","Baseline MSE","Candidate MSE","Zero MSE","Baseline MAE","Candidate MAE"],data.value_breakdowns.filter(r=>finite(r.metrics?.positions)&&r.metrics.positions>0).map(r=>{const m=r.metrics;return[r.label,count(m.games),count(m.positions),num(m.baseline?.mse,4),num(m.candidate?.mse,4),num(m.zero?.mse,4),num(m.baseline?.mae,4),num(m.candidate?.mae,4)];}));
const interval=x=>Array.isArray(x)&&x.length===2?"["+num(x[0],4)+", "+num(x[1],4)+"]":"unavailable",boot=values.bootstrap_game_level||{};
$("uncertainty-note").textContent="Paired whole-game bootstrap, 95% descriptive intervals for candidate minus baseline: MSE "+interval(boot.candidate_minus_baseline_mse)+"; MAE "+interval(boot.candidate_minus_baseline_mae)+". "+count(boot.scored_games)+" scored games, "+count(boot.samples)+" resamples. This accounts for within-game clustering and is not a playing-strength interval.";
function chart(id,rows,metric){const host=$(id),points=rows.filter(r=>finite(r.game)&&finite(r.metrics?.all?.candidate?.[metric]));if(!points.length){host.append(text("div","No measurements yet.","empty"));return;}const width=520,height=235,pad={l:50,r:15,t:15,b:32},xmin=Math.min(...points.map(r=>r.game)),xmax=Math.max(xmin+1,...points.map(r=>r.game));let ymax=.01;for(const r of points)for(const key of ["baseline","candidate","zero"])if(finite(r.metrics.all[key]?.[metric]))ymax=Math.max(ymax,r.metrics.all[key][metric]);ymax*=1.08;const xp=x=>pad.l+(x-xmin)/(xmax-xmin)*(width-pad.l-pad.r),yp=y=>height-pad.b-y/ymax*(height-pad.t-pad.b),ns="http://www.w3.org/2000/svg",svg=document.createElementNS(ns,"svg");svg.setAttribute("viewBox",`0 0 ${width} ${height}`);svg.setAttribute("role","img");svg.setAttribute("aria-label",metric.toUpperCase()+" by self-play games");const el=(tag,attrs,label)=>{const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(label!==undefined)e.textContent=label;svg.append(e);};for(let i=0;i<=4;i++){const y=ymax*i/4;el("line",{x1:pad.l,x2:width-pad.r,y1:yp(y),y2:yp(y),stroke:"#dce5e9"});el("text",{x:pad.l-8,y:yp(y)+4,"text-anchor":"end","font-size":11,fill:"#5a6e7b"},y.toFixed(2));}for(const x of [xmin,Math.round((xmin+xmax)/2),xmax])el("text",{x:xp(x),y:height-10,"text-anchor":"middle","font-size":11,fill:"#5a6e7b"},String(x));const series=[["candidate","After self-play","#247a9b","none"],["baseline","Supervised baseline","#ba7028","6 3"],["zero","Always zero","#708899","2 4"]],legend=text("div","","legend");for(const[key,label,color,dash]of series){const valid=points.filter(r=>finite(r.metrics.all[key]?.[metric]));el("polyline",{points:valid.map(r=>xp(r.game)+","+yp(r.metrics.all[key][metric])).join(" "),fill:"none",stroke:color,"stroke-width":2,"stroke-dasharray":dash});valid.forEach(r=>el("circle",{cx:xp(r.game),cy:yp(r.metrics.all[key][metric]),r:3,fill:color}));const item=text("span",label);item.style.setProperty("--series",color);legend.append(item);}host.append(svg,legend,text("p","Horizontal axis: completed self-play games","caption"));}
chart("value-chart",data.value_curve,"mse");chart("mae-chart",data.value_curve,"mae");table("curve-table",["Games trained","Scored positions","Candidate MSE","Baseline MSE","Zero MSE","Candidate MAE","Δ MSE"],data.value_curve.map(r=>{const m=r.metrics.all||{};return[r.game,count(m.positions),num(m.candidate?.mse,4),num(m.baseline?.mse,4),num(m.zero?.mse,4),num(m.candidate?.mae,4),num(m.candidate_minus_baseline_mse,4)];}));
const match=data.match_totals||{},bounds=match.candidate_score_all_games_bounds,comparison=data.comparison_config||{};table("match-table",["Candidate wins","Baseline wins","Actual draws","Unresolved","Completed-game candidate score"],finite(match.games)?[[count(match.candidate_wins),count(match.baseline_wins),count(match.actual_draws),count(match.unresolved_cutoffs),pct(match.candidate_score_on_completed_games)+" ("+count(match.completed_games)+" games)"]]:[]);$("bounds-note").textContent="Candidate score bounds across all "+count(match.games)+" games: "+(Array.isArray(bounds)?pct(bounds[0])+" to "+pct(bounds[1]):"unavailable")+".";$("search-note").textContent="Both models: "+count(comparison.sims)+" simulations per decision; deterministic selection, no exploration noise, fresh search roots. Budget: "+count(comparison.max_plies_after_opening)+" plies after each opening; "+count(comparison.match_pairs)+" color-swapped pairs. Equal simulations do not imply identical wall-clock time.";
const retentionRows=[];for(const[key,label]of models.slice(0,2)){const entry=data.human_retention[key];if(entry){const m=entry.metrics||entry;retentionRows.push([label,count(m.policy_sample_count),num(m.policy_nll_full_action,4),pct(m.policy_top1_full_action),pct(m.policy_top5_full_action),num(m.value_mse,4)]);}}table("retention-table",["Model","Policy positions","Policy NLL","Top-one","Top-five","Value MSE"],retentionRows);
table("probe-table",["Games trained","Search passed","Scored probes"],data.probes.map(row=>{const positions=(row.positions||[]).filter(p=>p.passed!==null&&p.passed!==undefined);return[row.game,positions.filter(p=>p.passed).length,positions.length];}));
const games=data.matches.filter(g=>Array.isArray(g.frames)&&g.frames.length),select=$("match-select");let selected=0,ply=0;games.forEach((g,i)=>{const option=text("option",`${i+1}. ${g.opening||g.game_id} · candidate ${g.candidate_color} · ${g.outcome}`);option.value=String(i);select.append(option);});const pieces={K:"♔",Q:"♕",R:"♖",B:"♗",N:"♘",P:"♙",k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟"};function show(){if(!games.length){$("match-empty").classList.remove("hidden");$("explorer").classList.add("hidden");$("previous").disabled=$("next").disabled=$("last").disabled=true;return;}const game=games[selected];ply=Math.max(0,Math.min(ply,game.frames.length-1));select.value=String(selected);const fen=game.frames[ply],last=ply?game.moves[ply-1]:null,squares=[];for(const symbol of fen.split(" ")[0]){if(symbol==="/")continue;if(/[1-8]/.test(symbol))for(let n=0;n<Number(symbol);n++)squares.push("");else squares.push(symbol);}const board=$("board");board.replaceChildren();for(let i=0;i<64;i++){const row=Math.floor(i/8),col=i%8,square=String.fromCharCode(97+col)+(8-row),mark=last&&(last.slice(0,2)===square||last.slice(2,4)===square),cell=text("div",pieces[squares[i]]||"","square "+((row+col)%2?"dark":"light")+(mark?" mark":""));cell.setAttribute("title",square);board.append(cell);}$("fen").textContent=fen;$("match-label").textContent=`${game.game_id} · candidate ${game.candidate_color} · ${game.termination} · ${game.outcome}`;$("ply-label").textContent=`Ply ${ply} / ${game.frames.length-1}${last?" · last move "+last:" · initial position"}`;$("moves").textContent=game.moves.map((move,index)=>`${index+1}: ${move}`).join("  ");$("previous").disabled=ply===0;$("next").disabled=ply===game.frames.length-1;}
select.addEventListener("change",()=>{selected=Number(select.value);ply=0;show();});$("previous").addEventListener("click",()=>{ply--;show();});$("next").addEventListener("click",()=>{ply++;show();});$("last").addEventListener("click",()=>{if(games.length)ply=games[selected].frames.length-1;show();});show();$("config").textContent=JSON.stringify({training:data.metadata,comparison:data.comparison_config,human_retention:data.human_retention.config||{}},null,2);$("download").addEventListener("click",()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:"application/json"})),a=document.createElement("a");a.href=url;a.download=data.run_name+"-selfplay.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
</script></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(generate_report(args.run_dir, args.output))


if __name__ == "__main__":
    main()
