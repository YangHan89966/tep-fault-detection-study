/* Read-only boundary/status audit; one repeated fault run plus two off-panel positive controls. */
import {readFileSync,writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import init,{Sim,Scenario,selfCheckDigest} from './rustupstream/tepsim_wasm.js';
const p=fileURLToPath(new URL('.',import.meta.url));
await init({module_or_path:readFileSync(p+'rustupstream/tepsim_wasm_bg.wasm')});
const proto=JSON.parse(readFileSync(p+'protocol.json'));
const m={seed:proto.cases.find(c=>c.role==='fault'&&c.id===1).seed};
const txt='tepsim.scenario.v1;seed='+m.seed+';hours=48;step=2.777777777777778e-4;every=1;faults=;controlled=1;idv12=0;trip=0;continuous=0;integrator=euler;events=8:start:9';
const sc=Scenario.fromText(txt),s=new Sim(sc);
let i=0,boundary=[],stamps={},onset,previousTime;
function take(n){const rows=s.stepChunk(n);if(rows.length!==54*n)throw Error('short clock replay'); const t=rows[54*(n-1)];i+=n;return {step:i,stamp_sec:t*3600,IDV9:Array.from(s.activeFaults).includes(9),since_onset_sec:(s.hoursSinceOnset(9)??0)*3600};}
const before=take(180*160-i);stamps[160]=before.stamp_sec; if(before.IDV9)throw Error("Fault active before scheduled boundary");
for(let j=0;j<4;j++){const a=take(1);boundary.push(a);if(a.IDV9&&onset===undefined){onset=a.stamp_sec-a.since_onset_sec;}}
if(onset===undefined)throw Error('activating step not observed');
for(const j of [161,200,201,320,321,960])stamps[j]=take(180*j-i).stamp_sec;
if(!boundary[0].IDV9||Math.abs(stamps[160]-(28800-1))>1e-5)throw Error('missing fault boundary');
if (!(stamps[320]-onset<=28800 && stamps[321]-onset>28800 && stamps[160]<onset && stamps[161]>onset))throw Error('8h row mismatch');
const status={};for(const end of [0,1]){
 const c=Scenario.fromText(txt.replace('hours=48;','hours=6;').replace('every=1;','every=180;').replace('controlled=1;','controlled=0;').replace('trip=0;','trip='+end+';').replace('events=8:start:9','events='));
 const v=new Sim(c);while(!v.isFinished)v.stepChunk(32);
 status[end]={mode:'uncontrolled separate positive control; never enters study results',trip_ends_the_run:end,reported:v.outcome,emitted:v.emittedSamples,requested:v.totalSamples,trip_hours:v.tripHours,trip_cause:v.tripCause};
 if(v.outcome!=='tripped'||!(v.tripHours>0) || ((v.emittedSamples===v.totalSamples)!==(end===0)))throw Error('trip diagnostic not sensitive');
 v.free();c.free();
}
const r={default_self_check:selfCheckDigest(),fault_run_checked:1,rows_1based:true,schedule_requested_sec:28800,activation_observed_sec:onset,clock_unit:'seconds',preceding_unfaulted_step:before,boundary_steps:boundary,record_stamps_sec:stamps,activation_to_row161_sec:stamps[161]-onset,activation_to_row321_sec:stamps[321]-onset,strict_8h_last_record:320,protocol_8h_last_record:321,positive_controls:status};
if(s.outcome!=='completed'||s.tripHours!=null)throw Error('unexpected repeated target status');
s.free();sc.free();writeFileSync(p+'CLOCK_TRIP_AUDIT.json',JSON.stringify(r,null,2)+'\n');
