import {readFileSync,writeFileSync} from 'node:fs';import {fileURLToPath} from 'node:url';
import init,{Sim,Scenario,selfCheckDigest} from './rustupstream/tepsim_wasm.js';
const p=fileURLToPath(new URL('.',import.meta.url));await init({module_or_path:readFileSync(p+'rustupstream/tepsim_wasm_bg.wasm')});
let v={selfCheckDigest:selfCheckDigest(),replays:[]};
for(const role of ['normal','fault']){
 const b=readFileSync(p+'runs/'+role+'/1.bin');const raw=new Float64Array(b.buffer,b.byteOffset,b.length/8);
 const m=JSON.parse(readFileSync(p+'runs/'+role+'/1.json'));const sc=Scenario.fromText(m.text.replace('every=1;','every=180;'));const sim=new Sim(sc);const grid=sim.stepChunk(960);
 if(grid.length!==960*54)throw Error('expected 960 grid records');
 let max=0,sum=0, tmax=0;for(let i=0;i<960;i++){tmax=Math.max(tmax,Math.abs(grid[i*54]*3600-((i+1)*180-1)));
  for(let j=0;j<52;j++){const dx=Math.abs(grid[i*54+1+j]-raw[i*93+41+j]);max=Math.max(max,dx);sum+=dx!==0?1:0;}}
 if(tmax>2e-6||sum!==0)throw Error('different native data or grid time');
 v.replays.push({role,run_id:1,records:960,channels:52,elements:960*52,different:sum,max_abs_error:max,clock_max_abs_error_sec:tmax});sc.free();sim.free();
}
writeFileSync(p+'native_replay.json',JSON.stringify(v,null,2));
