import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import init,{Sim,Scenario,selfCheckDigest,version} from './rustupstream/tepsim_wasm.js';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('.',import.meta.url));
await init({module_or_path:readFileSync(root+'rustupstream/tepsim_wasm_bg.wasm')});
if(selfCheckDigest()!=='c8a26889992f1719')throw Error('upstream self-check failed');
const input=JSON.parse(readFileSync(root+'protocol.json'));
const which=process.argv[2], workers=Number(process.argv[3]||1), worker=Number(process.argv[4]||0);
const cases=input.cases.filter(x=>x.role===which);
for (let n=worker;n<cases.length;n+=workers){
 const c=cases[n],scenario=Scenario.fromText('tepsim.scenario.v1;seed='+c.seed+';hours=48;step=2.777777777777778e-4;every=1;faults=;controlled=1;idv12=0;trip=0;continuous=0;integrator=euler;events='+(c.role==='fault'?'8:start:9':''));
 const sim=new Sim(scenario);let offset=0,record=0;
 const raw=new Float64Array(960*93);let onsetHi=null; const faultGrid=[];
 if(sim.rowWidth!==54||sim.totalSteps!==172800) throw Error('schema');
 let prev=null;
 while(!sim.isFinished) {
   const vals=sim.stepChunk(180);if(vals.length===0&&sim.isFinished)break;
   if(vals.length!==180*54)throw Error('wrong chunk');
   const i=178*54,j=179*54,k=record*93;
   raw.set(vals.slice(i+1,i+42),k);  // previous evaluation: actual input to just-fired fast controller
   raw.set(vals.slice(j+1,j+42),k+41); // current evaluation: standard Rust output
   raw.set(vals.slice(j+42,j+53),k+82); // held own valve commands 1-11, identical in both records
   ++record; offset+=180;
   if((sim.activeFaults.length>0)!==(c.role==='fault'&&record>160)) throw Error('fault schedule: '+record+':'+sim.activeFaults);
 }
 if(record!==960||offset!==172800||sim.outcome!=='completed')throw Error('incomplete/tripped '+record+' '+sim.outcome);
 mkdirSync(root+'runs/'+c.role,{recursive:true});
 writeFileSync(root+'runs/'+c.role+'/'+c.id+'.bin',Buffer.from(raw.buffer));
 writeFileSync(root+'runs/'+c.role+'/'+c.id+'.json',JSON.stringify({seed:c.seed,role:c.role,rows:record,outcome:sim.outcome,version:version(),digest:scenario.digest(),checksum:sim.checksum(),text:scenario.text}));
 sim.free();scenario.free();
}
