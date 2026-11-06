const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
let documentRowSearches=0,domWrites=0;
const nodes=new Map();
function node(id){if(!nodes.has(id))nodes.set(id,{style:{},dataset:{},classList:{add(){},remove(){}},querySelector(selector){return node(id+selector)},set innerHTML(value){domWrites++},set textContent(value){domWrites++}});return nodes.get(id)}
const rows=Array.from({length:input.count},(_,index)=>({dataset:{id:String(index),status:'pending',title:'synthetic'},style:{},classList:{add(){},remove(){}}}));
const data={auto_process:false,episodes:Object.fromEntries(rows.map(row=>[row.dataset.id,{status:'pending',progress_step:null,progress_percent:null,ads_removed_seconds:null,error_message:null}]))};
const sandbox={window:{location:{origin:'http://fixture.invalid'}},document:{hidden:false,addEventListener(){},querySelectorAll(){return rows},querySelector(){documentRowSearches++;return rows[0]},getElementById:node},setInterval(){return 1},setTimeout(){},clearInterval(){},console,fetch:async()=>({ok:true,json:async()=>data}),requestJSON:async()=>data,clearRequestNotice(){},reportRequestError(error){throw error},UIRequestError:Error};
vm.createContext(sandbox);vm.runInContext(input.source,sandbox);
(async()=>{
 await sandbox.pollStatuses();assert.strictEqual(documentRowSearches,0,'Per-episode document scan');
 assert(domWrites>=input.count,'Poll never applied server response');
 const prior=domWrites;await sandbox.pollStatuses();
 // Only the single auto-process label is allowed to be touched by a stable poll.
 assert(domWrites-prior<=1,'Unchanged rows were rewritten');
})().catch(error=>{console.error(error);process.exitCode=1});
