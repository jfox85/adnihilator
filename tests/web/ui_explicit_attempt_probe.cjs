const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8')),nodes=new Map();
let requests=0,scrolled=0,focused=0,reloads=0,partialFailure=true;
const control={disabled:false,isConnected:true,getClientRects(){return [{}]},focus(){focused++}};
function node(id){if(!nodes.has(id))nodes.set(id,{hidden:true,textContent:'',disabled:false,isConnected:true,style:{},classList:{remove(){},toggle(){},contains(){return false}},setAttribute(){},contains(){return false},focus(){focused++},getBoundingClientRect(){return {top:2000,bottom:2200}},scrollIntoView(){scrolled++}});return nodes.get(id)}
const sandbox={URL,console,window:{innerHeight:844,location:{origin:'http://fixture.invalid',pathname:'/',search:'',reload(){reloads++}}},document:{activeElement:control,addEventListener(){},getElementById:node,querySelector(){return node('toolsButton')},querySelectorAll(){return []}},setInterval(){return 1},setTimeout(){},alert(){},FormData:class{append(){}},fetch:async()=>{requests++;if(input.mode==='refresh-all-partial')return {ok:true,status:200,redirected:false,json:async()=>({results:{feed:{added:0,error:partialFailure?'synthetic refresh failure':null}}})};return {ok:false,status:input.mode==='opml'?input.opml_status:500,redirected:false}}};
vm.createContext(sandbox);vm.runInContext(input.base,sandbox);vm.runInContext(input.dashboard,sandbox);
(async()=>{
 if(input.mode==='refresh-all-partial'){
  await sandbox.refreshAll();assert.strictEqual(node('requestNotice').hidden,false);
  sandbox.dismissRequestNotice();await sandbox.refreshAll();
  assert.strictEqual(requests,2);assert.strictEqual(node('requestNotice').hidden,false,'Fresh per-feed refresh failure silently suppressed');
  assert.strictEqual(node('refreshAllBtn').disabled,false);
  partialFailure=false;await sandbox.refreshAll();
  assert.strictEqual(reloads,1);assert.strictEqual(node('requestNotice').hidden,true,'Successful fresh refresh left stale notice');
 }else if(input.mode==='opml'){
  const file={files:[{name:'fixture.opml'}],value:'file',disabled:false};
  await sandbox.importOPML(file);assert.strictEqual(node('requestNotice').hidden,false);
  sandbox.dismissRequestNotice();await sandbox.importOPML(file);
  assert.strictEqual(requests,2);assert.strictEqual(node('requestNotice').hidden,false,'Fresh malformed import silently suppressed');assert.strictEqual(file.disabled,false);
 }else{
  await sandbox.refreshPodcast('synthetic',control);assert.strictEqual(node('requestNotice').hidden,false);
  if(input.mode==='offscreen'){
   assert(scrolled>0,'Explicit failure stayed outside the viewport');
   assert(focused>0,'Error recovery is not keyboard reachable');
  }else{
   sandbox.dismissRequestNotice();await sandbox.refreshPodcast('synthetic',control);
   assert.strictEqual(requests,2);assert.strictEqual(node('requestNotice').hidden,false,'Fresh explicit action silently suppressed');assert.strictEqual(control.disabled,false);
  }
 }
 sandbox.showRequestNotice('Status is stale',{owner:'queue-poll'});sandbox.dismissRequestNotice();sandbox.showRequestNotice('Status is stale',{owner:'queue-poll'});assert.strictEqual(node('requestNotice').hidden,true,'Background fault reappeared');
})().catch(error=>{console.error(error);process.exitCode=1});
