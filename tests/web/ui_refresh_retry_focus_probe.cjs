const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8')),nodes=new Map();let requests=0,reloads=0,pending=null;
const document={activeElement:null,addEventListener(){},getElementById:node,querySelector(){return node('toolsButton')},querySelectorAll(){return []}};
function node(id){if(!nodes.has(id)){
 let hidden=true;const el={id,textContent:'',disabled:false,isConnected:true,style:{},classList:{remove(){},toggle(){},contains(){return false}},setAttribute(){},getClientRects(){return [{}]},contains(child){return id==='requestNotice'&&['requestRetry','requestDismiss','requestSignIn'].includes(child?.id)},focus(){document.activeElement=this},getBoundingClientRect(){return {top:100,bottom:200}},scrollIntoView(){}};
 Object.defineProperty(el,'hidden',{get(){return hidden},set(value){hidden=value;if(value&&el.contains(document.activeElement))document.activeElement=body}});nodes.set(id,el)
 }return nodes.get(id)}
const body={id:'BODY',isConnected:true,focus(){document.activeElement=this}};document.activeElement=node('refreshAllBtn');
const sandbox={URL,console,document,window:{innerHeight:844,location:{origin:'http://fixture.invalid',pathname:'/',search:'',reload(){reloads++}}},setInterval(){},setTimeout(){},fetch:()=>{requests++;return new Promise(resolve=>{assert.strictEqual(pending,null);pending=resolve})}};
vm.createContext(sandbox);vm.runInContext(input.base,sandbox);vm.runInContext(input.dashboard,sandbox);
async function finish(status,data){const resolve=pending;assert(resolve);pending=null;resolve({ok:status<400,status,redirected:false,json:async()=>data});await Promise.resolve();}
(async()=>{
 sandbox.showRequestNotice('Status is stale',{owner:'queue-poll'});sandbox.dismissRequestNotice();
 let work=sandbox.refreshAll();await finish(200,input.partial);await work;
 assert.strictEqual(node('requestNotice').hidden,false);assert.strictEqual(node('refreshAllBtn').disabled,false);
 node('requestRetry').focus();const label=node('requestNoticeText').textContent;
 work=node('requestRetry').onclick();assert.strictEqual(node('refreshAllBtn').disabled,true);assert.strictEqual(node('refreshAllBtn').textContent,'Refreshing...');
 assert.strictEqual(node('requestNotice').hidden,false,'Focused notice disappeared while Retry was awaiting response');
 assert.strictEqual(document.activeElement,node('requestRetry'),'Retry focus fell out of the notice in flight');assert.strictEqual(node('requestNoticeText').textContent,label);
 await sandbox.refreshAll();assert.strictEqual(requests,2,'Busy action launched a duplicate request');
 await finish(200,input.partial);await work;assert.strictEqual(document.activeElement,node('requestRetry'));assert.strictEqual(node('requestNotice').hidden,false);assert.strictEqual(node('refreshAllBtn').disabled,false);
 sandbox.dismissRequestNotice();work=sandbox.refreshAll();await finish(200,input.partial);await work;assert.strictEqual(node('requestNotice').hidden,false,'Explicit fresh partial failure remained dismissed');
 node('requestRetry').focus();work=node('requestRetry').onclick();assert.strictEqual(document.activeElement,node('requestRetry'));await finish(200,input.success);await work;assert.strictEqual(reloads,1);assert.strictEqual(node('requestNotice').hidden,true,'Successful response left stale failure feedback');
 work=sandbox.refreshAll();await finish(500,{});await work;assert.strictEqual(node('refreshAllBtn').disabled,false);assert.strictEqual(node('refreshAllBtn').textContent,'Refresh All');assert.strictEqual(node('requestNotice').hidden,false);assert.strictEqual(document.activeElement,node('requestNotice'),'500 feedback was not keyboard reachable');
 sandbox.clearRequestNotice();sandbox.showRequestNotice('Status is stale',{owner:'queue-poll'});assert.strictEqual(node('requestNotice').hidden,true,'Explicit refresh attempts erased the dismissed background fault');
})().catch(error=>{console.error(error);process.exitCode=1});
