const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8')),nodes=new Map();let pending=null,requests=0,reloads=0;
const body={id:'BODY',isConnected:true,focus(){document.activeElement=this}};
const document={body,activeElement:body,addEventListener(){},getElementById:node,querySelector(){return node('toolsButton')},querySelectorAll(){return []}};
function node(id){if(!nodes.has(id)){
 let hidden=true,disabled=false;const el={id,textContent:'',isConnected:true,style:{},classList:{remove(){},toggle(){},contains(){return false}},setAttribute(){},getClientRects(){return [{}]},contains(child){return child===this||(id==='requestNotice'&&['requestRetry','requestDismiss','requestSignIn'].includes(child?.id))},focus(){if(!disabled)document.activeElement=this},getBoundingClientRect(){return {top:100,bottom:200}},scrollIntoView(){}};
 Object.defineProperty(el,'disabled',{get(){return disabled},set(value){disabled=value;if(value&&document.activeElement===el)document.activeElement=body}});
 Object.defineProperty(el,'hidden',{get(){return hidden},set(value){hidden=value;if(value&&el.contains(document.activeElement))document.activeElement=body}});nodes.set(id,el)
 }return nodes.get(id)}
const sandbox={URL,console,document,window:{innerHeight:844,location:{origin:'http://fixture.invalid',pathname:'/',search:'',reload(){reloads++}}},setInterval(){},setTimeout(){},fetch:()=>{requests++;return new Promise(resolve=>{assert.strictEqual(pending,null);pending=resolve})}};
vm.createContext(sandbox);vm.runInContext(input.base,sandbox);vm.runInContext(input.dashboard,sandbox);
async function finish(status,data){const resolve=pending;assert(resolve);pending=null;resolve({ok:status<400,status,redirected:false,json:async()=>data});await Promise.resolve();}
(async()=>{
 sandbox.showRequestNotice('Status is stale',{owner:'queue-poll'});sandbox.dismissRequestNotice();
 const button=node('refreshAllBtn'),other=node('user-input');button.focus();let work=sandbox.refreshAll();
 assert.strictEqual(document.activeElement,body,'Disabled initiating button did not model browser blur');assert.strictEqual(button.disabled,true);
 if(input.moveFocus)other.focus();
 await finish(200,input.partial);await work;assert.strictEqual(button.disabled,false);assert.strictEqual(button.textContent,'Refresh All');assert.strictEqual(node('requestNotice').hidden,false);
 assert.strictEqual(document.activeElement,input.moveFocus?other:node('requestNotice'),input.moveFocus?'Partial response stole a new user focus target':'Initial partial failure left the keyboard user stranded on BODY');
 sandbox.dismissRequestNotice();assert.strictEqual(document.activeElement,input.moveFocus?other:button,'Dismiss did not return to usable recovery control');
 if(input.moveFocus)return;
 button.focus();work=sandbox.refreshAll();await sandbox.refreshAll();assert.strictEqual(requests,2,'Busy header action sent a duplicate request');await finish(200,input.partial);await work;assert.strictEqual(document.activeElement,node('requestNotice'));assert.strictEqual(node('requestNotice').hidden,false,'Fresh explicit partial failure stayed dismissed');
 node('requestRetry').focus();work=node('requestRetry').onclick();assert.strictEqual(document.activeElement,node('requestRetry'));assert.strictEqual(node('requestNotice').hidden,false);await finish(200,input.partial);await work;assert.strictEqual(document.activeElement,node('requestRetry'),'D18 regressed focused Retry');
 work=node('requestRetry').onclick();await finish(200,input.success);await work;assert.strictEqual(reloads,1);assert.strictEqual(node('requestNotice').hidden,true);
 sandbox.clearRequestNotice();sandbox.showRequestNotice('Status is stale',{owner:'queue-poll'});assert.strictEqual(node('requestNotice').hidden,true,'Explicit partial recovery unmuted a background fault');
})().catch(error=>{console.error(error);process.exitCode=1});
