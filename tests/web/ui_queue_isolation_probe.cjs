const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8')),nodes=new Map(),requests=[];
function node(id){if(!nodes.has(id))nodes.set(id,{id,hidden:true,textContent:'',style:{},dataset:{},classList:{add(){},remove(){},toggle(){},contains(){return false}},setAttribute(){},focus(){},remove(){this.removed=true},querySelector(selector){return node(id+selector)},querySelectorAll(){return []},appendChild(){}});return nodes.get(id)}
const bad=node('bad-row'),healthy=node('healthy-row');bad.dataset={episodeId:'bad-episode',podcastId:'bad'};healthy.dataset={episodeId:'healthy-episode',podcastId:'healthy'};
let badStatus=input.bad_status;
const sandbox={URL,console,window:{location:{origin:'http://fixture.invalid',pathname:'/queue',search:''}},document:{hidden:false,addEventListener(){},getElementById:node,querySelector:node,querySelectorAll(selector){return selector==='.queue-item'?[bad,healthy]:[]},createElement(){return node('created')}},setInterval(){return 1},clearInterval(){},setTimeout(){},fetch:async url=>{requests.push(url);return url.includes('/bad/')?{ok:false,status:badStatus,redirected:false}:{ok:true,status:200,redirected:false,json:async()=>input.healthy}}};
vm.createContext(sandbox);vm.runInContext(input.base,sandbox);vm.runInContext(input.queue,sandbox);
(async()=>{
 await sandbox.updateStatuses();
 assert(requests.some(url=>url.includes('/healthy/')),'First failed podcast starved healthy status');
 assert.strictEqual(node('healthy-row.status-badge').textContent,'processing');
 assert.strictEqual(node('requestNotice').hidden,false,'Partial failure cleared stale notice');
 assert(node('requestNoticeText').textContent.includes('out of date'));
 requests.length=0;badStatus=500;input.healthy.episodes['healthy-episode'].status='complete';
 await node('requestRetry').onclick();
 assert(requests.some(url=>url.includes('/healthy/')),'Retry starved healthy status');
 assert.strictEqual(healthy.removed,true,'Healthy completion never removed');
 assert.strictEqual(node('requestNotice').hidden,false,'Healthy update hid another group failure');
 badStatus=401;await sandbox.updateStatuses();
 assert.strictEqual(node('requestSignIn').hidden,false,'Auth failure lost sign-in recovery');
})().catch(error=>{console.error(error);process.exitCode=1});
