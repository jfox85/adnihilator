const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const nodes=new Map(),alerts=[];
function node(id){if(!nodes.has(id))nodes.set(id,{id,hidden:true,disabled:false,style:{},dataset:{},textContent:'',innerHTML:'',classList:{add(){},remove(){},contains(){return false},toggle(){}},querySelector(selector){return node(id+selector)},querySelectorAll(){return []},appendChild(){},remove(){},contains(){return false}});return nodes.get(id)}
const row=node('row');row.dataset={id:input.episode_id,status:'failed',title:'synthetic'};
let calls=0,resolveFetch,filterCalls=0;
const sandbox={console:{error(){}},URL,Error,window:{location:{origin:'http://fixture.invalid',pathname:'/podcasts/synthetic/episodes',search:'',reload(){}}},
 document:{hidden:false,addEventListener(){},getElementById:node,querySelector(selector){return selector.includes('data-id')?row:node(selector)},querySelectorAll(selector){return selector==='.episode-row'?[row]:[]},createElement(){return node('created')}},
 navigator:{},setInterval(){return 1},clearInterval(){},setTimeout(){},alert(message){alerts.push(message)},FormData:class{append(){}},
 fetch(){calls++;return new Promise(resolve=>{resolveFetch=resolve})}};
vm.createContext(sandbox);vm.runInContext(input.base,sandbox);vm.runInContext(input.episodes,sandbox);
sandbox.filterEpisodes=()=>{filterCalls++};
(async()=>{
 const first=sandbox.pollStatuses(),overlap=sandbox.pollStatuses();
 assert.strictEqual(calls,1,'Overlapping status requests');
 resolveFetch({ok:true,status:200,redirected:false,json:async()=>input.data});await Promise.all([first,overlap]);
 const priorFilters=filterCalls;
 sandbox.fetch=async()=>({ok:true,status:200,redirected:false,json:async()=>input.data});await sandbox.pollStatuses();
 assert.strictEqual(filterCalls,priorFilters,'Unchanged poll re-filters DOM');
 sandbox.fetch=async()=>({ok:false,status:500,redirected:false});await sandbox.pollStatuses();
 assert.strictEqual(node('requestNotice').hidden,false,'HTTP500 silently stale');
 assert(node('requestNoticeText').textContent.includes('out of date'));
 sandbox.fetch=async()=>{throw Error('SYNTHPRIVATEFETCH')};await sandbox.pollStatuses();
 assert(!node('requestNoticeText').textContent.includes('SYNTHPRIVATEFETCH'));
 sandbox.fetch=async()=>({ok:true,status:200,redirected:true,url:'http://fixture.invalid/login',json:async()=>{throw Error('HTML login')}});await sandbox.pollStatuses();
 assert.strictEqual(node('requestSignIn').hidden,false,'Redirected login has no recovery');
 const file={files:[{name:'synthetic.opml'}],value:'file',disabled:false};
 sandbox.fetch=async()=>({ok:false,status:500,json:async()=>{throw Error('SYNTHPRIVATEPARSE')}});
 await sandbox.importOPML(file);
 assert.strictEqual(file.disabled,false);assert.strictEqual(file.value,'');
 assert.strictEqual(node('requestNotice').hidden,false);
 assert(!node('requestNoticeText').textContent.includes('SYNTHPRIVATEPARSE'));
 assert(!alerts.join(' ').includes('SYNTHPRIVATEPARSE'));
 sandbox.fetch=async()=>{throw Error('SYNTHPRIVATENETWORK')};
 await sandbox.refreshPodcast();
 assert.strictEqual(node('refreshBtn').disabled,false);assert.strictEqual(node('refreshBtn').textContent,'Refresh');
 vm.runInContext(input.index,sandbox);
 const refresh=node('index-refresh');await sandbox.refreshPodcast('podcast',refresh);
 assert.strictEqual(refresh.disabled,false);assert.strictEqual(refresh.textContent,'Refresh');
 await sandbox.refreshAll();assert.strictEqual(node('refreshAllBtn').disabled,false);
 assert.strictEqual(node('refreshAllBtn').textContent,'Refresh All');
 const queueRow=node('queue-row');queueRow.dataset={episodeId:input.pending_id,podcastId:'podcast'};
 let rowQueries=0;
 queueRow.querySelector=selector=>{rowQueries++;return node('queue-row'+selector)};
 const queueSandbox={...sandbox,document:{...sandbox.document,querySelectorAll(selector){return selector==='.queue-item'?[queueRow]:[]}},fetch(){calls++;return new Promise(resolve=>{resolveFetch=resolve})}};
 vm.createContext(queueSandbox);vm.runInContext(input.base,queueSandbox);vm.runInContext(input.queue,queueSandbox);
 const priorCalls=calls,queueFirst=queueSandbox.updateStatuses(),queueOverlap=queueSandbox.updateStatuses();
 assert.strictEqual(calls,priorCalls+1,'Overlapping queue requests');
 resolveFetch({ok:true,status:200,redirected:false,json:async()=>input.data});await Promise.all([queueFirst,queueOverlap]);
 const previousQueries=rowQueries;
 queueSandbox.fetch=async()=>({ok:true,status:200,redirected:false,json:async()=>input.data});await queueSandbox.updateStatuses();
 assert.strictEqual(rowQueries,previousQueries,'Unchanged queue poll rewrites DOM');
 queueSandbox.fetch=async()=>({ok:false,status:500,redirected:false});await queueSandbox.updateStatuses();
 assert.strictEqual(node('requestNotice').hidden,false);
 assert(node('requestNoticeText').textContent.includes('out of date'));
})().catch(error=>{console.error(error);process.exitCode=1});
