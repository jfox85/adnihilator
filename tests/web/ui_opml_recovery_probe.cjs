const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8')),nodes=new Map();
function node(id){if(!nodes.has(id))nodes.set(id,{hidden:true,textContent:'',disabled:false,classList:{remove(){},toggle(){},contains(){return false}},setAttribute(){},focus(){}});return nodes.get(id)}
const sandbox={URL,console,window:{location:{origin:'http://fixture.invalid',pathname:'/',search:'',reload(){throw Error('Zero imports must not reload')}}},document:{addEventListener(){},getElementById:node},FormData:class{append(){}},alert(){},fetch:async()=>({ok:false,status:500,redirected:false})};
vm.createContext(sandbox);vm.runInContext(input.base,sandbox);
const file={files:[{name:'fixture.opml'}],value:'file',disabled:false};
(async()=>{
 await sandbox.importOPML(file);assert.strictEqual(node('requestNotice').hidden,false);
 sandbox.fetch=async()=>({ok:true,status:200,redirected:false,json:async()=>({imported:0,skipped:1})});
 await sandbox.importOPML(file);assert.strictEqual(node('requestNotice').hidden,true,'Successful retry kept stale import failure');
 sandbox.showRequestNotice('Unrelated live status is stale',{owner:'queue-poll'});
 await sandbox.importOPML(file);assert.strictEqual(node('requestNotice').hidden,false,'Import cleared unrelated status failure');
 assert.strictEqual(node('requestNoticeText').textContent,'Unrelated live status is stale');
 assert.strictEqual(file.disabled,false);
})().catch(error=>{console.error(error);process.exitCode=1});
