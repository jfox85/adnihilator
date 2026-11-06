const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8')),nodes=new Map();
function node(id){if(!nodes.has(id))nodes.set(id,{hidden:true,textContent:'',disabled:false});return nodes.get(id)}
const sandbox={window:{location:{origin:'http://fixture.invalid',pathname:'/queue',search:''}},document:{addEventListener(){},getElementById:node}};
vm.createContext(sandbox);vm.runInContext(input.base,sandbox);
sandbox.showRequestNotice('Status is stale',{owner:'queue-poll'});
assert.strictEqual(node('requestNotice').hidden,false);
assert.strictEqual(typeof sandbox.dismissRequestNotice,'function');sandbox.dismissRequestNotice();
assert.strictEqual(node('requestNotice').hidden,true);
sandbox.showRequestNotice('Status is stale',{owner:'queue-poll'});assert.strictEqual(node('requestNotice').hidden,true);
sandbox.showRequestNotice('Import failed',{owner:'opml'});assert.strictEqual(node('requestNotice').hidden,false);
sandbox.showRequestNotice('Status is stale',{owner:'queue-poll'});assert.strictEqual(node('requestNoticeText').textContent,'Import failed');
sandbox.clearRequestNotice('queue-poll');assert.strictEqual(node('requestNotice').hidden,false);
sandbox.showRequestNotice('Status is stale',{owner:'queue-poll'});assert.strictEqual(node('requestNotice').hidden,false);
sandbox.dismissRequestNotice();sandbox.showRequestNotice('Sign in required',{owner:'queue-poll',auth:true});
assert.strictEqual(node('requestNotice').hidden,false);assert.strictEqual(node('requestSignIn').hidden,false);
