"""Queue polling must retain durable hold labels instead of undoing SSR."""
import json
import re
import shutil
import subprocess

import pytest

from tests.web.test_provider_outage_contract import outage_client,expire_cooldown,hold,HEADERS


@pytest.mark.parametrize('rep',[1,2])
@pytest.mark.parametrize('recovering',[False,True])
def test_status_poll_exposes_hold_state_without_changing_episode_status(outage_client,rep,recovering):
    """Consumers need hold metadata while original status remains compatible."""
    client,path=outage_client
    client.post('/api/queue/claim',headers=HEADERS);hold(client)
    if recovering:
        expire_cooldown(path);client.post('/api/queue/claim',headers=HEADERS)
    data=client.get('/podcasts/pod/episodes/status').json()['episodes']
    assert data['job0']['provider_held'] is True
    assert data['job0']['status']==('processing' if recovering else 'pending')
    assert data['job1']['provider_held'] is False
    assert 'token' not in json.dumps(data)


@pytest.mark.skipif(not shutil.which('node'),reason='Node is unavailable')
@pytest.mark.parametrize('recovering',[False,True])
def test_rendered_queue_poll_keeps_hold_label_and_then_releases(outage_client,recovering):
    """Execute actual rendered polling repeatedly, including a later recovery."""
    client,path=outage_client
    client.post('/api/queue/claim',headers=HEADERS);hold(client)
    if recovering:
        expire_cooldown(path);client.post('/api/queue/claim',headers=HEADERS)
    html=client.get('/queue').text
    script=next(s for s in re.findall(r'<script>(.*?)</script>',html,re.S) if 'async function updateStatuses()' in s)
    data=client.get('/podcasts/pod/episodes/status').json()
    # Explicit contract marker also isolates the JS consumer before the server fix.
    data['episodes']['job0']['provider_held']=True
    program=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const input=JSON.parse(fs.readFileSync(0,'utf8'));const badge={className:'',textContent:''};
const actions={innerHTML:''};const item={dataset:{podcastId:'pod',episodeId:'job0'},querySelector(s){return s==='.status-badge'?badge:s==='.queue-item-actions'?actions:null},remove(){throw Error('A held episode was removed')}};
const sandbox={console,setInterval(){},document:{querySelectorAll(){return [item]},querySelector(){return item}},fetch:async()=>({ok:true,json:async()=>input.data})};
vm.createContext(sandbox);vm.runInContext(input.script,sandbox);
(async()=>{for(let rep=0;rep<3;rep++){await sandbox.updateStatuses();assert.equal(badge.textContent,input.recovering?'recovering':'held');}
input.data.episodes.job0.provider_held=false;await sandbox.updateStatuses();assert.equal(badge.textContent,input.recovering?'processing':'pending');})().catch(e=>{console.error(e);process.exitCode=1});
"""
    result=subprocess.run(['node','-e',program],input=json.dumps({'script':script,'data':data,'recovering':recovering}),text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
