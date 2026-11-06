"""Formatting and confirmed bulk action scope regressions."""

import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from web.models import Episode, Podcast


@pytest.fixture
def ui_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """Use a disposable database and synthetic credentials only."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "ui.db"))
    monkeypatch.setenv("FEED_SYNC_ENABLED", "false")
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "synthetic-password")
    from web.app import app
    from web import dependencies

    with TestClient(app) as client:
        client.auth = ("admin", "synthetic-password")
        with dependencies.SessionFactory() as db:
            podcast = Podcast(id="podcast", source_rss_url="http://example.invalid/rss", title="A podcast")
            db.add(podcast)
            for name, state, audio in [
                ("skip", "skipped", True), ("fail", "failed", True),
                ("pending", "pending", True), ("active", "processing", True),
                ("done", "complete", True), ("silent", "skipped", False),
            ]:
                db.add(Episode(id=name, podcast_id="podcast", guid=name, title=name,
                               status=state, duration=3602, processed_duration=3602,
                               ads_removed_seconds=62, llm_cost_usd=Decimal("0.001234"),
                               original_audio_url="https://example.invalid/audio.mp3" if audio else None))
            db.commit()
        yield client


class TestDisplayFormats:
    @pytest.mark.parametrize("value,expected", [(3602, "01:00:02"), (62.9, "00:01:02"), (0, "00:00:00"), (None, "—"), (float("inf"), "—")])
    def test_duration(self, value: Any, expected: str) -> None:
        from web.formatting import format_duration

        assert format_duration(value) == expected

    @pytest.mark.parametrize("value,expected", [(Decimal("0.001234"), "$0.0012"), (Decimal("0.03125"), "$0.0313"), (Decimal("1234.123456"), "$1,234.1235"), (0, "$0.0000"), (None, "—"), (float("nan"), "—")])
    def test_cost(self, value: Any, expected: str) -> None:
        from web.formatting import format_cost

        assert format_cost(value) == expected

    def test_episode_and_queue_use_same_display(self, ui_client: TestClient) -> None:
        episodes = ui_client.get("/podcasts/podcast/episodes").text
        queue = ui_client.get("/queue").text
        for page in (episodes, queue):
            assert "01:00:02" in page or "00:01:02" in page
            assert "$0.0012" in page
        assert "60min" not in episodes


class TestConfirmedBulkScope:
    def test_preview_has_eligible_ids_and_no_processing(self, ui_client: TestClient) -> None:
        response = ui_client.get("/podcasts/podcast/episodes/bulk-summary")
        assert response.status_code == 200
        summary = response.json()
        assert set(summary["queue"]["episode_ids"]) == {"skip", "fail"}
        assert summary["queue"]["count"] == 2
        assert set(summary["skip"]["episode_ids"]) == {"pending", "fail"}

    @pytest.mark.parametrize("action,selected,untouched", [("queue-all", "skip", "fail"), ("skip-all", "pending", "fail")])
    def test_only_confirmed_selection_changes(self, ui_client: TestClient, action: str, selected: str, untouched: str) -> None:
        response = ui_client.post(f"/podcasts/podcast/episodes/{action}", json={"episode_ids": [selected]})
        assert response.status_code == 200
        assert response.json()["count"] == 1
        statuses = ui_client.get("/podcasts/podcast/episodes/status").json()["episodes"]
        assert statuses[untouched]["status"] == "failed"
        assert statuses["active"]["status"] == "processing"

    @pytest.mark.parametrize("action", ["queue-all", "skip-all"])
    def test_canceled_or_empty_selection_cannot_mutate(self, ui_client: TestClient, action: str) -> None:
        response = ui_client.post(f"/podcasts/podcast/episodes/{action}", json={"episode_ids": []})
        assert response.status_code == 200
        assert response.json()["count"] == 0

    def test_newly_arrived_episode_is_not_in_confirmed_action(self, ui_client: TestClient) -> None:
        from web import dependencies

        preview = ui_client.get("/podcasts/podcast/episodes/bulk-summary").json()["queue"]
        with dependencies.SessionFactory() as db:
            db.add(Episode(id="arrival", podcast_id="podcast", guid="arrival", status="skipped",
                           original_audio_url="https://example.invalid/new.mp3"))
            db.commit()
        result = ui_client.post("/podcasts/podcast/episodes/queue-all", json={"episode_ids": preview["episode_ids"]}).json()
        assert result["count"] == 2
        statuses = ui_client.get("/podcasts/podcast/episodes/status").json()["episodes"]
        assert statuses["arrival"]["status"] == "skipped"

    def test_real_rendered_javascript_confirms_scope_and_cost(self, ui_client: TestClient) -> None:
        node = shutil.which("node")
        if node is None:
            pytest.skip("Node is required to execute the rendered browser script")
        scripts = re.findall(r"<script>(.*?)</script>", ui_client.get("/podcasts/podcast/episodes").text, re.S)
        preview = ui_client.get("/podcasts/podcast/episodes/bulk-summary").json()
        script = r'''
const vm = require('node:vm');
const assert = require('node:assert/strict');
const payload = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const calls = [], confirms = [], alerts = [];
let approve = false;
const button = {disabled:false};
const c = {console, Intl, Number, Date, String, setInterval:()=>1, clearInterval:()=>{},
 setTimeout:()=>1, document:{addEventListener:()=>{}, querySelector:()=>button},
 window:{location:{pathname:'/podcasts/podcast/episodes',search:'',assign:()=>{}}},
 alert:m=>alerts.push(m), confirm:m=>{confirms.push(m);return approve},
 fetch:async(url,options)=>{calls.push({url,options});return {ok:true,status:200,
 json:async()=>url.endsWith('bulk-summary')?payload.preview:{count:1,episode_ids:['skip']}}}};
vm.createContext(c); payload.scripts.forEach(s=>vm.runInContext(s,c));
c.updateEpisodeUI=()=>{}; c.filterEpisodes=()=>{};
c.pollStatuses=async()=>{calls.push({url:'status-refresh'})};
(async()=>{
 await c.queueAll(); assert.equal(calls.length,1); // Cancel spends no processing call.
 assert.match(confirms[0], /Queue 2/); assert.match(confirms[0], /paid API credits/);
 assert.match(confirms[0], /hidden by filters/); assert.equal(button.disabled,false);
 approve=true; await c.queueAll(); assert.equal(calls.length,4);
 assert.deepEqual(JSON.parse(calls[2].options.body).episode_ids.sort(),['fail','skip']);
 assert.match(alerts.at(-1), /1 of 2 episode\(s\) queued; 1 changed state since preview/);
 assert.equal(c.formatDuration(3602),'01:00:02');
 assert.equal(c.formatCost(0.03125),'$0.0313');
 assert.equal(c.formatCost(1234.123456),'$1,234.1235');
 console.log('rendered JS: cancel, confirmed IDs, cost warning, result count and format parity pass');
})().catch(e=>{console.error(e);process.exitCode=1});
'''
        result = subprocess.run([node, "-e", script], input=json.dumps({"scripts": scripts, "preview": preview}),
                                text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr


class TestBulkResultReconciliation:
    @pytest.mark.parametrize("action", ["queue", "skip"])
    @pytest.mark.parametrize("accepted", [0, 48, 50])
    def test_rendered_bulk_result_refreshes_changed_rows_and_explains_shortfall(
        self, ui_client: TestClient, action: str, accepted: int
    ) -> None:
        """Paused tabs reconcile every result, including zero accepted IDs."""
        node = shutil.which("node")
        if node is None:
            pytest.skip("Node is required to execute the rendered browser script")
        scripts = re.findall(r"<script>(.*?)</script>", ui_client.get("/podcasts/podcast/episodes").text, re.S)
        harness = r'''
const vm=require('node:vm');
const assert=require('node:assert/strict');
const payload=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const ids=Array.from({length:50},(_,i)=>'episode-'+i);
const button={disabled:false}; const events=[], states=new Map(ids.map(id=>[id,'failed']));
const verb=payload.action==='queue'?'queued':'skipped';
const target=payload.action==='queue'?'pending':'skipped';
const context={console,Intl,Number,Date,String,setInterval:()=>1,clearInterval:()=>{},setTimeout:()=>1,
 document:{addEventListener:()=>{},querySelector:()=>button},
 window:{location:{pathname:'/podcasts/podcast/episodes',search:'',assign:()=>{}}},
 confirm:()=>true,alert:message=>events.push(['alert',message]),
 fetch:async(url,options)=>({ok:true,status:200,json:async()=>url.endsWith('bulk-summary')
  ?{[payload.action]:{count:50,episode_ids:ids}}
  :{count:payload.accepted,episode_ids:ids.slice(0,payload.accepted)}})};
vm.createContext(context); payload.scripts.forEach(script=>vm.runInContext(script,context));
context.updateEpisodeUI=(id,row)=>states.set(id,row.status);
context.filterEpisodes=()=>{};
context.pollStatuses=async()=>{
 assert.equal(button.disabled,true,'bulk control must stay disabled until refresh finishes');
 events.push(['poll']);
 ids.forEach((id,index)=>states.set(id,payload.accepted===48&&index>=48?(index===48?'processing':'complete'):target));
};
(async()=>{
 await context.bulkAction(payload.action);
 assert.equal(events.filter(event=>event[0]==='poll').length,1,'every successful bulk result must force one status refresh');
 assert.equal(events[0][0],'poll','refresh must complete before result notice');
 const notice=events.find(event=>event[0]==='alert')[1];
 if(payload.accepted<50){
  assert.match(notice,new RegExp(payload.accepted+' of 50'));
  assert.match(notice,new RegExp((50-payload.accepted)+' changed state since preview'));
 }else assert.match(notice,new RegExp('50 episode\\(s\\) '+verb));
 assert.equal(states.get('episode-48'),payload.accepted===48?'processing':target);
 assert.equal(states.get('episode-49'),payload.accepted===48?'complete':target);
 assert.equal(button.disabled,false);
 console.log(JSON.stringify({action:payload.action,accepted:payload.accepted,notice,events}));
})().catch(error=>{console.error(error);process.exitCode=1});
'''
        result = subprocess.run([node, "-e", harness], input=json.dumps({"scripts": scripts, "action": action, "accepted": accepted}), text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr
