"""Catalogue filtering, ordering and polling stay bounded across page boundaries."""

import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from web.models import Episode, Podcast


@pytest.fixture
def catalogue_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """Seed a multi-page catalogue without external feed or worker activity."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "catalogue.db"))
    monkeypatch.setenv("FEED_SYNC_ENABLED", "false")
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "synthetic-password")
    from web import dependencies
    from web.app import app

    with TestClient(app) as client:
        client.auth = ("admin", "synthetic-password")
        with dependencies.SessionFactory() as db:
            db.add_all([Podcast(id="catalogue", source_rss_url="https://example.invalid/rss"),
                        Podcast(id="other", source_rss_url="https://example.invalid/other")])
            for number in range(251):
                db.add(Episode(id=f"ep-{number:03}", podcast_id="catalogue", guid=str(number),
                               title=f"Episode {number:03}" + (" 100%_literal" if number == 3 else ""),
                               status="failed" if number % 2 else "skipped",
                               published_at=datetime(2026, 1, 1) + timedelta(days=number),
                               original_audio_url="https://example.invalid/audio.mp3"))
            db.add(Episode(id="foreign", podcast_id="other", guid="foreign", status="pending"))
            db.commit()
        yield client


def row_ids(response: Response) -> list[str]:
    """Read actual rendered identities instead of counting incidental markup."""
    return re.findall(r'data-id="(ep-\d+)"', response.text)


class TestCataloguePagination:
    """Every catalogue entry remains reachable; filters precede LIMIT/OFFSET."""

    def test_pages_cover_catalogue_once_in_stable_order(self, catalogue_client: TestClient) -> None:
        pages = [catalogue_client.get(f"/podcasts/catalogue/episodes?page={p}") for p in (1, 2, 3)]
        assert all(p.status_code == 200 for p in pages)
        assert [len(row_ids(p)) for p in pages] == [100, 100, 51]
        assert [ep for page in pages for ep in row_ids(page)] == [f"ep-{n:03}" for n in range(250, -1, -1)]
        assert "Showing 101–200 of 251" in pages[1].text

    def test_filter_search_and_sort_apply_beyond_first_page(self, catalogue_client: TestClient) -> None:
        response = catalogue_client.get("/podcasts/catalogue/episodes", params={"q": "Episode 0", "status": "failed", "sort": "title-asc"})
        assert row_ids(response) == [f"ep-{n:03}" for n in range(1, 100, 2)]
        assert 'name="q"' in response.text
        response = catalogue_client.get("/podcasts/catalogue/episodes", params={"q": "%_literal"})
        assert row_ids(response) == ["ep-003"]

    def test_links_preserve_filters_and_escape_input(self, catalogue_client: TestClient) -> None:
        response = catalogue_client.get("/podcasts/catalogue/episodes", params={"status": "failed", "sort": "title-asc", "page": 1})
        assert "status=failed&amp;sort=title-asc&amp;page=2" in response.text
        assert len(row_ids(response)) == 100
        response = catalogue_client.get("/podcasts/catalogue/episodes", params={"q": '<script>alert("x")</script>'})
        assert row_ids(response) == []
        assert '&lt;script&gt;' in response.text

    def test_out_of_range_page_clamps_and_invalid_inputs_reject(self, catalogue_client: TestClient) -> None:
        response = catalogue_client.get("/podcasts/catalogue/episodes?page=999")
        assert len(row_ids(response)) == 51
        assert "Page 3 of 3" in response.text
        for query in ("page=0", "page=-1", "sort=unsafe", "status=unknown"):
            assert catalogue_client.get(f"/podcasts/catalogue/episodes?{query}").status_code == 422

    def test_polling_limits_to_selected_page_and_podcast(self, catalogue_client: TestClient) -> None:
        identities = row_ids(catalogue_client.get("/podcasts/catalogue/episodes?page=2"))
        response = catalogue_client.get("/podcasts/catalogue/episodes/status", params=[("episode_id", i) for i in identities])
        assert set(response.json()["episodes"]) == set(identities)
        foreign = catalogue_client.get("/podcasts/catalogue/episodes/status?episode_id=foreign")
        assert foreign.json()["episodes"] == {}
        too_many = catalogue_client.get("/podcasts/catalogue/episodes/status", params=[("episode_id", str(i)) for i in range(101)])
        assert too_many.status_code == 422

    def test_bulk_preview_still_covers_whole_catalogue(self, catalogue_client: TestClient) -> None:
        result = catalogue_client.get("/podcasts/catalogue/episodes/bulk-summary").json()
        assert result["queue"]["count"] == 251
        assert len(result["queue"]["episode_ids"]) == 251

    def test_unicode_and_empty_titles_search_by_displayed_name(self, catalogue_client: TestClient) -> None:
        from web import dependencies

        with dependencies.SessionFactory() as db:
            db.get(Episode, "ep-003").title = "ÉCOLE ÁRBOL Straße"
            episode = db.get(Episode, "ep-004")
            episode.title = ""
            episode.guid = "visible-empty-fallback"
            db.commit()
        for query in ("ÉCOLE", "école", "ÁRBOL", "árbol", "STRASSE"):
            response = catalogue_client.get("/podcasts/catalogue/episodes", params={"q": query})
            assert row_ids(response) == ["ep-003"]
        response = catalogue_client.get("/podcasts/catalogue/episodes?q=visible-empty-fallback")
        assert row_ids(response) == ["ep-004"]

    def test_navigation_is_relative_to_current_origin(self, catalogue_client: TestClient) -> None:
        response = catalogue_client.get("/podcasts/catalogue/episodes?page=2&status=failed")
        assert 'href="/podcasts/catalogue/episodes?' in response.text
        assert 'href="http://testserver/podcasts/' not in response.text

    def test_catalogue_poll_reconciles_incoming_bulk_matches(self, catalogue_client: TestClient) -> None:
        url = "/podcasts/catalogue/episodes/status?catalogue_page=1&status=pending"
        assert catalogue_client.get(url).json()["catalogue"]["total"] == 0
        preview = catalogue_client.get("/podcasts/catalogue/episodes/bulk-summary").json()["queue"]
        queued = catalogue_client.post("/podcasts/catalogue/episodes/queue-all", json={"episode_ids": preview["episode_ids"]})
        assert queued.json()["count"] == 251
        data = catalogue_client.get(url).json()
        assert data["catalogue"]["total"] == 251
        assert len(data["episodes"]) == len(data["catalogue"]["ids"]) == 100
        assert all(ep["status"] == "pending" for ep in data["episodes"].values())

    def test_status_sorted_poll_reconciles_changed_membership(self, catalogue_client: TestClient) -> None:
        from web import dependencies

        url = "/podcasts/catalogue/episodes/status?catalogue_page=1&sort=status"
        before = catalogue_client.get(url).json()["catalogue"]["ids"]
        with dependencies.SessionFactory() as db:
            db.get(Episode, before[0]).status = "complete"
            db.commit()
        after = catalogue_client.get(url).json()["catalogue"]["ids"]
        assert len(after) == 100
        assert before[0] not in after
        assert before != after

    @pytest.mark.parametrize("action,visible_status", [("queue", "pending"), ("skip", "skipped")])
    def test_background_poll_cannot_navigate_before_delayed_bulk_result(
        self, catalogue_client: TestClient, action: str, visible_status: str,
    ) -> None:
        """Replay a committed action whose response arrives after the next poll."""
        if not shutil.which("node"):
            pytest.skip("Node is required to execute actual rendered scripts")
        from web import dependencies

        if action == "skip":
            with dependencies.SessionFactory() as db:
                db.query(Episode).filter_by(podcast_id="catalogue").update({Episode.status: "pending"})
                db.commit()
        page = catalogue_client.get(f"/podcasts/catalogue/episodes?status={visible_status}")
        assert row_ids(page) == []
        preview = catalogue_client.get("/podcasts/catalogue/episodes/bulk-summary").json()
        result = catalogue_client.post(
            f"/podcasts/catalogue/episodes/{action}-all",
            json={"episode_ids": preview[action]["episode_ids"]},
        ).json()
        assert result["count"] == 251
        statuses = catalogue_client.get(
            f"/podcasts/catalogue/episodes/status?catalogue_page=1&status={visible_status}"
        ).json()
        program = r'''
const vm=require('node:vm'),assert=require('node:assert/strict');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const buttons=[{id:'catalogueQueueAll',disabled:false},{id:'catalogueSkipAll',disabled:false}];
const nodes={searchInput:{value:''},autoProcessStatus:{textContent:''}};
let deliverBulk;const events=[];
const response=body=>({ok:true,status:200,json:async()=>body});
const sandbox={console,URLSearchParams,Intl,Number,Date,String,setInterval:()=>1,clearInterval(){},setTimeout(){},
 document:{activeElement:buttons[input.action==='queue'?0:1],addEventListener(){},querySelectorAll(){return[]},
 getElementById:id=>nodes[id],querySelector:s=>s.startsWith('[data-id')?null:buttons[s.includes('nth-child(2)')?1:0]},
 window:{scrollY:0,location:{href:'http://fixture/podcasts/catalogue/episodes',pathname:'/podcasts/catalogue/episodes',search:'',reload(){events.push(['reload'])}},history:{state:{},replaceState(state){this.state=state}}},
 confirm:()=>true,alert:message=>events.push(['alert',message]),
 fetch:async url=>url.endsWith('bulk-summary')?response(input.preview):url.includes('/episodes/status?')?response(input.statuses):new Promise(resolve=>{deliverBulk=()=>resolve(response(input.result))})};
vm.createContext(sandbox);input.scripts.forEach(script=>vm.runInContext(script,sandbox));
(async()=>{
 const action=sandbox.bulkAction(input.action);
 for(let i=0;i<20&&!deliverBulk;i++)await new Promise(setImmediate);
 assert(deliverBulk,'bulk request did not start');
 await sandbox.pollStatuses(); // Same callback used by the ordinary interval.
 assert.deepEqual(events,[],'background polling navigated before the committed result arrived');
 deliverBulk();await action;
 assert.equal(events[0][0],'alert');assert.match(events[0][1],/251 episode\(s\) (queued|skipped)/);
 assert.equal(events[1][0],'reload');assert(!events.some(e=>String(e[1]).includes('Could not')));
 assert(buttons.every(button=>!button.disabled));
})().catch(error=>{console.error(error);process.exitCode=1});
'''
        observed = subprocess.run(
            ["node", "-e", program],
            input=json.dumps({"action": action, "scripts": re.findall(r"<script>(.*?)</script>", page.text, re.S),
                              "preview": preview, "result": result, "statuses": statuses}),
            text=True, capture_output=True, timeout=10,
        )
        assert observed.returncode == 0, observed.stdout + observed.stderr
