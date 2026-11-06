"""Server-rendered controls expose keyboard and accessible-name semantics."""

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
import pytest

from web.database import get_session_factory
from web.models import Episode, Podcast


@pytest.fixture
def pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, BeautifulSoup]]:
    """Render actual routes with an isolated database and no feed sync."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "ui.db"))
    monkeypatch.setenv("ADMIN_USERNAME", "ui-test")
    monkeypatch.setenv("ADMIN_PASSWORD", "synthetic-password")
    monkeypatch.setenv("WORKER_API_KEY", "synthetic-worker")
    monkeypatch.setenv("FEED_SYNC_ENABLED", "false")
    from web.app import app

    with TestClient(app) as client:
        with get_session_factory()() as session:
            podcast = Podcast(id="podcast", title="Synthetic Podcast", source_rss_url="https://example.invalid/feed")
            session.add(podcast)
            session.add_all([
                Episode(id=status, podcast_id="podcast", guid=status, title=f"Synthetic {status}", status=status)
                for status in ["pending", "processing", "complete", "failed", "skipped"]
            ])
            session.commit()
        yield {
            name: BeautifulSoup(client.get(url, auth=("ui-test", "synthetic-password")).text, "html.parser")
            for name, url in {"dashboard": "/", "episodes": "/podcasts/podcast/episodes", "queue": "/queue"}.items()
        }


class TestAccessibleControls:
    """Exercise semantics and disclosure behavior exposed to keyboard users."""

    def test_episode_details_have_native_buttons_and_named_panels(self, pages: dict) -> None:
        """Keyboard users can reach each closed row without nested controls."""
        rows = pages["episodes"].select(".episode-row")
        assert len(rows) == 5
        for row in rows:
            toggle = row.select_one("button.episode-toggle")
            assert toggle is not None
            assert toggle["aria-expanded"] == "false"
            panel = row.find(id=toggle["aria-controls"])
            assert panel is not None and panel.has_attr("hidden")
            assert not toggle.select("button, a, input")

    def test_tools_opml_can_be_reached_with_keyboard(self, pages: dict) -> None:
        """The disclosure exposes state and import has a native activation target."""
        page = pages["dashboard"]
        trigger = page.select_one(".tools-btn")
        assert trigger["aria-expanded"] == "false"
        assert trigger["aria-controls"] == "toolsDropdown"
        assert trigger.get("aria-label") == "Tools"
        assert page.select_one("#toolsDropdown button").get_text(strip=True) == "Import OPML…"

    def test_form_controls_and_artwork_have_names(self, pages: dict) -> None:
        """Names survive missing artwork and placeholders do not replace labels."""
        for page in pages.values():
            for control in page.select('input:not([type="file"]), select'):
                assert control.get("aria-label") or page.find("label", attrs={"for": control.get("id")})
            for artwork in page.select('a[class$="artwork-link"]'):
                assert artwork.get("aria-label") or artwork.get_text(strip=True) or artwork.find("img", alt=True)

    def test_tools_and_episode_state_changes_execute_real_script(self, pages: dict) -> None:
        """Actual emitted functions keep visibility, expanded state and focus in sync."""
        node = shutil.which("node")
        if node is None:
            pytest.skip("Node is needed for emitted JavaScript behavior checks")
        scripts = "\n".join(script.string or "" for script in pages["episodes"].find_all("script"))
        harness = r'''
const assert = require('node:assert/strict');
const listeners = {};
function element() {
  const classes = new Set();
  return {hidden:true, attrs:{}, innerHTML:'', focused:false,
    classList:{contains:x=>classes.has(x), add:x=>classes.add(x), remove:x=>classes.delete(x),
      toggle:(x,v)=>{if(v===undefined)v=!classes.has(x);v?classes.add(x):classes.delete(x);return v;}},
    setAttribute(k,v){this.attrs[k]=String(v)}, focus(){this.focused=true},
    contains(){return false}, querySelector(){return null}};
}
const elements = Object.fromEntries(['toolsDropdown','toolsButton','details-test','icon-test','toggle-test'].map(id=>[id,element()]));
global.document={getElementById:id=>elements[id],querySelector:()=>elements.toolsButton,
  addEventListener:(name,fn)=>{(listeners[name]??=[]).push(fn)}};
global.window={location:{origin:'http://synthetic.invalid'}};
global.setInterval=()=>1;
global.setTimeout=()=>1;
'''
        assertions = r'''
toggleDetails('test');
assert.equal(elements['toggle-test'].attrs['aria-expanded'],'true');
assert.equal(elements['details-test'].hidden,false);
toggleDetails('test');
assert.equal(elements['toggle-test'].attrs['aria-expanded'],'false');
assert.equal(elements['details-test'].hidden,true);
toggleToolsMenu();
assert.equal(elements.toolsButton.attrs['aria-expanded'],'true');
for(const fn of listeners.keydown??[]) fn({key:'Escape',preventDefault(){}});
assert.equal(elements.toolsButton.attrs['aria-expanded'],'false');
assert.equal(elements.toolsButton.focused,true);
'''
        subprocess.run([node, "-e", harness + scripts + assertions], check=True, capture_output=True, text=True)
