"""Durable provider holds protect retry budgets and serialize recovery probes."""
import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

HEADERS={'X-Worker-API-Key':'synthetic-worker','X-Worker-Protocol':'provider-outage-v1'}


@pytest.fixture
def outage_client(monkeypatch,tmp_path):
    """Each contract replay owns a real SQLite file and authenticated app."""
    path=tmp_path/'outages.sqlite'
    for key,value in {'DATABASE_PATH':str(path),'ADMIN_USERNAME':'outage-admin','ADMIN_PASSWORD':'synthetic-admin','WORKER_API_KEY':'synthetic-worker','FEED_SYNC_ENABLED':'false'}.items():monkeypatch.setenv(key,value)
    from web.app import app
    from web import dependencies
    from web.models import Podcast,Episode
    with TestClient(app) as client:
        assert client.post('/login',data={'username':'outage-admin','password':'synthetic-admin'},follow_redirects=False).status_code==303
        with dependencies.SessionFactory() as db:
            db.add(Podcast(id='pod',title='Synthetic',source_rss_url='https://example.invalid/feed'));db.flush()
            for i in range(3):db.add(Episode(id=f'job{i}',podcast_id='pod',guid=f'job{i}',title=f'Episode{i}',original_audio_url=f'https://example.invalid/{i}.mp3',status='pending',retry_count=1,created_at=datetime.utcnow()+timedelta(seconds=i)))
            db.commit()
        yield client,path


def hold(client,episode='job0',proofs=None):
    response=client.post(f'/api/queue/{episode}/provider-hold',headers=HEADERS,json={'provider':'gemini','category':'billing','error':'credits depleted; key=synthetic-worker','recovery_tokens':proofs or []})
    assert response.status_code==200
    return response


def expire_cooldown(path):
    with sqlite3.connect(path) as db:db.execute("UPDATE provider_outages SET retry_after='2000-01-01 00:00:00'")


def complete(client,job,proofs=None):
    return client.post(f"/api/queue/{job['id']}/complete",headers=HEADERS,json={'processed_audio_key':'synthetic/clean.mp3','processed_duration':120,'ads_removed_seconds':0,'recovery_tokens':proofs if proofs is not None else job.get('provider_recovery',[]),'provider_successes':[p['provider'] for p in job.get('provider_recovery',[])]})


class TestDurableOutageContract:
    def test_hold_preserves_retry_budget_blocks_claims_and_survives_restart(self,outage_client):
        client,path=outage_client
        assert client.post('/api/queue/claim',headers=HEADERS).json()['id']=='job0'
        hold(client)
        assert client.post('/api/queue/claim',headers=HEADERS).json() is None
        for page in ['/','/queue']:
            response=client.get(page);assert response.status_code==200
            assert 'Provider processing paused' in response.text and 'Retry Gemini now' in response.text
            assert 'synthetic-worker' not in response.text
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT status,retry_count,claimed_at FROM episodes WHERE id='job0'").fetchone()==('pending',1,None)
            assert db.execute('SELECT active FROM provider_outages').fetchone()[0]==1
        from web.app import app
        with TestClient(app) as restarted:assert restarted.post('/api/queue/claim',headers=HEADERS).json() is None

    def test_due_cooldown_grants_one_probe_then_success_releases_holds(self,outage_client):
        client,path=outage_client;client.post('/api/queue/claim',headers=HEADERS);hold(client);expire_cooldown(path)
        probe=client.post('/api/queue/claim',headers=HEADERS).json()
        assert probe['id']=='job0' and len(probe['provider_recovery'])==1
        assert client.post('/api/queue/claim',headers=HEADERS).json() is None
        assert complete(client,probe).status_code==200
        assert client.post('/api/queue/claim',headers=HEADERS).json()['id']=='job1'
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT retry_count FROM episodes WHERE id='job0'").fetchone()[0]==1
            assert db.execute('SELECT active FROM provider_outages').fetchone()[0]==0
            assert db.execute('SELECT count(*) FROM episode_provider_holds').fetchone()[0]==0

    def test_repeated_probe_failure_has_bounded_cooldown_and_no_episode_cost(self,outage_client):
        client,path=outage_client;client.post('/api/queue/claim',headers=HEADERS);hold(client)
        for _ in range(5):
            expire_cooldown(path);probe=client.post('/api/queue/claim',headers=HEADERS).json();hold(client,episode=probe['id'],proofs=probe['provider_recovery'])
            with sqlite3.connect(path) as db:
                retry_at=datetime.fromisoformat(db.execute('SELECT retry_after FROM provider_outages').fetchone()[0]);delay=(retry_at-datetime.utcnow()).total_seconds()
                assert 290<=delay<=1800
                assert db.execute("SELECT retry_count FROM episodes WHERE id='job0'").fetchone()[0]==1
        assert delay>1700

    def test_manual_retry_accelerates_but_cannot_duplicate_current_probe(self,outage_client):
        client,path=outage_client;client.post('/api/queue/claim',headers=HEADERS);hold(client)
        assert client.post('/provider-outages/gemini/retry',headers={'Origin':'http://testserver'},follow_redirects=False).status_code==303
        probe=client.post('/api/queue/claim',headers=HEADERS).json();assert probe['provider_recovery']
        client.post('/provider-outages/gemini/retry',headers={'Origin':'http://testserver'},follow_redirects=False)
        assert client.post('/api/queue/claim',headers=HEADERS).json() is None
        assert complete(client,probe,proofs=[]).status_code==409
        assert complete(client,probe).status_code==200

    @pytest.mark.parametrize('rep',[1,2])
    def test_concurrent_sqlite_claimers_share_one_recovery_lease(self,outage_client,rep):
        client,path=outage_client;client.post('/api/queue/claim',headers=HEADERS);hold(client);expire_cooldown(path)
        from web import dependencies
        from web.routes.api import claim_episode
        barrier=Barrier(6)
        def claim(_):
            with dependencies.SessionFactory() as db:
                barrier.wait()
                return asyncio.run(claim_episode(db=db,_=None,x_worker_protocol='provider-outage-v1'))
        with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(claim,range(6)))
        claimed=[result for result in results if result is not None]
        assert len(claimed)==1 and claimed[0]['id']=='job0'
        with sqlite3.connect(path) as db:assert db.execute("SELECT retry_count FROM episodes WHERE id='job0'").fetchone()[0]==1

    def test_stale_probe_is_reheld_without_retry_loss_and_old_token_rejected(self,outage_client):
        client,path=outage_client;client.post('/api/queue/claim',headers=HEADERS);hold(client);expire_cooldown(path)
        stale=client.post('/api/queue/claim',headers=HEADERS).json()
        with sqlite3.connect(path) as db:db.execute("UPDATE episodes SET claimed_at='2000-01-01 00:00:00' WHERE id='job0'")
        assert client.post('/api/queue/claim',headers=HEADERS).json() is None
        with sqlite3.connect(path) as db:assert db.execute("SELECT status,retry_count FROM episodes WHERE id='job0'").fetchone()==('pending',1)
        expire_cooldown(path);current=client.post('/api/queue/claim',headers=HEADERS).json()
        assert current['provider_recovery']!=stale['provider_recovery']
        assert complete(client,stale).status_code==409
        assert complete(client,current).status_code==200

    def test_normal_episode_failure_budget_remains_unchanged(self,outage_client):
        client,path=outage_client;client.post('/api/queue/claim',headers=HEADERS)
        response=client.post('/api/queue/job0/fail',headers=HEADERS,json={'error':'per-episode corrupt audio'})
        assert response.status_code==200
        with sqlite3.connect(path) as db:assert db.execute("SELECT retry_count FROM episodes WHERE id='job0'").fetchone()[0]==2

    def test_rollout_pause_survives_restart_until_owner_resume(self,outage_client):
        client,path=outage_client
        from web import dependencies
        from web.services.provider_outages import pause_for_rollout
        with dependencies.SessionFactory() as db:pause_for_rollout(db)
        expire_cooldown(path)
        assert client.post('/api/queue/claim',headers=HEADERS).json() is None
        from web.app import app
        with TestClient(app) as restarted:assert restarted.post('/api/queue/claim',headers=HEADERS).json() is None
        response=client.get('/');assert 'Resume Gemini processing' in response.text
        assert client.post('/provider-outages/gemini/retry',headers={'Origin':'http://testserver'},follow_redirects=False).status_code==303
        assert client.post('/api/queue/claim',headers=HEADERS).json()['provider_recovery']

    def test_cross_origin_retry_cannot_advance_paid_recovery(self,outage_client):
        client,path=outage_client;client.post('/api/queue/claim',headers=HEADERS);hold(client)
        with sqlite3.connect(path) as db:before=db.execute('SELECT retry_after FROM provider_outages').fetchone()[0]
        assert client.post('/provider-outages/gemini/retry',headers={'Origin':'https://attacker.invalid'},follow_redirects=False).status_code==403
        assert client.post('/provider-outages/gemini/retry',follow_redirects=False).status_code==403
        with sqlite3.connect(path) as db:assert db.execute('SELECT retry_after FROM provider_outages').fetchone()[0]==before

    def test_success_without_provider_use_does_not_clear_the_outage(self,outage_client):
        client,path=outage_client;client.post('/api/queue/claim',headers=HEADERS);hold(client);expire_cooldown(path)
        probe=client.post('/api/queue/claim',headers=HEADERS).json()
        response=client.post('/api/queue/job0/complete',headers=HEADERS,json={'processed_audio_key':'synthetic/clean.mp3','processed_duration':120,'ads_removed_seconds':0,'recovery_tokens':probe['provider_recovery'],'provider_successes':[]})
        assert response.status_code==200
        with sqlite3.connect(path) as db:
            assert db.execute('SELECT active,probe_token FROM provider_outages').fetchone()==(1,None)
            assert db.execute("SELECT status,retry_count FROM episodes WHERE id='job0'").fetchone()==('complete',1)
        assert client.post('/api/queue/claim',headers=HEADERS).json() is None

    def test_multiple_provider_pauses_share_one_probe_and_partial_success(self,outage_client):
        client,path=outage_client
        client.post('/api/queue/claim',headers=HEADERS);client.post('/api/queue/claim',headers=HEADERS)
        hold(client)
        assert client.post('/api/queue/job1/provider-hold',headers=HEADERS,json={'provider':'openai','category':'authentication','error':'invalid key'}).status_code==200
        expire_cooldown(path);probe=client.post('/api/queue/claim',headers=HEADERS).json()
        assert {p['provider'] for p in probe['provider_recovery']}=={'gemini','openai'}
        response=client.post('/api/queue/job0/complete',headers=HEADERS,json={'processed_audio_key':'synthetic/clean.mp3','processed_duration':120,'ads_removed_seconds':0,'recovery_tokens':probe['provider_recovery'],'provider_successes':['gemini']})
        assert response.status_code==200
        with sqlite3.connect(path) as db:assert dict(db.execute('SELECT provider,active FROM provider_outages'))=={'gemini':0,'openai':1}
        assert client.post('/api/queue/claim',headers=HEADERS).json() is None
        expire_cooldown(path);next_probe=client.post('/api/queue/claim',headers=HEADERS).json()
        assert next_probe['id']=='job1' and [p['provider'] for p in next_probe['provider_recovery']]==['openai']
        assert complete(client,next_probe).status_code==200

    def test_duplicate_and_inflight_holds_do_not_slide_recovery_deadline(self,outage_client):
        client,path=outage_client
        client.post('/api/queue/claim',headers=HEADERS);client.post('/api/queue/claim',headers=HEADERS)
        hold(client)
        with sqlite3.connect(path) as db:before=db.execute('SELECT retry_after,failure_count FROM provider_outages').fetchone()
        hold(client);hold(client,episode='job1')
        with sqlite3.connect(path) as db:
            assert db.execute('SELECT retry_after,failure_count FROM provider_outages').fetchone()==before
            assert db.execute('SELECT count(*) FROM episode_provider_holds').fetchone()[0]==2
            assert [v for (v,) in db.execute('SELECT retry_count FROM episodes ORDER BY id')]==[1,1,1]

    def test_only_current_probe_can_renew_its_recovery_lease(self,outage_client):
        client,path=outage_client;client.post('/api/queue/claim',headers=HEADERS);hold(client);expire_cooldown(path)
        stale=client.post('/api/queue/claim',headers=HEADERS).json()
        with sqlite3.connect(path) as db:db.execute("UPDATE episodes SET claimed_at='2000-01-01 00:00:00' WHERE id='job0'")
        client.post('/api/queue/claim',headers=HEADERS);expire_cooldown(path)
        current=client.post('/api/queue/claim',headers=HEADERS).json()
        with sqlite3.connect(path) as db:before=db.execute("SELECT claimed_at FROM episodes WHERE id=?",(current["id"],)).fetchone()[0]
        for proofs in [[],stale['provider_recovery']]:
            assert client.post(f"/api/queue/{current['id']}/progress",headers=HEADERS,json={'step':'transcribing','percent':10,'recovery_tokens':proofs}).status_code==409
        with sqlite3.connect(path) as db:assert db.execute("SELECT claimed_at FROM episodes WHERE id=?",(current["id"],)).fetchone()[0]==before
        assert client.post(f"/api/queue/{current['id']}/progress",headers=HEADERS,json={'step':'transcribing','percent':50,'recovery_tokens':current['provider_recovery']}).status_code==200
        assert client.post('/api/queue/claim',headers=HEADERS).json() is None
        with sqlite3.connect(path) as db:assert db.execute("SELECT retry_count FROM episodes WHERE id='job0'").fetchone()[0]==1


@pytest.mark.parametrize('rep',[1,2])
def test_mixed_success_receipts_clear_only_the_actually_leased_provider(outage_client,rep):
    """A healthy second provider's successful use cannot reject recovery."""
    client,path=outage_client
    client.post('/api/queue/claim',headers=HEADERS);hold(client);expire_cooldown(path)
    probe=client.post('/api/queue/claim',headers=HEADERS).json()
    response=client.post('/api/queue/job0/complete',headers=HEADERS,json={
        'processed_audio_key':'synthetic/clean.mp3','processed_duration':120,
        'ads_removed_seconds':0,'recovery_tokens':probe['provider_recovery'],
        'provider_successes':['gemini','openai'],
    })
    assert response.status_code==200
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status,retry_count FROM episodes WHERE id='job0'").fetchone()==('complete',1)
        assert db.execute('SELECT provider,active FROM provider_outages').fetchall()==[('gemini',0)]
    assert client.post('/api/queue/claim',headers=HEADERS).json()['id']=='job1'


@pytest.mark.parametrize('rep',[1,2])
def test_failed_probe_rotates_to_other_pending_episodes_without_retry_cost(outage_client,rep):
    """Two broken sources cannot starve a third eligible recovery episode."""
    client,path=outage_client
    client.post('/api/queue/claim',headers=HEADERS);hold(client)
    picks=[]
    for expected in ['job0','job1','job2','job0']:
        expire_cooldown(path);probe=client.post('/api/queue/claim',headers=HEADERS).json();picks.append(probe['id'])
        assert probe['id']==expected,picks
        assert client.post('/api/queue/claim',headers=HEADERS).json() is None
        assert client.post(f"/api/queue/{probe['id']}/fail",headers=HEADERS,json={'error':'synthetic download404 before provider','recovery_tokens':probe['provider_recovery']}).status_code==200
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT retry_count FROM episodes ORDER BY id').fetchall()==[(1,),(1,),(1,)]
        assert db.execute('SELECT active,probe_token FROM provider_outages').fetchone()==(1,None)


@pytest.mark.parametrize('rep',[1,2])
def test_other_episode_can_verify_provider_after_poisoned_held_source(outage_client,rep):
    """The provider can recover even while its original episode stays held."""
    client,path=outage_client
    client.post('/api/queue/claim',headers=HEADERS);hold(client);expire_cooldown(path)
    bad=client.post('/api/queue/claim',headers=HEADERS).json()
    client.post('/api/queue/job0/fail',headers=HEADERS,json={'error':'synthetic audio404','recovery_tokens':bad['provider_recovery']})
    expire_cooldown(path);healthy=client.post('/api/queue/claim',headers=HEADERS).json()
    assert healthy['id']=='job1'
    assert complete(client,healthy).status_code==200
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status,retry_count FROM episodes WHERE id='job0'").fetchone()==('pending',1)
        assert db.execute('SELECT active FROM provider_outages').fetchone()[0]==0
    assert client.post('/api/queue/claim',headers=HEADERS).json()['id']=='job0'
