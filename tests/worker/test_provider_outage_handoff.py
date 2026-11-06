"""Real worker/client/server handoff preserves global-outage retry budgets."""
import sqlite3
import socket
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from google.genai import errors
from openai import OpenAI as SDKOpenAI

from adnihilator.gemini_audio import GeminiAudioClient
from tests.test_provider_completion_safety import google_response,google_sdk
from tests.web.test_provider_outage_contract import outage_client,expire_cooldown,HEADERS
from tests.worker.test_authoritative_transcripts import worker as bare_worker,pipeline,transcript
from worker.client import WorkerClient


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args,**kwargs):raise AssertionError('No external network in outage handoff tests')
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr(socket,'getaddrinfo',denied)


def configured_worker(monkeypatch,client):
    """Use actual API methods over in-process HTTP, never construct a daemon."""
    api=WorkerClient('http://testserver','synthetic-worker');api.client.close();api.client=client
    result=bare_worker.__wrapped__(monkeypatch);result.api_client=api
    result.config.llm.provider='none';result.config.gemini.enabled=True
    return result


class TestProviderOutageHandoff:
    @pytest.mark.parametrize('parallel',[False,True])
    def test_global_gemini_failure_holds_then_actual_success_recovers(self,outage_client,monkeypatch,parallel):
        client,path=outage_client;worker=configured_worker(monkeypatch,client)
        worker.config.detection.parallel_enabled=parallel
        sdk=google_sdk(google_response({'ads':[]}));sdk.models.generate_content.side_effect=errors.ClientError(402,{'error':{'message':'credits depleted'}})
        detector=GeminiAudioClient('synthetic',model='gemini-2.5-flash');detector._client=sdk;worker._get_gemini_client.return_value=detector
        with pipeline(worker,'whisper',transcript(ad=False)) as splice:
            assert worker.run_once() is True
            splice.assert_not_called();worker.r2_client.upload_file.assert_not_called()
            assert sdk.models.generate_content.call_count==1
            with sqlite3.connect(path) as db:
                assert db.execute("SELECT status,retry_count FROM episodes WHERE id='job0'").fetchone()==('pending',1)
                assert db.execute('SELECT provider,category FROM provider_outages').fetchone()==('gemini','billing')
            assert worker.run_once() is False
            expire_cooldown(path);sdk.models.generate_content.side_effect=None
            assert worker.run_once() is True
            assert sdk.models.generate_content.call_count==2
            with sqlite3.connect(path) as db:
                assert db.execute("SELECT status,retry_count FROM episodes WHERE id='job0'").fetchone()==('complete',1)
                assert db.execute('SELECT active FROM provider_outages').fetchone()[0]==0

    def test_required_openai_auth_failure_uses_hold_not_episode_failure(self,outage_client,monkeypatch):
        client,path=outage_client;worker=configured_worker(monkeypatch,client)
        worker.config.gemini.enabled=False;worker.config.llm.provider='openai'
        def reject(request):return httpx.Response(401,json={'error':{'message':'invalid API key','type':'invalid_request_error','code':'invalid_api_key'}})
        sdk=SDKOpenAI(api_key='synthetic',base_url='https://provider.invalid/v1',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(reject)))
        try:
            with patch('openai.OpenAI',return_value=sdk),pipeline(worker,'whisper',transcript()) as splice:
                assert worker.run_once() is True
                splice.assert_not_called();worker.r2_client.upload_file.assert_not_called()
            with sqlite3.connect(path) as db:
                assert db.execute("SELECT status,retry_count FROM episodes WHERE id='job0'").fetchone()==('pending',1)
                assert db.execute('SELECT provider,category FROM provider_outages').fetchone()==('openai','authentication')
        finally:sdk.close()

    def test_completed_fastpath_does_not_falsely_verify_an_unused_provider(self,outage_client,monkeypatch):
        client,path=outage_client;worker=configured_worker(monkeypatch,client)
        sdk=google_sdk(google_response({'ads':[]}));sdk.models.generate_content.side_effect=errors.ClientError(402,{'error':{'message':'credits depleted'}})
        detector=GeminiAudioClient('synthetic',model='gemini-2.5-flash');detector._client=sdk;worker._get_gemini_client.return_value=detector
        with pipeline(worker,'whisper',transcript(ad=False)):
            assert worker.run_once() is True
            expire_cooldown(path)
            from types import SimpleNamespace
            stamp=SimpleNamespace(start=10,end=50,confidence=.95,label='Synthetic',extraction_method='test')
            with patch('worker.daemon.extract_ad_timestamps',return_value=[stamp]):assert worker.run_once() is True
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT status FROM episodes WHERE id='job0'").fetchone()[0]=='complete'
            assert db.execute('SELECT active,probe_token FROM provider_outages').fetchone()==(1,None)
        assert sdk.models.generate_content.call_count==1

    def test_versioned_claim_never_falls_back_to_an_older_server(self):
        calls=[]
        def older(request):calls.append(request.url.path);return httpx.Response(404,json={'detail':'Not Found'})
        api=WorkerClient('https://old.invalid','synthetic-worker');api.client.close();api.client=httpx.Client(transport=httpx.MockTransport(older))
        try:assert api.claim() is None
        finally:api.close()
        assert calls==['/api/queue/claim-with-provider-recovery']

    def test_unacknowledged_hold_blocks_more_claims_until_replayed(self,monkeypatch):
        """An API interruption must not turn one provider failure into more jobs."""
        from unittest.mock import Mock
        from adnihilator.provider_errors import ProviderOutageError,classify_provider_error
        from adnihilator.config import Config
        from worker.daemon import WorkerDaemon
        from worker.client import EpisodeJob
        worker=WorkerDaemon.__new__(WorkerDaemon);worker.config=Config();worker.api_client=Mock()
        worker.api_client.claim.return_value=EpisodeJob(id='job',podcast_id='pod',guid='guid',title='Synthetic',original_audio_url='https://example.invalid/audio.mp3')
        worker.process_job=Mock(side_effect=ProviderOutageError('gemini',classify_provider_error('402 credits depleted')))
        worker.api_client.hold.side_effect=[httpx.ConnectError('API unavailable'),None]
        with pytest.raises(Exception):worker.run_once()
        assert worker.api_client.claim.call_count==1
        assert worker.run_once() is True
        assert worker.api_client.claim.call_count==1 and worker.process_job.call_count==1
        worker.api_client.fail.assert_not_called()

    def test_hold_requires_an_actual_server_acknowledgment(self):
        """A successful proxy response is not proof a durable hold was written."""
        api=WorkerClient('https://server.invalid','synthetic-worker');api.client.close()
        api.client=httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={'status':'not-held'})))
        try:
            with pytest.raises(RuntimeError,match='acknowledge'):api.hold('job','gemini','billing','credits depleted')
        finally:api.close()

    def test_transient_exhaustion_during_probe_preserves_the_held_retry_budget(self,outage_client,monkeypatch):
        client,path=outage_client;worker=configured_worker(monkeypatch,client)
        sdk=google_sdk(google_response({'ads':[]}));sdk.models.generate_content.side_effect=errors.ClientError(402,{'error':{'message':'credits depleted'}})
        detector=GeminiAudioClient('synthetic',model='gemini-2.5-flash');detector._client=sdk;worker._get_gemini_client.return_value=detector
        with pipeline(worker,'whisper',transcript(ad=False)),patch('time.sleep'):
            assert worker.run_once() is True
            expire_cooldown(path);sdk.models.generate_content.side_effect=errors.ServerError(503,{'error':{'message':'unavailable'}})
            assert worker.run_once() is True
        assert sdk.models.generate_content.call_count==4
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT status,retry_count FROM episodes WHERE id='job0'").fetchone()==('pending',1)
            assert db.execute('SELECT active,probe_token,failure_count FROM provider_outages').fetchone()==(1,None,2)

    def test_definitive_stale_hold_rejection_does_not_keep_old_context(self):
        api=WorkerClient('https://server.invalid','synthetic-worker');api.client.close()
        api._recovery_tokens['job']=[{'provider':'gemini','token':'a'*32}]
        api.client=httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(409,json={'detail':'Recovery lease is no longer current'})))
        try:
            api.hold('job','gemini','billing','credits depleted')
            assert 'job' not in api._recovery_tokens
        finally:api.close()
