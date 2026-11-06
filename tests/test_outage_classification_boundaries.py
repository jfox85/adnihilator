"""Invalid model data must never impersonate an account-wide SDK outage."""
import json
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from adnihilator.ad_llm import OpenAIClient
from adnihilator.config import Config
from adnihilator.gemini_audio import GeminiAudioClient
from adnihilator.gemini_chunked import GeminiChunkedClient
from adnihilator.provider_errors import ProviderOutageError
from tests.test_provider_completion_safety import google_response,google_sdk,openai_response

TEXT=['API_KEY_INVALID','Your prepayment credits are depleted','Error code: 404 model unavailable']


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args,**kwargs):raise AssertionError('No network in classification tests')
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr(socket,'getaddrinfo',denied)


@pytest.mark.parametrize('text',TEXT)
@pytest.mark.parametrize('mode',['single','chunked'])
def test_invalid_gemini_ads_content_cannot_trigger_global_pause(text,mode,tmp_path):
    sdk=google_sdk(google_response({'ads':text}))
    detector=(GeminiAudioClient if mode=='single' else GeminiChunkedClient)('synthetic',model='gemini-2.5-flash');detector._client=sdk
    with patch('adnihilator.gemini_chunked.split_audio_into_chunks',return_value=[(tmp_path/'synthetic.mp3',0,120)]),patch('time.sleep'),pytest.raises(Exception) as caught:
        detector.detect_ads(tmp_path/'synthetic.mp3',duration=120)
    assert not isinstance(caught.value,ProviderOutageError)
    assert sdk.models.generate_content.call_count==2


@pytest.mark.parametrize('text',TEXT)
def test_invalid_openai_merge_content_cannot_trigger_global_pause(text):
    with openai_response(json.dumps({'ads':text})) as sdk,patch('openai.OpenAI',return_value=sdk):
        with pytest.raises(Exception) as caught:OpenAIClient('synthetic').merge_and_refine({'duration':120},Config())
    assert not isinstance(caught.value,ProviderOutageError)
