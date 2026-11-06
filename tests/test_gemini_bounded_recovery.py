"""Owner-authorized Gemini retries have one shared finite decision budget."""
import socket
from pathlib import Path
from unittest.mock import Mock,patch

import httpx
import pytest
from google.genai import errors,types

from adnihilator.gemini_audio import GeminiAudioClient
from adnihilator.gemini_chunked import detect_ads_in_chunk
from tests.test_provider_completion_safety import google_response,google_sdk


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """All provider/audio boundaries are local models and mocks."""
    def denied(*args,**kwargs):raise AssertionError('No network in retry regressions')
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr(socket,'getaddrinfo',denied)


def sdk_error(code:int,message:str='temporary unavailable') -> Exception:
    """Use real SDK error status fields."""
    cls=errors.ServerError if code>=500 else errors.ClientError
    return cls(code,{'error':{'code':code,'message':message}})


def execute(sequence:list,mode:str='chunk',max_attempts:int=3):
    """Run the actual single/chunk detection around mocked transport calls."""
    sdk=google_sdk(google_response({'ads':[]}))
    sdk.models.generate_content.side_effect=sequence
    with patch('time.sleep') as sleep:
        if mode=='chunk':
            result=detect_ads_in_chunk(sdk,Path('synthetic.mp3'),0,120,'gemini-2.5-flash',max_retries=max_attempts)
        else:
            client=GeminiAudioClient('synthetic',model='gemini-2.5-flash');client._client=sdk
            result=client.detect_ads(Path('synthetic.mp3'),duration=120)[0]
    return result,sdk,sleep


class TestBoundedTransientRecovery:
    @pytest.mark.parametrize('mode',['single','chunk'])
    @pytest.mark.parametrize('error',[sdk_error(c) for c in [429,500,502,503,504]]+[httpx.ReadTimeout('timeout'),httpx.ConnectError('connection interrupted'),ConnectionResetError('reset')])
    def test_transient_then_stop_recovers_with_bounded_backoff(self,mode,error):
        result,sdk,sleep=execute([error,google_response({'ads':[]})],mode)
        assert result==[] and sdk.models.generate_content.call_count==2
        assert [c.args[0] for c in sleep.call_args_list]==[30]

    @pytest.mark.parametrize('mode',['single','chunk'])
    def test_exhaustion_never_exceeds_three_model_attempts(self,mode):
        sdk=google_sdk(google_response({'ads':[]}));sdk.models.generate_content.side_effect=[sdk_error(503)]*4
        client=GeminiAudioClient('synthetic',model='gemini-2.5-flash');client._client=sdk
        with patch('time.sleep') as sleep,pytest.raises(Exception):
            if mode=='chunk':detect_ads_in_chunk(sdk,Path('synthetic.mp3'),0,120,'gemini-2.5-flash',max_retries=99)
            else:client.detect_ads(Path('synthetic.mp3'),duration=120)
        assert sdk.models.generate_content.call_count==3
        assert [c.args[0] for c in sleep.call_args_list]==[30,60]

    @pytest.mark.parametrize('mode',['single','chunk'])
    @pytest.mark.parametrize('code,message',[(402,'credits depleted'),(429,'prepaid credits depleted'),(400,'API_KEY_INVALID'),(401,'invalid key'),(403,'forbidden'),(404,'model unavailable')])
    def test_permanent_failure_never_retries(self,mode,code,message):
        sdk=google_sdk(google_response({'ads':[]}));sdk.models.generate_content.side_effect=[sdk_error(code,message),google_response({'ads':[]})]
        client=GeminiAudioClient('synthetic',model='gemini-2.5-flash');client._client=sdk
        with patch('time.sleep') as sleep,pytest.raises(Exception):
            if mode=='chunk':detect_ads_in_chunk(sdk,Path('synthetic.mp3'),0,120,'gemini-2.5-flash')
            else:client.detect_ads(Path('synthetic.mp3'),duration=120)
        assert sdk.models.generate_content.call_count==1 and sleep.call_count==0

    @pytest.mark.parametrize('mode',['single','chunk'])
    def test_one_invalid_json_reask_can_recover(self,mode):
        result,sdk,_=execute([google_response({}),google_response({'ads':[]})],mode)
        assert result==[] and sdk.models.generate_content.call_count==2

    @pytest.mark.parametrize('mode',['single','chunk'])
    def test_invalid_json_reask_shares_the_three_attempt_budget(self,mode):
        result,sdk,sleep=execute([sdk_error(503),google_response({}),google_response({'ads':[]})],mode)
        assert result==[] and sdk.models.generate_content.call_count==3
        assert [c.args[0] for c in sleep.call_args_list]==[30]

    @pytest.mark.parametrize('mode',['single','chunk'])
    def test_second_invalid_json_cannot_trigger_another_reask(self,mode):
        sdk=google_sdk(google_response({}));sdk.models.generate_content.side_effect=[google_response({})]*3
        client=GeminiAudioClient('synthetic',model='gemini-2.5-flash');client._client=sdk
        with patch('time.sleep'),pytest.raises(Exception):
            if mode=='chunk':detect_ads_in_chunk(sdk,Path('synthetic.mp3'),0,120,'gemini-2.5-flash')
            else:client.detect_ads(Path('synthetic.mp3'),duration=120)
        assert sdk.models.generate_content.call_count==2

    @pytest.mark.parametrize('mode',['single','chunk'])
    def test_valid_json_with_nonstop_metadata_is_not_a_json_reask(self,mode):
        sdk=google_sdk(google_response({'ads':[]},finish=types.FinishReason.MAX_TOKENS))
        client=GeminiAudioClient('synthetic',model='gemini-2.5-flash');client._client=sdk
        with patch('time.sleep'),pytest.raises(Exception):
            if mode=='chunk':detect_ads_in_chunk(sdk,Path('synthetic.mp3'),0,120,'gemini-2.5-flash')
            else:client.detect_ads(Path('synthetic.mp3'),duration=120)
        assert sdk.models.generate_content.call_count==1


class TestActualSDKAttemptBudget:
    @pytest.mark.parametrize('mode',['single','chunk'])
    def test_sdk_transport_cannot_multiply_application_attempts(self,mode):
        """Actual SDK HTTP attempts, not just a mocked generate method counter."""
        from adnihilator.gemini_common import create_genai_client
        calls=[]
        def respond(request):
            calls.append(request.url.path)
            return httpx.Response(503,json={'error':{'code':503,'message':'temporary unavailable','status':'UNAVAILABLE'}})
        sdk=create_genai_client('synthetic')
        hooks=sdk._api_client._httpx_client.event_hooks
        sdk._api_client._httpx_client.close()
        sdk._api_client._httpx_client=httpx.Client(transport=httpx.MockTransport(respond),event_hooks=hooks)
        sdk.files.upload=Mock(return_value=types.File(name='files/synthetic',uri='https://generativelanguage.googleapis.com/files/synthetic',mime_type='audio/mpeg',state=types.FileState.ACTIVE))
        try:
            with patch('time.sleep') as sleep,pytest.raises(Exception) as caught:
                if mode=='chunk':detect_ads_in_chunk(sdk,Path('synthetic.mp3'),0,120,'gemini-2.5-flash')
                else:
                    detector=GeminiAudioClient('synthetic',model='gemini-2.5-flash');detector._client=sdk
                    detector.detect_ads(Path('synthetic.mp3'),duration=120)
            assert len(calls)==3, (type(caught.value).__name__,str(caught.value).replace('synthetic','[FAKE]'))
            assert [c.args[0] for c in sleep.call_args_list]==[30,60]
        finally:sdk.close()

    def test_usage_counts_the_invalid_completed_response_and_corrected_reask(self):
        """Paid decision usage is accumulated even when its JSON needs correction."""
        from adnihilator.gemini_retry import retry_decision,InvalidGeminiDecision
        responses=[google_response({}),google_response({'ads':[]})]
        for response in responses:response.usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=11,candidates_token_count=3)
        source=Mock(side_effect=responses);usage={}
        def parse(response):
            if response is responses[0]:raise InvalidGeminiDecision('missing ads')
            return []
        assert retry_decision(source,parse,usage_totals=usage)==[]
        assert usage['input_tokens']==22 and usage['output_tokens']==6

    @pytest.mark.parametrize('code,expected_uploads',[(402,1),(401,1),(503,3)])
    def test_resumable_upload_sdk_cannot_hide_extra_retries(self,tmp_path,code,expected_uploads):
        """Resumable data uploads must expose errors to the same owner budget."""
        from adnihilator.gemini_common import create_genai_client
        from adnihilator.gemini_retry import retry_operation
        audio=tmp_path/'synthetic.mp3';audio.write_bytes(b'synthetic')
        uploads=[]
        def respond(request):
            if request.url.path.startswith('/upload/'):
                return httpx.Response(200,json={},headers={'x-goog-upload-url':'https://generativelanguage.googleapis.com/synthetic-upload'})
            assert request.url.path=='/synthetic-upload'
            uploads.append(request.url.path)
            return httpx.Response(code,json={'error':{'code':code,'message':'credits depleted' if code==402 else 'authentication failed' if code==401 else 'temporarily unavailable'}})
        sdk=create_genai_client('synthetic')
        hooks=sdk._api_client._httpx_client.event_hooks
        sdk._api_client._httpx_client.close();sdk._api_client._httpx_client=httpx.Client(transport=httpx.MockTransport(respond),event_hooks=hooks)
        try:
            with patch('time.sleep'),pytest.raises(Exception):retry_operation(lambda:sdk.files.upload(file=str(audio)))
            assert len(uploads)==expected_uploads
        finally:sdk.close()

    @pytest.mark.parametrize('mode',['single','chunk'])
    def test_successful_stop_with_blank_json_uses_only_one_reask(self,mode):
        blank=google_response({'ads':[]});blank.candidates[0].content.parts[0].text=' \n '
        result,sdk,_=execute([blank,google_response({'ads':[]})],mode)
        assert result==[] and sdk.models.generate_content.call_count==2

    @pytest.mark.parametrize('mode',['single','chunk'])
    def test_generic_generation_404_is_a_model_outage_without_message_matching(self,mode):
        from adnihilator.provider_errors import ProviderOutageError
        sdk=google_sdk(google_response({'ads':[]}));sdk.models.generate_content.side_effect=sdk_error(404,'Requested entity was not found')
        detector=GeminiAudioClient('synthetic',model='gemini-2.5-flash');detector._client=sdk
        with patch('time.sleep') as sleep,pytest.raises(ProviderOutageError) as caught:
            if mode=='chunk':detect_ads_in_chunk(sdk,Path('synthetic.mp3'),0,120,'gemini-2.5-flash')
            else:detector.detect_ads(Path('synthetic.mp3'),duration=120)
        assert caught.value.failure.category=='model' and sdk.models.generate_content.call_count==1
        sleep.assert_not_called()

    def test_generic_file_404_is_not_misclassified_as_model_outage(self):
        from adnihilator.provider_errors import ProviderOutageError
        sdk=google_sdk(google_response({'ads':[]}));sdk.files.upload.side_effect=sdk_error(404,'Requested entity was not found')
        detector=GeminiAudioClient('synthetic',model='gemini-2.5-flash');detector._client=sdk
        with patch('time.sleep') as sleep,pytest.raises(ValueError) as caught:detector.detect_ads(Path('synthetic.mp3'),duration=120)
        assert not isinstance(caught.value,ProviderOutageError)
        assert sdk.files.upload.call_count==1 and sdk.models.generate_content.call_count==0
        sleep.assert_not_called()
