"""Explicit offline migrations preserve durable outages and unrelated data."""
import hashlib
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.web.test_provider_outage_contract import outage_client,expire_cooldown,hold,HEADERS

SCRIPT=Path(__file__).parents[2]/'scripts/migrate_provider_outages.py'


def migrate(path:Path|None,env:dict[str,str]|None=None)->subprocess.CompletedProcess[str]:
    """Run the actual explicit CLI; never resolve or replace its interpreter."""
    assert SCRIPT.is_file(),'Explicit provider outage migration CLI is required'
    return subprocess.run([sys.executable,str(SCRIPT),*(['--db',str(path)] if path is not None else [])],env=env,text=True,capture_output=True)


def digest_rows(db:sqlite3.Connection,query:str)->str:
    """Compare private state without exposing opaque recovery proofs on failure."""
    return hashlib.sha256(repr(db.execute(query).fetchall()).encode()).hexdigest()


def r1_hold(client:TestClient,path:Path,leased:bool=False)->dict[str,str]:
    """Construct the exact pre-last_probe_at shape only in the disposable DB."""
    client.post('/api/queue/claim',headers=HEADERS);hold(client)
    with sqlite3.connect(path) as db:
        db.execute('ALTER TABLE episode_provider_holds DROP COLUMN last_probe_at')
        if leased:
            db.execute("UPDATE episodes SET status='processing',claimed_at=CURRENT_TIMESTAMP WHERE id='job0'")
            db.execute("UPDATE provider_outages SET probe_episode_id='job0',probe_token=?",('c'*32,))
        result = {table:digest_rows(db,f'SELECT * FROM {table}') for table in ['episodes','podcasts','provider_outages','episode_provider_holds']}
        result['unrelated_schema'] = digest_rows(db, "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE tbl_name NOT IN ('provider_outages','episode_provider_holds') ORDER BY type,name")
        return result


class TestProviderOutageMigration:
    def test_requires_explicit_existing_db_and_ignores_environment_default(self,tmp_path:Path)->None:
        """Missing paths must never create or silently select another database."""
        ignored=tmp_path/'environment.sqlite';missing=tmp_path/'missing.sqlite'
        env=dict(os.environ,DATABASE_PATH=str(ignored))
        assert migrate(None,env).returncode!=0
        assert migrate(missing,env).returncode!=0
        assert not missing.exists() and not ignored.exists()
        assert migrate(tmp_path,env).returncode!=0

    @pytest.mark.parametrize('rep',[1,2])
    def test_existing_empty_db_gets_only_two_additive_tables_idempotently(self,tmp_path:Path,rep:int)->None:
        path=tmp_path/'existing.sqlite';path.touch()
        assert migrate(path).returncode==0
        with sqlite3.connect(path) as db:
            tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert tables=={'provider_outages','episode_provider_holds'}
            assert [row[1] for row in db.execute('PRAGMA table_info(episode_provider_holds)')].count('last_probe_at')==1
            schema=digest_rows(db,"SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")
        assert migrate(path).returncode==0
        with sqlite3.connect(path) as db:assert digest_rows(db,"SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")==schema

    @pytest.mark.parametrize('rep',[1,2])
    @pytest.mark.parametrize('leased',[False,True])
    def test_r1_upgrade_preserves_holds_budgets_and_live_lease(self,outage_client:tuple[TestClient,Path],rep:int,leased:bool)->None:
        client,path=outage_client;before=r1_hold(client,path,leased)
        assert migrate(path).returncode==0
        with sqlite3.connect(path) as db:
            for table in ['episodes','podcasts','provider_outages']:assert digest_rows(db,f'SELECT * FROM {table}')==before[table]
            assert digest_rows(db, "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE tbl_name NOT IN ('provider_outages','episode_provider_holds') ORDER BY type,name")==before['unrelated_schema']
            assert digest_rows(db,'SELECT episode_id,provider,created_at FROM episode_provider_holds')==before['episode_provider_holds']
            assert db.execute('SELECT last_probe_at FROM episode_provider_holds').fetchall()==[(None,)]
        from web.app import app
        with TestClient(app) as restarted:
            assert restarted.post('/login',data={'username':'outage-admin','password':'synthetic-admin'},follow_redirects=False).status_code==303
            assert restarted.get('/queue').status_code==200
            assert restarted.get('/').status_code==200
            if leased:
                assert restarted.post('/api/queue/claim-with-provider-recovery',headers=HEADERS).json() is None
                response=restarted.post('/api/queue/job0/progress',headers=HEADERS,json={'step':'detecting','recovery_tokens':[{'provider':'gemini','token':'c'*32}]})
                assert response.status_code==200
            else:
                expire_cooldown(path)
                probe=restarted.post('/api/queue/claim-with-provider-recovery',headers=HEADERS).json()
                assert probe['id']=='job0'
        with sqlite3.connect(path) as db:current={table:digest_rows(db,f'SELECT * FROM {table}') for table in ['episodes','provider_outages','episode_provider_holds']}
        assert migrate(path).returncode==0
        with sqlite3.connect(path) as db:
            for table in current:assert digest_rows(db,f'SELECT * FROM {table}')==current[table]

    @pytest.mark.parametrize('rep',[1,2])
    def test_two_cli_processes_serialize_the_same_legacy_upgrade(self,outage_client:tuple[TestClient,Path],rep:int)->None:
        client,path=outage_client;before=r1_hold(client,path)
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:migrate(path),range(2)))
        assert [result.returncode for result in results]==[0,0]
        with sqlite3.connect(path) as db:
            assert [row[1] for row in db.execute('PRAGMA table_info(episode_provider_holds)')].count('last_probe_at')==1
            assert digest_rows(db,'SELECT * FROM episodes')==before['episodes']
            assert digest_rows(db,'SELECT * FROM provider_outages')==before['provider_outages']
            assert digest_rows(db,'SELECT episode_id,provider,created_at FROM episode_provider_holds')==before['episode_provider_holds']

    @pytest.mark.parametrize('rep',[1,2])
    def test_invalid_schema_rolls_back_new_ddl_without_dropping_any_rows(self,tmp_path:Path,rep:int)->None:
        path=tmp_path/'invalid-legacy.sqlite'
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE episode_provider_holds(episode_id TEXT PRIMARY KEY)")
            db.execute("INSERT INTO episode_provider_holds VALUES('held-episode')")
            db.execute("CREATE TABLE protected_transcripts(id TEXT PRIMARY KEY,body TEXT CHECK(length(body)>0))")
            db.execute("INSERT INTO protected_transcripts VALUES('private','keep canonical text')")
            before=digest_rows(db,"SELECT name,sql FROM sqlite_master ORDER BY name")
        assert migrate(path).returncode!=0
        with sqlite3.connect(path) as db:
            assert digest_rows(db,"SELECT name,sql FROM sqlite_master ORDER BY name")==before
            assert db.execute('SELECT count(*) FROM episode_provider_holds').fetchone()[0]==1
            assert db.execute('SELECT body FROM protected_transcripts').fetchone()[0]=='keep canonical text'

    def test_non_sqlite_file_is_left_byte_identical(self,tmp_path:Path)->None:
        path=tmp_path/'not-a-db';path.write_bytes(b'non SQLite sentinel')
        before=path.read_bytes();assert migrate(path).returncode!=0;assert path.read_bytes()==before
