"""Background ingestion jobs and the index read-write lock (no models)."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from src.utils.rwlock import ReadWriteLock


def finished(client, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/ingest/{job_id}").json()
        if job["finished_at"]:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


@pytest.fixture
def api(make_rag):
    from src.api.app import create_app

    rag, chat = make_rag()
    app = create_app(rag_type="naive", vector_store_provider="faiss")
    app.state.rag = rag
    app.state.rag_type = "advanced"
    return TestClient(app), rag


def upload(name, text):
    return ("files", (name, text.encode("utf-8"), "text/markdown"))


def test_ingest_returns_a_job_and_indexes_in_the_background(api):
    client, rag = api

    response = client.post("/ingest", files=[upload("son_tinh.md", "# Sơn Tinh\n\nSơn Tinh dời núi chặn lũ.")])

    assert response.status_code == 202
    job = response.json()
    assert job["status"] in ("queued", "running", "succeeded")
    assert response.headers["location"] == f"/ingest/{job['job_id']}"
    job = finished(client, job["job_id"])
    assert job["status"] == "succeeded" and job["chunks"] >= 1 and job["files"] == ["son_tinh.md"]

    docs = rag.retrieve("Sơn Tinh dời núi", filter={"file_name": "son_tinh.md"})
    assert docs and docs[0].metadata["source"] == "upload/son_tinh.md"
    assert "absolute_source" not in docs[0].metadata
    assert [j["job_id"] for j in client.get("/ingest").json()] == [job["job_id"]]


def test_uploading_a_name_again_replaces_the_old_version(api):
    client, rag = api

    client.post("/ingest", params={"wait": True}, files=[upload("son_tinh.md", "# Sơn Tinh\n\nBản cũ: voi chín ngà.")])
    response = client.post("/ingest", params={"wait": True},
                           files=[upload("son_tinh.md", "# Sơn Tinh\n\nBản mới: gà chín cựa.")])

    assert response.status_code == 200 and response.json()["status"] == "succeeded"
    texts = " ".join(c.page_content for c in rag._chunks if c.metadata.get("relative_source") == "son_tinh.md")
    assert "Bản mới" in texts and "Bản cũ" not in texts
    stored = rag.vector_store.similarity_search("Bản cũ voi chín ngà", k=20, filter={"file_name": "son_tinh.md"})
    assert all("Bản cũ" not in doc.page_content for doc in stored)


def test_ingest_rejects_bad_uploads_and_reports_failed_jobs(api, monkeypatch, tmp_path):
    client, rag = api

    assert client.post("/ingest", files=[upload("virus.exe", "x")]).status_code == 415
    assert client.post("/ingest", files=[upload("a.md", "x"), upload("a.md", "y")]).status_code == 400
    assert client.get("/ingest/unknown").status_code == 404

    seen = []

    def broken(paths):
        seen.extend(paths.values())
        raise RuntimeError("disk full")

    monkeypatch.setattr(rag, "add_files", broken)
    response = client.post("/ingest", params={"wait": True}, files=[upload("a.md", "# A")])
    assert response.status_code == 500 and "disk full" in response.json()["detail"]
    job = client.get("/ingest").json()[0]
    assert job["status"] == "failed" and job["error"] == "RuntimeError: disk full"
    assert seen and not seen[0].exists()  # the uploaded copy is removed either way


def test_queries_run_while_a_job_loads_its_files(api, monkeypatch):
    client, rag = api
    loading, release = threading.Event(), threading.Event()
    load = rag.document_loader.load

    def slow_load(*args, **kwargs):
        loading.set()
        release.wait(timeout=5)
        return load(*args, **kwargs)

    monkeypatch.setattr(rag.document_loader, "load", slow_load)
    job = client.post("/ingest", files=[upload("son_tinh.md", "# Sơn Tinh\n\nSơn Tinh dời núi.")]).json()
    assert loading.wait(timeout=5)

    # Loading (and OCR) happens outside the index lock: searches still answer.
    results = []
    search = threading.Thread(target=lambda: results.append(rag.retrieve("Thạch Sanh", k=1)))
    search.start()
    search.join(timeout=1)
    assert results and results[0], "retrieve waited for the job's file loading"
    assert client.get(f"/ingest/{job['job_id']}").json()["status"] == "running"
    release.set()
    assert finished(client, job["job_id"])["status"] == "succeeded"


def test_read_write_lock_shares_reads_and_serializes_writes():
    lock = ReadWriteLock()
    events = []

    with lock.read():
        with_second_reader = threading.Event()

        def reader():
            with lock.read():
                with_second_reader.set()

        threading.Thread(target=reader).start()
        assert with_second_reader.wait(timeout=2)  # readers share

        def writer():
            with lock.write():
                events.append("write")

        writing = threading.Thread(target=writer)
        writing.start()
        time.sleep(0.05)
        assert events == []  # the writer waits for the reader

        late_reader_done = threading.Event()

        def late_reader():
            with lock.read():
                events.append("late read")
                late_reader_done.set()

        threading.Thread(target=late_reader).start()
        time.sleep(0.05)
        assert events == []  # a waiting writer goes before new readers

    writing.join(timeout=2)
    assert late_reader_done.wait(timeout=2)
    assert events == ["write", "late read"]
