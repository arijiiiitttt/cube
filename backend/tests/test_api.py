def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_no_auth_routes_exposed(client):
    assert client.post("/auth/login", json={}).status_code == 404


def test_estimate_endpoint(client):
    r = client.post(
        "/training/estimate",
        json={"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "dataset_total_tokens": 5_000_000},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["estimated_seconds"] > 0
    assert body["model_size_b_params"] == 1.5


def test_estimate_validates_input(client):
    r = client.post("/training/estimate", json={"base_model": "x", "dataset_total_tokens": 0})
    assert r.status_code == 422


def test_start_training_rejects_invalid_config(client):
    r = client.post("/training/start", json={"dataset_id": "d", "use_lora": False, "load_in_4bit": True})
    assert r.status_code == 422


def test_list_models_empty(client):
    r = client.get("/models/")
    assert r.status_code == 200
    assert r.json() == []


def test_unknown_dataset_is_404(client):
    r = client.get("/dataset/does-not-exist/status")
    assert r.status_code == 404
