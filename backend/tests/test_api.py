def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_questions_seeded(client):
    r = client.get("/api/questions")
    assert r.status_code == 200 and len(r.json()) == 5
    assert r.json()[0]["text"].startswith("1분간")


def test_cors_allows_localhost_3000(client):
    r = client.options(
        "/health", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"}
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
