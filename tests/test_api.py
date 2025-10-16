from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def _meta():
    r = client.get("/metadata")
    assert r.status_code == 200
    return r.json()


def _feature_order():
    return _meta()["feature_order"]


def _default_threshold():
    # May be None with v0.1 artifacts; present in v0.2+
    return _meta().get("high_risk_threshold")


def test_root_redirects_to_docs():
    r = client.get("/", allow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers.get("location") == "/docs"


def test_health_live_ready_ok():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    # Acceptance sugar
    assert "model_version" in j

    r = client.get("/live")
    assert r.status_code == 200
    assert r.json()["status"] == "live"

    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_metadata_has_required_fields():
    j = _meta()
    for k in ("model", "version", "feature_order"):
        assert k in j
    assert isinstance(j["feature_order"], list) and len(j["feature_order"]) == 10


def test_predict_one_ok():
    # Build a valid record with all 10 features
    fo = _feature_order()
    rec = {k: 0.0 for k in fo}

    # The /predict_one endpoint expects flat fields, not nested "records"
    payload = {
        "age": rec["age"], "sex": rec["sex"], "bmi": rec["bmi"], "bp": rec["bp"],
        "s1": rec["s1"], "s2": rec["s2"], "s3": rec["s3"], "s4": rec["s4"], "s5": rec["s5"], "s6": rec["s6"]
    }
    r = client.post("/predict_one", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "prediction" in j
    assert isinstance(j["prediction"], (int, float))


def test_predict_records_ok_with_flags():
    fo = _feature_order()
    thr = _default_threshold()

    rec = {k: 0.0 for k in fo}
    body = {"records": [rec], "include_flag": True}
    # If artifact has no default threshold (v0.1), provide one
    if thr is None:
        body["threshold"] = 0.0

    r = client.post("/predict", json=body)
    assert r.status_code == 200
    j = r.json()
    assert "scores" in j and len(j["scores"]) == 1
    assert j["feature_order"] == fo
    # If we asked for flags, we should get them (either default or override)
    assert "flags" in j and len(j["flags"]) == 1
    assert "threshold_used" in j


def test_predict_vectors_ok_with_flags():
    fo = _feature_order()
    thr = _default_threshold()

    vec = [0.0] * len(fo)
    body = {"vectors": [vec], "include_flag": True}
    if thr is None:
        body["threshold"] = 0.0

    r = client.post("/predict_vectors", json=body)
    assert r.status_code == 200
    j = r.json()
    assert "scores" in j and len(j["scores"]) == 1
    assert j["feature_order"] == fo
    assert "flags" in j and len(j["flags"]) == 1
    assert "threshold_used" in j


def test_predict_missing_feature_400():
    fo = _feature_order()
    # Drop one key to simulate a bad record
    bad = {k: 0.0 for k in fo if k != fo[-1]}
    r = client.post("/predict", json={"records": [bad], "include_flag": False})
    assert r.status_code == 400
    j = r.json()
    assert "detail" in j and j["detail"].get("error") in {"missing_features", "missing_features"}  # name check
    assert fo[-1] in j["detail"].get("missing", [])


def test_predict_vectors_bad_length_400():
    fo = _feature_order()
    # Only 9 features instead of 10
    bad_vec = [0.0] * (len(fo) - 1)
    r = client.post("/predict_vectors", json={"vectors": [bad_vec], "include_flag": False})
    assert r.status_code == 400
    j = r.json()
    assert j.get("detail", {}).get("error") == "bad_vector_length"
    assert j["detail"].get("expected") == len(fo)
