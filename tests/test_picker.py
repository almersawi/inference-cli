import inference


def test_pick_model_returns_chosen_model():
    models = [
        {"model": "a", "base_url": "x", "api_key": "k"},
        {"model": "b", "base_url": "y", "api_key": "k"},
    ]
    captured = {}

    def fake_select(message, choices):
        captured["message"] = message
        captured["choices"] = choices
        # simulate user picking the second item
        return choices[1]

    result = inference.pick_model(models, _select=fake_select)
    assert result == {"model": "b", "base_url": "y", "api_key": "k"}
    assert captured["choices"][0] == "a"
    assert captured["choices"][1] == "b"
    assert captured["choices"][-1] == "+ add new model"


def test_pick_model_returns_add_sentinel_when_user_picks_add():
    models = [{"model": "a", "base_url": "x", "api_key": "k"}]

    def fake_select(message, choices):
        return "+ add new model"

    assert inference.pick_model(models, _select=fake_select) == "__add__"


def test_pick_model_works_with_empty_list():
    def fake_select(message, choices):
        return choices[0]

    assert inference.pick_model([], _select=fake_select) == "__add__"


def test_make_client_passes_base_url_and_api_key(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(inference, "OpenAI", FakeOpenAI)
    inference.make_client({"model": "m", "base_url": "http://x/v1", "api_key": "k"})
    assert captured == {"base_url": "http://x/v1", "api_key": "k"}
