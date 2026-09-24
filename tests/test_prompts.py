import pytest

from specheads.train.prompts import (
    EVAL_SOURCES,
    SOURCES,
    PromptSet,
    balance_by_tokens,
    decontaminate,
    extract_prompt,
    ngrams,
    normalise,
)


def test_normalise_strips_punctuation_and_case():
    assert normalise("Hello, World! 42x") == ["hello", "world", "42x"]


def test_ngrams_of_short_text_is_the_whole_text():
    assert ngrams(["a", "b"], 13) == {("a", "b")}
    assert ngrams([], 13) == set()


def test_decontaminate_drops_an_overlapping_prompt():
    shared = " ".join(f"w{i}" for i in range(20))
    training = [shared, "something completely different and unrelated"]
    kept, dropped = decontaminate(training, [shared])
    assert dropped == 1
    assert kept == ["something completely different and unrelated"]


def test_decontaminate_tolerates_reworded_overlap():
    """A prefix/suffix change must not hide a shared 13-gram."""
    body = " ".join(f"token{i}" for i in range(30))
    kept, dropped = decontaminate([f"Please solve: {body} thanks"], [body])
    assert dropped == 1 and kept == []


def test_decontaminate_keeps_short_incidental_matches():
    """Ordinary phrasing shorter than the n-gram must not trigger a drop."""
    kept, dropped = decontaminate(["write a function"], ["write a function that sorts"])
    assert dropped == 0 and len(kept) == 1


def test_decontaminate_with_no_eval_prompts_is_a_passthrough():
    kept, dropped = decontaminate(["a", "b"], [])
    assert kept == ["a", "b"] and dropped == 0


def test_balance_by_tokens_equalises_tokens_not_prompts():
    """Code answers run long, so equal prompt counts would skew the mixture."""
    sets = {"chat": ["a", "b", "c", "d"], "code": ["x", "y"]}
    counts = {"chat": [10, 10, 10, 10], "code": [40, 40]}
    out = balance_by_tokens(sets, counts, budget=40)
    assert len(out["chat"]) == 4   # 4 x 10 tokens
    assert len(out["code"]) == 1   # 1 x 40 tokens


def test_prompt_set_hash_is_order_sensitive_and_stable():
    a = PromptSet("chat", "s", "mit", ["one", "two"])
    b = PromptSet("chat", "s", "mit", ["one", "two"])
    c = PromptSet("chat", "s", "mit", ["two", "one"])
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash


def test_extract_prompt_handles_each_source_shape():
    assert extract_prompt("chat", {"messages": [{"role": "user", "content": "hi"}]}) == "hi"
    assert extract_prompt("chat", {"prompt": ["first turn", "second"]}) == "first turn"
    assert extract_prompt("code", {"question": "q"}) == "q"
    assert extract_prompt("code", {"prompt": "def f():"}) == "def f():"
    assert extract_prompt("math", {"question": "2+2?"}) == "2+2?"


def test_extract_prompt_rejects_unknown_domain():
    with pytest.raises(ValueError, match="unknown domain"):
        extract_prompt("biology", {})


def test_declared_sources_match_the_resolved_plan():
    assert SOURCES["code"]["id"] == "glaiveai/glaive-code-assistant"
    assert SOURCES["code"]["license"] == "apache-2.0"
    # GSM8K train for training, test for eval -- disjoint by construction.
    assert SOURCES["math"]["split"] == "train"
    assert EVAL_SOURCES["math"]["split"] == "test"
