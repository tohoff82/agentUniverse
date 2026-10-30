"""Hermetic qualification oracle for the Implementation-Supervision contour.

IS-07 is intentionally not asserted: ISWorkPattern has no terminal/stop signal in
its public or internal contract at this baseline, so terminal behavior is HOLD.
"""

import asyncio
from typing import ClassVar

import pytest

from agentuniverse.agent.input_object import InputObject
from agentuniverse.agent.output_object import OutputObject
from agentuniverse.agent.template.implementation_agent_template import (
    ImplementationAgentTemplate,
)
from agentuniverse.agent.template.is_agent_template import ISAgentTemplate
from agentuniverse.agent.template.supervision_agent_template import (
    SupervisionAgentTemplate,
)
from agentuniverse.agent.work_pattern.is_work_pattern import ISWorkPattern


def _output(value):
    if value is None:
        return None
    if hasattr(value, "get_data"):
        return value.get_data("output")
    return value


TRACE = []


class QualificationImplementation(ImplementationAgentTemplate):
    """Scripted implementation/correction agent with an auditable event trace."""

    events: ClassVar[list] = []
    correction_outputs: ClassVar[dict] = {}

    @classmethod
    def reset(cls, correction_outputs):
        cls.events = []
        cls.correction_outputs = dict(correction_outputs)

    @classmethod
    def _result(cls, kwargs):
        checkpoint = kwargs["checkpoint_index"]
        previous = _output(kwargs.get("implementation_result"))
        if kwargs.get("correction_mode"):
            event = (
                "correction",
                checkpoint,
                previous,
                kwargs.get("supervision_feedback"),
            )
            cls.events.append(event)
            TRACE.append(event)
            corrected = cls.correction_outputs.get(checkpoint)
            if corrected is None:
                return {}
            if isinstance(corrected, dict):
                return corrected
            return {"output": corrected, "checkpoint_output": corrected}

        current = f"impl-{checkpoint}<-{previous or 'seed'}"
        event = ("implementation", checkpoint, previous, current)
        cls.events.append(event)
        TRACE.append(event)
        return {"output": current, "checkpoint_output": current}

    def run(self, **kwargs):
        return OutputObject(self._result(kwargs))

    async def async_run(self, **kwargs):
        return OutputObject(self._result(kwargs))


class QualificationSupervisor(SupervisionAgentTemplate):
    """Scripted supervisor that records exactly which content it reviewed."""

    decisions: ClassVar[list] = []
    events: ClassVar[list] = []

    @classmethod
    def reset(cls, decisions):
        cls.decisions = list(decisions)
        cls.events = []

    @classmethod
    def _result(cls, kwargs):
        checkpoint = kwargs["checkpoint_index"]
        reviewed = _output(kwargs.get("implementation_result"))
        needs_correction = (
            cls.decisions[checkpoint]
            if checkpoint < len(cls.decisions)
            else False
        )
        event = ("supervision", checkpoint, reviewed, needs_correction)
        cls.events.append(event)
        TRACE.append(event)
        return {
            "needs_correction": needs_correction,
            "feedback": f"review:{reviewed}",
            "score": 50,
        }

    def run(self, **kwargs):
        return OutputObject(self._result(kwargs))

    async def async_run(self, **kwargs):
        return OutputObject(self._result(kwargs))


def _run(*, asynchronous, decisions, correction_outputs, max_corrections):
    TRACE.clear()
    QualificationImplementation.reset(correction_outputs)
    QualificationSupervisor.reset(decisions)
    pattern = ISWorkPattern()
    pattern.implementation = QualificationImplementation()
    pattern.supervision = QualificationSupervisor()
    input_object = InputObject({"input": "qualify IS"})
    work_pattern_input = {
        "input": "qualify IS",
        "checkpoint_count": len(decisions),
        "max_corrections": max_corrections,
    }

    if asynchronous:
        result = asyncio.run(pattern.async_invoke(input_object, work_pattern_input))
    else:
        result = pattern.invoke(input_object, work_pattern_input)
    return result, (
        list(QualificationImplementation.events),
        list(QualificationSupervisor.events),
        list(TRACE),
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_is_01_corrected_content_reaches_next_checkpoint(asynchronous):
    result, events = _run(
        asynchronous=asynchronous,
        decisions=[True, False],
        correction_outputs={0: "corrected-0"},
        max_corrections=1,
    )

    assert events[0][2] == ("implementation", 1, "corrected-0", "impl-1<-corrected-0")
    assert events[1][1] == ("supervision", 1, "impl-1<-corrected-0", False)
    assert result["execution_context"]["checkpoint_history"][0]["implementation"] == {
        "output": "corrected-0",
        "checkpoint_output": "corrected-0",
    }


@pytest.mark.parametrize("asynchronous", [False, True])
def test_is_02_requested_but_budget_blocked_is_not_recorded_as_applied(asynchronous):
    result, events = _run(
        asynchronous=asynchronous,
        decisions=[True],
        correction_outputs={0: "must-not-run"},
        max_corrections=0,
    )

    context = result["execution_context"]
    assert context["corrections_made"] == 0
    assert context["checkpoint_history"][0]["corrected"] is False
    assert "correction_result" not in result["result"][0]
    assert all(event[0] != "correction" for event in events[0])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_is_03_exhausted_budget_does_not_reapply_stale_correction(asynchronous):
    result, events = _run(
        asynchronous=asynchronous,
        decisions=[True, True, False],
        correction_outputs={0: "corrected-0", 1: "must-not-run"},
        max_corrections=1,
    )

    history = result["execution_context"]["checkpoint_history"]
    assert result["execution_context"]["corrections_made"] == 1
    assert [item["corrected"] for item in history] == [True, False, False]
    assert "correction_result" not in result["result"][1]
    assert [event[1] for event in events[0] if event[0] == "correction"] == [0]
    assert history[1]["implementation"]["output"] == "impl-1<-corrected-0"
    assert history[2]["implementation"]["output"] == "impl-2<-impl-1<-corrected-0"


def test_is_04_sync_and_async_paths_have_identical_semantics():
    arguments = {
        "decisions": [True, False, True],
        "correction_outputs": {0: "corrected-0", 2: "corrected-2"},
        "max_corrections": 2,
    }
    sync_result, sync_events = _run(asynchronous=False, **arguments)
    async_result, async_events = _run(asynchronous=True, **arguments)

    assert async_result == sync_result
    assert async_events == sync_events


@pytest.mark.parametrize("asynchronous", [False, True])
def test_is_05_history_and_trace_follow_execution_order(asynchronous):
    result, events = _run(
        asynchronous=asynchronous,
        decisions=[True, False],
        correction_outputs={0: "corrected-0"},
        max_corrections=1,
    )

    assert [(event[0], event[1]) for event in events[2]] == [
        ("implementation", 0),
        ("supervision", 0),
        ("correction", 0),
        ("implementation", 1),
        ("supervision", 1),
    ]
    assert [item["checkpoint"] for item in result["execution_context"]["checkpoint_history"]] == [0, 1]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_is_06_unavailable_correction_is_not_applied_or_leaked(asynchronous):
    result, events = _run(
        asynchronous=asynchronous,
        decisions=[True, False],
        correction_outputs={},
        max_corrections=1,
    )

    context = result["execution_context"]
    assert result["result"][0]["correction_result"] == {}
    assert context["corrections_made"] == 0
    assert context["checkpoint_history"][0]["corrected"] is False
    assert context["checkpoint_history"][0]["implementation"]["output"] == "impl-0<-seed"
    assert events[0][2] == (
        "implementation",
        1,
        "impl-0<-seed",
        "impl-1<-impl-0<-seed",
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "unavailable_correction",
    [
        {"output": "", "checkpoint_output": ""},
        {"output": " \t", "checkpoint_output": "\n "},
    ],
    ids=["production-shaped-blank", "production-shaped-whitespace"],
)
def test_is_06a_shaped_blank_correction_is_not_applied_or_leaked(
    asynchronous,
    unavailable_correction,
):
    result, events = _run(
        asynchronous=asynchronous,
        decisions=[True, False],
        correction_outputs={0: unavailable_correction},
        max_corrections=1,
    )

    context = result["execution_context"]
    assert result["result"][0]["correction_result"] == unavailable_correction
    assert context["corrections_made"] == 0
    assert [item["corrected"] for item in context["checkpoint_history"]] == [False, False]
    assert context["checkpoint_history"][0]["implementation"]["output"] == "impl-0<-seed"
    assert events[0][2] == (
        "implementation",
        1,
        "impl-0<-seed",
        "impl-1<-impl-0<-seed",
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_is_06b_blank_after_success_preserves_current_checkpoint_state(asynchronous):
    blank_correction = {"output": "", "checkpoint_output": ""}
    result, events = _run(
        asynchronous=asynchronous,
        decisions=[True, True, False],
        correction_outputs={0: "corrected-0", 1: blank_correction},
        max_corrections=2,
    )

    context = result["execution_context"]
    history = context["checkpoint_history"]
    assert context["corrections_made"] == 1
    assert [item["corrected"] for item in history] == [True, False, False]
    assert result["result"][1]["correction_result"] == blank_correction
    assert history[1]["implementation"]["output"] == "impl-1<-corrected-0"
    assert history[2]["implementation"]["output"] == "impl-2<-impl-1<-corrected-0"
    assert events[0][4] == (
        "implementation",
        2,
        "impl-1<-corrected-0",
        "impl-2<-impl-1<-corrected-0",
    )
    assert ISAgentTemplate().parse_result(result)["output"] == (
        "impl-2<-impl-1<-corrected-0"
    )


@pytest.mark.parametrize(
    "correction_result",
    [
        {},
        {"output": "", "checkpoint_output": ""},
        {"output": " ", "checkpoint_output": "\t"},
        {"metadata": "only"},
        None,
        ["not", "a", "mapping"],
    ],
)
def test_is_06c_unrepresented_correction_is_fail_closed(correction_result):
    assert ISWorkPattern._is_materially_applicable_correction(correction_result) is False


@pytest.mark.parametrize("asynchronous", [False, True])
def test_is_08_each_supervision_cycle_reviews_current_content(asynchronous):
    result, events = _run(
        asynchronous=asynchronous,
        decisions=[False, False, False],
        correction_outputs={},
        max_corrections=2,
    )

    implementations = [
        checkpoint["implementation_result"]["output"]
        for checkpoint in result["result"]
    ]
    reviewed = [event[2] for event in events[1]]
    assert reviewed == implementations
    assert len(set(reviewed)) == 3
