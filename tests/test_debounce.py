"""DebounceTracker 与 MergeBuffer 单元测试（纯逻辑，不依赖 AstrBot）。"""

from __future__ import annotations

import pytest

from debounce import (
    DebounceTracker,
    MergeBuffer,
    format_log_prefix,
    is_real_message,
    is_command_event,
    normalize_session_list,
    session_target,
)


class FakeClock:
    """可手动推进的单调时钟。"""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_tracker(window: float, clock: FakeClock) -> DebounceTracker:
    return DebounceTracker(window=window, clock=clock)


def test_first_message_is_never_cancelled(clock: FakeClock) -> None:
    tracker = make_tracker(3.0, clock)
    ts = tracker.on_message("k")
    assert not tracker.should_cancel("k", ts)


def test_message_within_window_cancels_previous(clock: FakeClock) -> None:
    tracker = make_tracker(3.0, clock)
    ts1 = tracker.on_message("k")
    clock.advance(2.0)
    ts2 = tracker.on_message("k")
    assert tracker.should_cancel("k", ts1)
    assert not tracker.should_cancel("k", ts2)


def test_chain_of_messages_only_last_survives(clock: FakeClock) -> None:
    tracker = make_tracker(3.0, clock)
    ts1 = tracker.on_message("k")
    clock.advance(1.0)
    ts2 = tracker.on_message("k")
    clock.advance(1.0)
    ts3 = tracker.on_message("k")
    assert tracker.should_cancel("k", ts1)
    assert tracker.should_cancel("k", ts2)
    assert not tracker.should_cancel("k", ts3)


def test_message_after_window_does_not_cancel(clock: FakeClock) -> None:
    tracker = make_tracker(3.0, clock)
    ts1 = tracker.on_message("k")
    clock.advance(3.5)
    ts2 = tracker.on_message("k")
    assert not tracker.should_cancel("k", ts1)
    assert not tracker.should_cancel("k", ts2)


def test_window_boundary_is_inclusive(clock: FakeClock) -> None:
    tracker = make_tracker(3.0, clock)
    ts1 = tracker.on_message("k")
    clock.advance(3.0)
    tracker.on_message("k")
    assert tracker.should_cancel("k", ts1)


def test_zero_window_never_cancels(clock: FakeClock) -> None:
    tracker = make_tracker(0.0, clock)
    ts1 = tracker.on_message("k")
    clock.advance(0.5)
    ts2 = tracker.on_message("k")
    assert not tracker.should_cancel("k", ts1)
    assert not tracker.should_cancel("k", ts2)


def test_negative_window_raises(clock: FakeClock) -> None:
    with pytest.raises(ValueError):
        make_tracker(-1.0, clock)


def test_keys_are_isolated(clock: FakeClock) -> None:
    tracker = make_tracker(3.0, clock)
    ts_a = tracker.on_message("a")
    clock.advance(1.0)
    ts_b = tracker.on_message("b")
    clock.advance(1.0)
    tracker.on_message("a")  # a 的第二条，只应标记 a 的第一条
    assert tracker.should_cancel("a", ts_a)
    assert not tracker.should_cancel("b", ts_b)


def test_should_cancel_consumes_mark_once(clock: FakeClock) -> None:
    tracker = make_tracker(3.0, clock)
    ts1 = tracker.on_message("k")
    clock.advance(1.0)
    tracker.on_message("k")
    assert tracker.should_cancel("k", ts1)
    assert not tracker.should_cancel("k", ts1)


def test_unknown_key_or_timestamp_returns_false(clock: FakeClock) -> None:
    tracker = make_tracker(3.0, clock)
    tracker.on_message("k")
    assert not tracker.should_cancel("missing", 1.0)
    assert not tracker.should_cancel("k", 123.0)


def test_cancel_mark_survives_slow_reply_within_ttl(clock: FakeClock) -> None:
    tracker = make_tracker(3.0, clock)
    ts1 = tracker.on_message("k")
    clock.advance(1.0)
    tracker.on_message("k")
    clock.advance(60.0)  # 模拟回复迟迟未到达发送阶段
    assert tracker.should_cancel("k", ts1)


def test_stale_key_pruned_after_ttl(clock: FakeClock) -> None:
    tracker = make_tracker(3.0, clock)
    ts1 = tracker.on_message("old")
    clock.advance(1.0)
    tracker.on_message("old")
    clock.advance(700.0)  # 超过 600s 的保留时长
    tracker.on_message("other")  # 触发清理
    assert not tracker.should_cancel("old", ts1)


# ── MergeBuffer ────────────────────────────────────────────────────

def make_buffer(window: float, clock: FakeClock) -> MergeBuffer:
    return MergeBuffer(window=window, clock=clock)


def test_buffer_counts_consecutive_messages(clock: FakeClock) -> None:
    buf = make_buffer(3.0, clock)
    assert buf.add("k", "a") == 1
    clock.advance(1.0)
    assert buf.add("k", "b") == 2
    clock.advance(1.0)
    assert buf.add("k", "c") == 3
    assert buf.peek("k") == ["a", "b", "c"]


def test_buffer_resets_after_window_gap(clock: FakeClock) -> None:
    buf = make_buffer(3.0, clock)
    buf.add("k", "a")
    clock.advance(2.0)
    buf.add("k", "b")
    clock.advance(3.5)  # 超过窗口
    assert buf.add("k", "c") == 1
    assert buf.peek("k") == ["c"]


def test_buffer_has_pending_only_after_second_message(clock: FakeClock) -> None:
    buf = make_buffer(3.0, clock)
    buf.add("k", "a")
    assert not buf.has_pending("k")
    clock.advance(1.0)
    buf.add("k", "b")
    assert buf.has_pending("k")
    assert not buf.has_pending("missing")


def test_buffer_peek_does_not_clear_and_missing_empty(clock: FakeClock) -> None:
    buf = make_buffer(3.0, clock)
    buf.add("k", "a")
    assert buf.peek("k") == ["a"]
    assert buf.peek("k") == ["a"]
    assert buf.peek("missing") == []


def test_buffer_clear_if_last(clock: FakeClock) -> None:
    buf = make_buffer(3.0, clock)
    ts1 = clock.now
    buf.add("k", "a")
    clock.advance(1.0)
    ts2 = clock.now
    buf.add("k", "b")
    assert not buf.clear_if_last("k", ts1)  # ts1 不是最后一条
    assert buf.peek("k") == ["a", "b"]
    assert buf.clear_if_last("k", ts2)  # ts2 是最后一条 → 清空
    assert buf.peek("k") == []
    assert not buf.clear_if_last("k", ts2)  # 已清空
    assert not buf.clear_if_last("missing", ts2)


def test_buffer_keys_are_isolated(clock: FakeClock) -> None:
    buf = make_buffer(3.0, clock)
    buf.add("a", "a1")
    clock.advance(1.0)
    buf.add("b", "b1")
    clock.advance(1.0)
    buf.add("a", "a2")
    assert buf.peek("a") == ["a1", "a2"]
    assert buf.peek("b") == ["b1"]


def test_buffer_zero_window_always_single(clock: FakeClock) -> None:
    buf = make_buffer(0.0, clock)
    buf.add("k", "a")
    clock.advance(0.5)
    assert buf.add("k", "b") == 1


def test_buffer_negative_window_raises(clock: FakeClock) -> None:
    with pytest.raises(ValueError):
        make_buffer(-1.0, clock)


def test_buffer_prunes_stale_keys(clock: FakeClock) -> None:
    buf = make_buffer(3.0, clock)
    buf.add("old", "x")
    clock.advance(700.0)  # 超过 600s 保留时长
    buf.add("other", "y")  # 触发清理
    assert buf.peek("old") == []


def test_buffer_counts_empty_text_entries(clock: FakeClock) -> None:
    """纯媒体消息（空文本）也计入缓冲条目，保证关串比较能命中载体。"""
    buf = make_buffer(3.0, clock)
    buf.add("k", "a")
    clock.advance(1.0)
    ts_media = clock.now
    assert buf.add("k", "") == 2  # 图片/语音等纯媒体消息
    assert buf.clear_if_last("k", ts_media)  # 媒体载体也能正常关串
    assert buf.peek("k") == []


# ── 白名单辅助函数 ─────────────────────────────────────────────────

def test_normalize_session_list() -> None:
    parsed, invalid = normalize_session_list(
        ["G:10001", " f:20002 ", "G:", "10001", None, ""]
    )
    assert parsed == {"G:10001", "F:20002"}
    assert invalid == 2  # "G:" 与裸 ID "10001"
    assert normalize_session_list("not-a-list") == (set(), 0)
    assert normalize_session_list(None) == (set(), 0)
    assert normalize_session_list(()) == (set(), 0)


def test_session_target_uses_g_f_prefix() -> None:
    assert session_target(True, "group_demo", "10001") == "F:10001"
    assert session_target(False, "group_demo", "10001") == "G:group_demo"


class CommandFilter:
    pass


class _OtherFilter:
    pass


class _Handler:
    def __init__(self, *filters) -> None:
        self.event_filters = list(filters)


def test_is_command_event() -> None:
    assert is_command_event(None) is False
    assert is_command_event([]) is False
    assert is_command_event([_Handler(_OtherFilter())]) is False
    # 类名兜底（本地无法导入真实 CommandFilter）
    assert is_command_event([_Handler(_OtherFilter(), CommandFilter())]) is True
    # 显式传入类型时按 isinstance 判定
    assert is_command_event([_Handler(CommandFilter())], CommandFilter) is True
    assert is_command_event([_Handler(_OtherFilter())], CommandFilter) is False


def test_format_log_prefix() -> None:
    assert format_log_prefix("message_debounce") == "[message_debounce]"
    assert format_log_prefix("message_debounce", "BOT1", False) == "[message_debounce]"
    assert (
        format_log_prefix("message_debounce", "BOT1", True)
        == "[message_debounce][platform:BOT1]"
    )
    assert format_log_prefix("message_debounce", "", True) == "[message_debounce]"


def test_is_real_message() -> None:
    assert is_real_message("message", True) is True
    assert is_real_message("message", False) is False
    # NapCat 的 input_status「正在输入」：post_type=notice 且消息链为空
    assert is_real_message("notice", False) is False
    assert is_real_message("notice", True) is False
    assert is_real_message("request", False) is False
    # 非 OneBot 平台（raw 无 post_type）只看消息链
    assert is_real_message(None, True) is True
    assert is_real_message(None, False) is False
