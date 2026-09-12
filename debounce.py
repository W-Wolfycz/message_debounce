"""按用户维度的时间防抖与 prompt 合并核心逻辑。

纯 Python 实现，不依赖 AstrBot，便于单元测试。

DebounceTracker 语义：
同一用户键下，若一条消息在防抖窗口内到达，则其紧邻的上一条消息的
待发送回复被标记取消；每条新消息只与上一条比较，因此窗口内连续
多条消息只有最新一条会保留回复。

MergeBuffer 语义：
按用户键缓冲窗口内连续到达的消息文本；间隔超过窗口则重新开始。
最终载体请求在 on_llm_request 阶段取出缓冲文本合并进 ProviderRequest.prompt，
事件原文（message_str）不做任何改写。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

Clock = Callable[[], float]

# 已取消时间戳/缓冲键的保留时长（秒）：超过该时长仍未到达发送阶段的回复视为失效，
# 清理以控内存。
_ENTRY_TTL_SECONDS = 600.0
# 单个用户键的已取消时间戳上限，超出丢弃最旧项（防窗口内洪水堆积）。
_MAX_CANCELLED_PER_KEY = 1000
# 用户键总数上限，超出时按最近活跃时间清理最旧键。
_MAX_KEYS = 10000


@dataclass
class _KeyState:
    last: float
    cancelled: set[float] = field(default_factory=set)


@dataclass
class _BufferState:
    last: float
    texts: list[str]


class DebounceTracker:
    """按用户维度的时间防抖跟踪器。"""

    def __init__(self, window: float, clock: Clock = time.monotonic) -> None:
        if window < 0:
            raise ValueError(f"window 不能为负数: {window}")
        self.window = window
        self._clock = clock
        self._states: dict[str, _KeyState] = {}

    def on_message(self, key: str) -> float:
        """记录一条新消息到达并返回其时间戳。

        若与上一条消息的时间差落在 (0, window] 内，则标记上一条消息待取消。
        """
        now = self._clock()
        self._prune(now)
        state = self._states.get(key)
        if state is None:
            self._states[key] = _KeyState(last=now)
            return now
        if 0 < now - state.last <= self.window:
            state.cancelled.add(state.last)
            if len(state.cancelled) > _MAX_CANCELLED_PER_KEY:
                state.cancelled.discard(min(state.cancelled))
        state.last = now
        return now

    def should_cancel(self, key: str, timestamp: float) -> bool:
        """该时间戳对应的回复是否应被取消。

        命中一次即消耗标记，重复调用返回 False，保证幂等。
        """
        state = self._states.get(key)
        if state is None or timestamp not in state.cancelled:
            return False
        state.cancelled.discard(timestamp)
        return True

    def _prune(self, now: float) -> None:
        """清理已过期条目与长期不活跃的用户键，控制内存占用。"""
        horizon = now - _ENTRY_TTL_SECONDS
        for key, state in list(self._states.items()):
            stale = {t for t in state.cancelled if t < horizon}
            state.cancelled.difference_update(stale)
            if not state.cancelled and state.last < horizon:
                del self._states[key]
        if len(self._states) > _MAX_KEYS:
            excess = len(self._states) - _MAX_KEYS
            oldest = sorted(self._states, key=lambda k: self._states[k].last)[:excess]
            for key in oldest:
                del self._states[key]


class MergeBuffer:
    """按用户键缓冲窗口内连续消息文本，供最终 prompt 合并。"""

    def __init__(self, window: float, clock: Clock = time.monotonic) -> None:
        if window < 0:
            raise ValueError(f"window 不能为负数: {window}")
        self.window = window
        self._clock = clock
        self._states: dict[str, _BufferState] = {}

    def add(self, key: str, text: str) -> int:
        """记录一条消息文本；与上一条间隔超过窗口时清空旧缓冲重新开始。

        返回当前缓冲条数。
        """
        now = self._clock()
        self._prune(now)
        state = self._states.get(key)
        if state is None or now - state.last > self.window:
            self._states[key] = _BufferState(last=now, texts=[text])
        else:
            state.last = now
            state.texts.append(text)
        return len(self._states[key].texts)

    def has_pending(self, key: str) -> bool:
        """该键是否已有可合并的多条消息。"""
        state = self._states.get(key)
        return bool(state) and len(state.texts) > 1

    def peek(self, key: str) -> list[str]:
        """查看该键缓冲的文本列表（不清理）。"""
        state = self._states.get(key)
        return list(state.texts) if state else []

    def clear_if_last(self, key: str, timestamp: float) -> bool:
        """缓冲最后一条仍是 timestamp 对应的消息（其后无新消息）时清空。

        返回是否清空。若期间已有更新的消息到达，保留缓冲给下一位载体合并。
        """
        state = self._states.get(key)
        if state is None or state.last != timestamp:
            return False
        del self._states[key]
        return True

    def _prune(self, now: float) -> None:
        horizon = now - _ENTRY_TTL_SECONDS
        for key in [k for k, s in self._states.items() if s.last < horizon]:
            del self._states[key]


def normalize_session_list(raw: object) -> tuple[set[str], int]:
    """解析会话名单为归一化集合，条目格式 `G:<群号>` 或 `F:<用户号>`。

    群号与用户号可能同号，必须带前缀区分（与 bubble_reply 的名单一致）。
    返回 (合法项集合, 被忽略的非法项数量)。
    """
    if not isinstance(raw, (list, tuple, set)):
        return set(), 0
    result: set[str] = set()
    invalid = 0
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        prefix, separator, identifier = text.partition(":")
        prefix = prefix.strip().upper()
        identifier = identifier.strip()
        if separator and identifier and prefix in ("G", "F"):
            result.add(f"{prefix}:{identifier}")
        else:
            invalid += 1
    return result, invalid


def session_target(is_private: bool, group_id: str, sender_id: str) -> str:
    """名单匹配目标：私聊 `F:<用户号>`，群聊 `G:<群号>`。"""
    return f"F:{sender_id}" if is_private else f"G:{group_id}"


def format_log_prefix(
    plugin_name: str, platform_id: str = "", with_bot_id: bool = False
) -> str:
    """日志前缀：`log_with_bot_id` 启用且拿到平台实例时附带 platform 标识。"""
    if with_bot_id and platform_id:
        return f"[{plugin_name}][platform:{platform_id}]"
    return f"[{plugin_name}]"


def is_real_message(post_type: object, has_components: bool) -> bool:
    """是否是需要参与防抖的真实消息事件。

    OneBot 系适配器会把通知 / 请求类事件（如 NapCat 的 `input_status`
    「正在输入」）也转成 AstrMessageEvent：`post_type` 不是 `message`、
    消息链为空。这类事件不是用户发言，不能参与防抖。
    """
    if post_type is not None and post_type != "message":
        return False
    return bool(has_components)


def is_command_event(
    activated_handlers: object, command_filter_type: type | None = None
) -> bool:
    """该事件是否会被命令 Handler 处理。

    命令由指令 Handler 直接回复、默认不会进入 LLM，因此不参与防抖。
    `command_filter_type` 为 None 时按过滤器类名兜底判定。
    """
    if not isinstance(activated_handlers, (list, tuple)):
        return False
    for handler in activated_handlers:
        for filter_obj in getattr(handler, "event_filters", ()) or ():
            if command_filter_type is not None and isinstance(
                filter_obj, command_filter_type
            ):
                return True
            if filter_obj.__class__.__name__ == "CommandFilter":
                return True
    return False
