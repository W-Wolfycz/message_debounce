"""消息防抖插件：窗口时间内同一用户的新消息会取消上一条消息的待发送回复，
并把窗口内的连续消息合并进最终请求的 prompt（不改写事件原文）。

消息流转中的拦截点：
1. ProcessStage（event_message_type ALL）：记录每条消息的到达时间戳并缓冲
   其文本（只记录，不拦截、不改写事件 → chat_memory 等插件仍按原始
   message_str 逐条归档）；
2. on_llm_request（最早）：已被更新的消息取代时 stop_event 中止请求；若自己是
   窗口内最后一条且带着被取消的前文，则等待窗口静默后把缓冲文本合并进
   ProviderRequest.prompt，并打 event extra `message_debounce_merge` 供插件链
   感知，同时通过 extra_user_content_parts 告知 LLM；
3. on_llm_request（最晚）：兜底——链上耗时插件（如 CM/LM 查询）期间新消息到达
   时，在请求真正发出前再查一次取消标记，命中则中止，省掉一次 LLM 调用；
4. on_llm_response：生成期间被新消息取代时 stop_event 中止（非流式路径核心
   会跳过发送与历史保存）；未命中则回复注定会发出，此刻关闭该串缓冲。

已发送或已开始流式输出的回复不追回。
"""

from __future__ import annotations

import asyncio
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

from .debounce import DebounceTracker, MergeBuffer

PLUGIN_NAME = "message_debounce"
_TS_MARKER = f"_{PLUGIN_NAME}_arrive_ts"
_MERGE_MARKER = f"{PLUGIN_NAME}_merge"
_WAITED_MARKER = f"_{PLUGIN_NAME}_waited"

# 到达记录需要尽早打时间戳，避免被其他插件 Handler 的耗时拉偏；
# 拦截也需要尽早判定中止，避免其他插件对即将丢弃的回复做无用工。
_PRIORITY = 100_000
# 请求链最末尾的兜底取消：低于本环境所有已知 on_llm_request 优先级
# （self_utils 用到 -1000000），在真正发出请求前再查一次取消标记。
_FALLBACK_PRIORITY = -10_000_000
# 唤醒事件字典上限：超出整体清空（事件对象廉价，等待方靠取消标记兜底）。
_MAX_WAKE_EVENTS = 10000


class MsgDebouncePlugin(Star):
    """消息防抖：按用户维度做时间防抖，窗口内消息合并进最终 prompt。"""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        try:
            window = float(config.get("window", 0.0))
        except (TypeError, ValueError):
            window = 0.0
        window = max(0.0, window)
        self.window = window
        self.enabled = window > 0
        self.tracker = DebounceTracker(window)
        self.buffer = MergeBuffer(window)
        self._wake_events: dict[str, asyncio.Event] = {}
        logger.info(
            f"[{PLUGIN_NAME}] 已加载：window={self.window:g}s"
            f"（{'启用' if self.enabled else '关闭'}）"
        )

    @staticmethod
    def _user_key(event: AstrMessageEvent) -> str:
        """用户维度键：平台实例 + 会话 + 发送者。"""
        platform = event.get_platform_id() or event.get_platform_name()
        return f"{platform}:{event.get_session_id()}:{event.get_sender_id()}"

    def _wake_event(self, key: str) -> asyncio.Event:
        evt = self._wake_events.get(key)
        if evt is None:
            if len(self._wake_events) >= _MAX_WAKE_EVENTS:
                self._wake_events.clear()
            evt = self._wake_events[key] = asyncio.Event()
        return evt

    # ── 拦截点 1：消息到达 ─────────────────────────────────────────

    @filter.event_message_type(filter.EventMessageType.ALL, priority=_PRIORITY)
    async def record_arrival(self, event: AstrMessageEvent) -> None:
        """记录到达时间（窗口内到达标记上一条待取消），并缓冲消息文本。

        bot 自身消息（平台回显）不参与防抖：不打标、不缓冲、不唤醒。
        """
        if not self.enabled:
            return
        if event.get_sender_id() == event.get_self_id():
            return
        key = self._user_key(event)
        event.set_extra(_TS_MARKER, self.tracker.on_message(key))
        text = (event.get_message_str() or "").strip()
        self.buffer.add(key, text)
        self._wake_event(key).set()

    # ── 拦截点 2：LLM 请求发起前 ───────────────────────────────────

    @filter.on_llm_request(priority=_PRIORITY)
    async def handle_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """取消被取代的请求；窗口内最后一条请求合并前文并等待窗口静默。"""
        if not self.enabled:
            return
        timestamp = event.get_extra(_TS_MARKER)
        if timestamp is None:
            return
        key = self._user_key(event)
        if self.tracker.should_cancel(key, timestamp):
            logger.debug(f"[{PLUGIN_NAME}] 中止过期 LLM 请求 user_key={key}")
            event.stop_event()
            return
        if event.get_extra(_WAITED_MARKER):
            return
        if not self.buffer.has_pending(key):
            return
        event.set_extra(_WAITED_MARKER, True)
        await self._wait_window_quiet(key, timestamp)
        if self.tracker.should_cancel(key, timestamp):
            logger.debug(f"[{PLUGIN_NAME}] 载体被更新消息取代 user_key={key}")
            event.stop_event()
            return
        raw = self.buffer.peek(key)
        texts = [t for t in raw if t]
        # 纯媒体消息（空文本）也算缓冲条目；无可合并文本时跳过。
        if not texts or (len(texts) == 1 and raw[-1] == texts[0]):
            return
        texts[-1] = req.prompt or texts[-1]
        req.prompt = "\n".join(texts)
        event.set_extra(
            _MERGE_MARKER, {"count": len(raw), "window": self.window}
        )
        parts = getattr(req, "extra_user_content_parts", None)
        if parts is None:
            parts = []
            req.extra_user_content_parts = parts
        parts.append(
            {
                "type": "text",
                "text": (
                    f"（系统说明：本条消息由用户连续发送的 "
                    f"{len(raw)} 条消息合并而成，请整体理解并回应。）"
                ),
                "_no_save": True,
            }
        )
        logger.debug(f"[{PLUGIN_NAME}] 合并 {len(texts)} 条文本 user_key={key}")

    async def _wait_window_quiet(self, key: str, timestamp: float) -> None:
        """等待剩余窗口静默；期间新消息到达会立即唤醒（超时/唤醒后按取消标记判定）。"""
        remaining = self.window - (time.monotonic() - timestamp)
        if remaining <= 0:
            return
        evt = self._wake_event(key)
        evt.clear()
        try:
            await asyncio.wait_for(evt.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            pass

    # ── 拦截点 3：请求链末尾兜底取消（真正发请求前的最后一道） ────────

    @filter.on_llm_request(priority=_FALLBACK_PRIORITY)
    async def fallback_cancel(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """请求链最末尾再查一次取消标记。

        链上耗时插件（CM/LM 查询等）运行期间新消息到达时，早检查已过，
        此处中止可在请求真正发出前省掉一次 LLM 调用。
        """
        if not self.enabled:
            return
        timestamp = event.get_extra(_TS_MARKER)
        if timestamp is None:
            return
        key = self._user_key(event)
        if self.tracker.should_cancel(key, timestamp):
            logger.debug(f"[{PLUGIN_NAME}] 兜底中止过期请求 user_key={key}")
            event.stop_event()

    # ── 拦截点 4：LLM 响应刚返回 ───────────────────────────────────

    @filter.on_llm_response(priority=_PRIORITY)
    async def stop_stale_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        """LLM 响应刚返回时检查：已被更新的消息取代则中止，不发送、不写历史。

        未命中时回复注定会发出，此刻关闭该串缓冲，堵住「response → 发送」
        期间新消息重复合并已提交文本的竞态。
        """
        if not self.enabled:
            return
        timestamp = event.get_extra(_TS_MARKER)
        if timestamp is None:
            return
        key = self._user_key(event)
        if self.tracker.should_cancel(key, timestamp):
            logger.debug(f"[{PLUGIN_NAME}] 中止过期回复 user_key={key}")
            event.stop_event()
            return
        if not event.get_extra("agent_user_aborted"):
            self.buffer.clear_if_last(key, timestamp)
