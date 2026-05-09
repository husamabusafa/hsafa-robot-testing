"""scheduler_skill.py — Local in-memory schedule runner for Haseef.

Runs a background polling thread that checks for due schedules
and fires a callback.  No database — everything lives in RAM.
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

log = logging.getLogger("scheduler_skill")

try:
    from croniter import croniter
    _HAS_CRONITER = True
except ImportError:  # pragma: no cover
    croniter = None  # type: ignore
    _HAS_CRONITER = False


try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover  # py < 3.9
    from pytz import timezone as ZoneInfo  # type: ignore


@dataclasses.dataclass
class Schedule:
    id: str
    description: str
    type: str                      # "one_time" | "recurring"
    scheduled_at: Optional[float]  # epoch seconds for one_time
    cron_expression: Optional[str]
    timezone: str
    active: bool = True
    last_run_at: Optional[float] = None
    next_run_at: Optional[float] = None


class SchedulerSkill:
    """In-memory scheduler.  Fire-and-forget: when a schedule is due the
    ``on_trigger`` callback is invoked from a background thread."""

    def __init__(
        self,
        on_trigger: Optional[Callable[[Schedule], None]] = None,
    ) -> None:
        self._on_trigger = on_trigger
        self._schedules: Dict[str, Schedule] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, poll_interval: float = 30.0) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(poll_interval,),
            daemon=True,
            name="scheduler",
        )
        self._thread.start()
        log.info("Scheduler started (poll every %.1fs)", poll_interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("Scheduler stopped.")

    def _run_loop(self, poll_interval: float) -> None:
        self._tick()  # fire immediately on start
        while not self._stop.is_set():
            self._stop.wait(poll_interval)
            if not self._stop.is_set():
                self._tick()

    # ------------------------------------------------------------------
    # Tick / fire
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        now = time.time()
        with self._lock:
            due = [
                s for s in self._schedules.values()
                if s.active and s.next_run_at is not None and s.next_run_at <= now
            ]

        for schedule in due:
            try:
                self._fire(schedule, now)
            except Exception:
                log.exception("Failed to fire schedule %s", schedule.id)

    def _fire(self, schedule: Schedule, now: float) -> None:
        log.info("Firing schedule '%s' (%s)", schedule.description, schedule.id[:8])

        if self._on_trigger:
            try:
                self._on_trigger(schedule)
            except Exception:
                log.exception("on_trigger callback failed for %s", schedule.id)

        with self._lock:
            schedule.last_run_at = now
            if schedule.type == "one_time":
                schedule.active = False
                schedule.next_run_at = None
            elif schedule.type == "recurring" and schedule.cron_expression:
                if _HAS_CRONITER:
                    try:
                        tz = ZoneInfo(schedule.timezone)
                        base = datetime.datetime.now(tz)
                        itr = croniter(schedule.cron_expression, base)
                        next_dt = itr.get_next(datetime.datetime)
                        schedule.next_run_at = next_dt.timestamp()
                        log.info(
                            "Next run for '%s' at %s",
                            schedule.description, next_dt.isoformat(),
                        )
                    except Exception:
                        log.error(
                            "Invalid cron '%s' for schedule %s — deactivating",
                            schedule.cron_expression, schedule.id,
                        )
                        schedule.active = False
                        schedule.next_run_at = None
                else:
                    log.error(
                        "croniter not installed; cannot reschedule '%s' — deactivating",
                        schedule.description,
                    )
                    schedule.active = False
                    schedule.next_run_at = None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_schedule(
        self,
        description: str,
        type: str = "one_time",
        scheduled_at: Optional[float] = None,
        cron_expression: Optional[str] = None,
        timezone: str = "UTC",
    ) -> str:
        """Create a new schedule and return its id."""
        sid = str(uuid.uuid4())
        now = time.time()
        next_run: Optional[float] = None

        if type == "one_time":
            next_run = scheduled_at
        elif type == "recurring":
            if not _HAS_CRONITER:
                raise RuntimeError(
                    "Recurring schedules require 'croniter'. "
                    "Install it: pip install croniter"
                )
            if not cron_expression:
                raise ValueError("Recurring schedules need a cron_expression")
            try:
                tz = ZoneInfo(timezone)
                base = datetime.datetime.now(tz)
                itr = croniter(cron_expression, base)
                next_run = itr.get_next(datetime.datetime).timestamp()
            except Exception as exc:
                raise ValueError(f"Invalid cron expression: {exc}")
        else:
            raise ValueError(f"Unknown schedule type: {type}")

        schedule = Schedule(
            id=sid,
            description=description,
            type=type,
            scheduled_at=scheduled_at,
            cron_expression=cron_expression,
            timezone=timezone,
            active=True,
            next_run_at=next_run,
        )
        with self._lock:
            self._schedules[sid] = schedule
        log.info(
            "Added schedule '%s' (%s) next_run=%s",
            description, sid[:8],
            datetime.datetime.fromtimestamp(next_run).isoformat() if next_run else None,
        )
        return sid

    def list_schedules(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "id": s.id,
                    "description": s.description,
                    "type": s.type,
                    "active": s.active,
                    "next_run_at": s.next_run_at,
                    "last_run_at": s.last_run_at,
                    "cron_expression": s.cron_expression,
                    "timezone": s.timezone,
                }
                for s in self._schedules.values()
            ]

    def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        with self._lock:
            return self._schedules.get(schedule_id)

    def delete_schedule(self, schedule_id: str) -> bool:
        with self._lock:
            if schedule_id in self._schedules:
                del self._schedules[schedule_id]
                return True
            return False

    def cancel_schedule(self, schedule_id: str) -> bool:
        with self._lock:
            s = self._schedules.get(schedule_id)
            if s is None:
                return False
            s.active = False
            return True
