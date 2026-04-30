"""載入 odds/odds_snapshots/*.json 並組成 per-TW-date timeline。

設計重點：
- game_key 含 commence_utc（ISO 字串）→ doubleheader-safe
- 隊名一致性檢查：ml 兩 key 必須等於 {away, home}，否則跳該場
- pinnacle 缺漏 → 跳該場
- snapshot_time_utc >= commence_utc → 跳（避免 live 賽中賠率污染）
- timeline 內按 snapshot_time_utc 由舊到新排序
- 對外時區為 TW；ET 欄位保留供 anchor 與 debug 使用
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


GameKey = tuple[str, str, str]   # (away_team, home_team, commence_utc_iso)


# MLB 球季固定 EDT = UTC-4；TW = UTC+8
ET = timezone(timedelta(hours=-4))
TW = timezone(timedelta(hours=+8))


@dataclass
class Snapshot:
    snapshot_time_et: datetime
    snapshot_time_utc: datetime
    snapshot_time_tw: datetime          # naive TW wall clock
    games: list[dict]


@dataclass
class GameRecord:
    game_key: GameKey
    away: str
    home: str
    commence_utc: datetime
    commence_et_label: str
    pinnacle: dict
    snapshot_time_et: datetime
    snapshot_time_et_label: str         # 例 "00:00" / "04:00"
    # ── TW 欄位 ──
    commence_tw_label: str              # 例 "2026-04-30 01:11 TW"
    snapshot_time_tw: datetime          # naive TW wall clock
    snapshot_time_tw_label: str         # 例 "04-29 09:24"（含 MM-DD 前綴）
    game_date_tw: str                   # 例 "2026-04-30"


def _parse_et_label(label: str) -> datetime:
    """解析 'YYYY-MM-DD HH:MM ET' → naive datetime（時區資訊靠檔名與 utc 欄位攜帶）。"""
    return datetime.strptime(label.replace(" ET", "").strip(), "%Y-%m-%d %H:%M")


def load_snapshots_for_et_date(et_date: str, snapshot_dir) -> list[Snapshot]:
    """讀 snapshot_dir 下所有 *-ET.json 快照，按 snapshot_time_utc 排序。

    註：函式名保留 et_date 參數，但本實作不再依檔名前綴過濾日期——避免「跨日 snapshot
    在後一日分析時被忽略」的 silent data loss。日期過濾完全交給下游 `collect_game_timeline`
    的 `commence_utc → game_date_tw` 比對。
    """
    p = Path(snapshot_dir)
    if not p.exists():
        return []
    out: list[Snapshot] = []
    for f in p.glob("*-ET.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"[snapshot_loader] WARN 讀取 {f.name} 失敗：{e}", file=sys.stderr)
            continue
        try:
            snap_et = _parse_et_label(data["snapshot_time_et"])
            snap_utc = datetime.fromisoformat(data["snapshot_time_utc"])
            snap_tw = snap_utc.astimezone(TW).replace(tzinfo=None)
            out.append(Snapshot(
                snapshot_time_et=snap_et,
                snapshot_time_utc=snap_utc,
                snapshot_time_tw=snap_tw,
                games=data.get("games", []),
            ))
        except (KeyError, ValueError) as e:
            print(f"[snapshot_loader] WARN {f.name} 欄位異常：{e}", file=sys.stderr)
            continue
    out.sort(key=lambda s: s.snapshot_time_utc)
    return out


def collect_game_timeline(
    snapshots: list[Snapshot],
    game_date_tw: str,
) -> dict[GameKey, list[GameRecord]]:
    """以 (away, home, commence_utc_iso) 分組，回每場按 snapshot 時間排序的 record list。

    過濾條件：
    - commence_utc 缺/異常 → 跳
    - bookmakers.pinnacle 缺 → 跳
    - pinnacle.ml.keys 與 {away, home} 不一致 → 跳並 log warning
    - commence_utc.astimezone(TW).strftime("%Y-%m-%d") != game_date_tw → 跳
    - snapshot_time_utc >= commence_utc → 跳（避免 live 賽中賠率污染）
    """
    timelines: dict[GameKey, list[GameRecord]] = {}

    for snap in snapshots:
        for g in snap.games:
            pinnacle = g.get("bookmakers", {}).get("pinnacle")
            if not pinnacle:
                continue
            away = g.get("away_team")
            home = g.get("home_team")
            if not away or not home:
                continue
            ml_keys = set(pinnacle.get("ml", {}).keys())
            if ml_keys != {away, home}:
                print(
                    f"[snapshot_loader] WARN team name mismatch — game={g.get('game')} "
                    f"ml.keys={ml_keys} expected={{{away},{home}}}; 跳過",
                    file=sys.stderr,
                )
                continue
            commence_utc_s = g.get("commence_utc")
            if not commence_utc_s:
                continue
            try:
                commence_utc = datetime.fromisoformat(commence_utc_s.replace("Z", "+00:00"))
            except ValueError:
                continue
            # TW 日期過濾（取代舊有 g["game_date_et"] 比對）
            commence_tw = commence_utc.astimezone(TW)
            game_date_tw_computed = commence_tw.strftime("%Y-%m-%d")
            if game_date_tw_computed != game_date_tw:
                continue
            # 過濾賽中 snapshot：snapshot_time_utc >= commence_utc 視為已開球
            if snap.snapshot_time_utc.tzinfo is None:
                snap_utc_aware = snap.snapshot_time_utc.replace(tzinfo=commence_utc.tzinfo)
            else:
                snap_utc_aware = snap.snapshot_time_utc
            if snap_utc_aware >= commence_utc:
                continue
            commence_tw_label = commence_tw.strftime("%Y-%m-%d %H:%M TW")
            game_key: GameKey = (away, home, commence_utc_s)
            record = GameRecord(
                game_key=game_key,
                away=away,
                home=home,
                commence_utc=commence_utc,
                commence_et_label=g.get("commence_et", ""),
                pinnacle=pinnacle,
                snapshot_time_et=snap.snapshot_time_et,
                snapshot_time_et_label=snap.snapshot_time_et.strftime("%H:%M"),
                commence_tw_label=commence_tw_label,
                snapshot_time_tw=snap.snapshot_time_tw,
                snapshot_time_tw_label=snap.snapshot_time_tw.strftime("%m-%d %H:%M"),
                game_date_tw=game_date_tw_computed,
            )
            timelines.setdefault(game_key, []).append(record)

    for k in timelines:
        timelines[k].sort(key=lambda r: r.snapshot_time_tw)

    return timelines


def select_anchor(timeline: list[GameRecord]) -> GameRecord:
    """回 timeline[0]。timeline 不可為空。"""
    if not timeline:
        raise ValueError("select_anchor: timeline must not be empty")
    return timeline[0]
