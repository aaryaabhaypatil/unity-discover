"""
Manual AAC Grid Explorer Bot
=============================
Web-based GUI version. Run with:
    python3 bot.py
Then open http://localhost:8080 in your browser.
"""

import asyncio
import hashlib
import json
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aiohttp import web
from extract_grid import GridIcon, parse_grid
from ipad_screenshot import screenshot

STATE_FILE = Path("state.json")
SCREENSHOTS_DIR = Path("screenshots")
SCREENSHOTS_DIR.mkdir(exist_ok=True)
PORT = 8080

# ---------------------------------------------------------------------------
# Shared server state
# ---------------------------------------------------------------------------

_bot_status: str = "starting"
_display_data: dict | None = None
_response_future: asyncio.Future | None = None


def _set_status(status: str, data: dict | None = None) -> None:
    global _bot_status, _display_data
    _bot_status = status
    _display_data = data


async def _wait_for_response() -> dict:
    global _response_future
    _response_future = asyncio.get_event_loop().create_future()
    result = await _response_future
    _response_future = None
    return result


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class IconRecord:
    row: int
    col: int
    label: str | None
    outcome: str | None = None
    text_added: str | None = None
    action_description: str | None = None
    leads_to_page_id: str | None = None


@dataclass
class PageRecord:
    page_id: str
    screenshot_path: str
    nav_path: str = "HOME"
    icons: list[IconRecord] = field(default_factory=list)
    fully_explored: bool = False


@dataclass
class GraphState:
    pages: dict[str, PageRecord] = field(default_factory=dict)
    queue: list = field(default_factory=list)
    root_page_id: str | None = None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_state(state: GraphState) -> None:
    def serialise_icon(icon):
        return {
            "row": icon.row,
            "col": icon.col,
            "label": icon.label,
            "outcome": icon.outcome,
            "text_added": icon.text_added,
            "action_description": icon.action_description,
            "leads_to_page_id": icon.leads_to_page_id,
        }
    
    def serialise(obj):
        if isinstance(obj, PageRecord):
            return {
                "page_id": obj.page_id,
                "screenshot_path": obj.screenshot_path,
                "nav_path": obj.nav_path,
                "icons": [serialise_icon(i) for i in obj.icons],
                "fully_explored": obj.fully_explored,
            }
        raise TypeError(f"Not serialisable: {type(obj)}")

    data = {
        "pages": {k: serialise(v) for k, v in state.pages.items()},
        "queue": state.queue,
        "root_page_id": state.root_page_id,
    }
    try:
        STATE_FILE.write_text(json.dumps(data, indent=2))
        text_count = sum(1 for p in state.pages.values() for ic in p.icons if ic.text_added)
        print(f"✓ Saved: {len(state.pages)} pages, {text_count} text entries, queue={len(state.queue)}")
    except Exception as e:
        print(f"ERROR saving state: {e}")



def load_state() -> GraphState | None:
    if not STATE_FILE.exists():
        return None
    data = json.loads(STATE_FILE.read_text())
    pages = {}
    for pid, pd in data["pages"].items():
        icons = [IconRecord(**i) for i in pd["icons"]]
        pages[pid] = PageRecord(
            page_id=pd["page_id"],
            screenshot_path=pd["screenshot_path"],
            nav_path=pd.get("nav_path", "HOME"),
            icons=icons,
            fully_explored=pd["fully_explored"],
        )
    state = GraphState(
        pages=pages,
        queue=data["queue"],
        root_page_id=data["root_page_id"],
    )
    _rebuild_nav_paths(state)
    return state


def _rebuild_nav_paths(state: GraphState) -> None:
    if not state.root_page_id or state.root_page_id not in state.pages:
        return
    visited = {state.root_page_id}
    q: deque = deque([state.root_page_id])
    while q:
        pid = q.popleft()
        page = state.pages[pid]
        for ic in page.icons:
            if ic.outcome == "grid" and ic.leads_to_page_id:
                child_id = ic.leads_to_page_id
                if child_id in state.pages and child_id not in visited:
                    child = state.pages[child_id]
                    if child_id != state.root_page_id and child.nav_path == "HOME":
                        child.nav_path = child_nav_path(page.nav_path, ic.row, ic.col, ic.label)
                    visited.add(child_id)
                    q.append(child_id)


# ---------------------------------------------------------------------------
# Fingerprinting & path helpers
# ---------------------------------------------------------------------------

def fingerprint(icons: list[GridIcon]) -> str:
    tokens = sorted(f"{ic.label or ''}|{ic.background}" for ic in icons)
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()[:16]


def icon_name(row: int, col: int, label: str | None) -> str:
    return label if label else f"[{row + 1},{col + 1}]"


def child_nav_path(parent_path: str, row: int, col: int, label: str | None) -> str:
    return f"{parent_path} > {icon_name(row, col, label)}"


def _find_nav_path(state: GraphState, page_id: str) -> list[tuple[str, int, int, str | None]]:
    if not state.root_page_id or state.root_page_id not in state.pages:
        return []

    if page_id == state.root_page_id:
        return []

    visited = {state.root_page_id}
    parent_map: dict[str, tuple[str, int, int, str | None]] = {}
    q: deque = deque([state.root_page_id])

    while q:
        pid = q.popleft()
        page = state.pages[pid]
        for ic in page.icons:
            if ic.outcome != "grid" or not ic.leads_to_page_id:
                continue
            child_id = ic.leads_to_page_id
            if child_id in visited:
                continue
            visited.add(child_id)
            parent_map[child_id] = (pid, ic.row, ic.col, ic.label)
            if child_id == page_id:
                q.clear()
                break
            q.append(child_id)

    if page_id not in parent_map:
        return []

    chain: list[tuple[str, int, int, str | None]] = []
    pid = page_id
    while pid != state.root_page_id:
        parent_id, row, col, label = parent_map[pid]
        chain.append((parent_id, row, col, label))
        pid = parent_id

    chain.reverse()
    return chain


def build_display_data(state: GraphState, page_id: str, target_row: int, target_col: int, target_label: str | None) -> dict:
    """Build the data dict the HTML render() function expects."""
    chain = _find_nav_path(state, page_id)

    path = [
        {"pageImage": state.pages[pid].screenshot_path, "nextCell": [row, col], "nextLabel": label}
        for pid, row, col, label in chain
    ]

    current_page = state.pages[page_id]
    prev_actions = sorted({
        ic.action_description
        for page in state.pages.values()
        for ic in page.icons
        if ic.outcome == "action" and ic.action_description
    })

    return {
        "path": path,
        "pageImage": current_page.screenshot_path,
        "nextCell": [target_row, target_col],
        "nextLabel": target_label,
        "previousActions": prev_actions,
    }


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

async def capture_current_page(state: GraphState) -> tuple[str, PageRecord] | tuple[None, None]:
    _set_status("capturing")
    img = await screenshot()
    if img is None:
        return None, None
    crop = (2, 196, 2356, 1608)
    icons = await parse_grid(img, crop)

    if not icons:
        return None, None

    fprint = fingerprint(icons)
    shot_path = str(SCREENSHOTS_DIR / f"page_{fprint}.png")
    if not os.path.exists(shot_path):
        img.crop(crop).save(shot_path)

    if fprint in state.pages:
        return fprint, state.pages[fprint]

    icon_records = [IconRecord(row=ic.row, col=ic.col, label=ic.label) for ic in icons]
    record = PageRecord(page_id=fprint, screenshot_path=shot_path, icons=icon_records)
    state.pages[fprint] = record
    return fprint, record


# ---------------------------------------------------------------------------
# Web server handlers
# ---------------------------------------------------------------------------

async def handle_root(request: web.Request) -> web.Response:
    return web.FileResponse("index.html")


async def handle_screenshots(request: web.Request) -> web.Response:
    filename = request.match_info["filename"]
    path = SCREENSHOTS_DIR / filename
    if not path.exists() or not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


async def handle_state(request: web.Request) -> web.Response:
    print("State requested")
    return web.json_response({"status": _bot_status, "data": _display_data})


async def handle_respond(request: web.Request) -> web.Response:
    global _response_future
    if _response_future is None or _response_future.done():
        print('No pending action for respond')
        return web.json_response({"error": "no pending action"}, status=400)
    body = await request.json()
    print('Respond:', body)
    _response_future.set_result(body)
    return web.json_response({"ok": True})


async def handle_reset(request: web.Request) -> web.Response:
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------------------
# BFS traversal
# ---------------------------------------------------------------------------

async def run_bot() -> None:
    existing = load_state()
    print("here")

    if existing:
        _set_status("resuming")
        
    else:
        _set_status("starting")
    print("here 2")

    # Wait for user to confirm via the UI
    response = await _wait_for_response()
    fresh = response.get("fresh", False)
    print("here 3")

    state = None if fresh else existing
    if state is None:
        print("New state")
        state = GraphState()
        page_id, record = await capture_current_page(state)
        if page_id is None:
            _set_status("error", {"message": "Could not capture root page — is the app showing a grid?"})
            return
        state.root_page_id = page_id
        record.nav_path = "HOME"
        for ic in record.icons:
            state.queue.append((page_id, ic.row, ic.col))
        save_state(state)

    queue = deque(state.queue)
    print("here 4")

    while queue:
        print(len(queue), "tasks remaining")
        task = queue.popleft()
        state.queue = list(queue)

        if task is None:
            continue

        page_id, target_row, target_col = task

        if page_id not in state.pages:
            save_state(state)
            continue

        page = state.pages[page_id]
        icon_rec = next(
            (ic for ic in page.icons if ic.row == target_row and ic.col == target_col),
            None,
        )
        if icon_rec is None or icon_rec.outcome is not None:
            save_state(state)
            continue
        print(f"Exploring page {page_id} icon at row {target_row} col {target_col}")

        display = build_display_data(state, page_id, target_row, target_col, icon_rec.label)
        print(f"Built display data for page {page_id} icon ({target_row}, {target_col})")
        _set_status("pending", display)
        print(f"Set status to pending for page {page_id} icon ({target_row}, {target_col})")

        response = await _wait_for_response()
        print(f"Received response for page {page_id} icon ({target_row}, {target_col})")
        outcome = response.get("outcome")
        print(f"Received response with outcome: {outcome}")
        if outcome == "grid":
            new_page_id, new_record = await capture_current_page(state)
            if new_page_id is None:
                icon_rec.outcome = "action"
                icon_rec.action_description = "led somewhere but capture failed"
            else:
                if new_record.nav_path == "HOME" and new_page_id != state.root_page_id:
                    new_record.nav_path = child_nav_path(page.nav_path, target_row, target_col, icon_rec.label)
                icon_rec.outcome = "grid"
                icon_rec.leads_to_page_id = new_page_id
                for ic in new_record.icons:
                    if ic.outcome is None:
                        entry = (new_page_id, ic.row, ic.col)
                        if entry not in queue and entry not in state.queue:
                            queue.append(entry)

        elif outcome == "text":
            icon_rec.outcome = "text"
            icon_rec.text_added = response.get("text_added", "").strip()
            print(f"TEXT: Set icon ({target_row}, {target_col}) text='{icon_rec.text_added}'")


        elif outcome == "action":
            icon_rec.outcome = "action"
            icon_rec.action_description = response.get("action_description", "")

        else:
            icon_rec.outcome = "nothing"

        state.queue = list(queue)
        save_state(state)

    for page in state.pages.values():
        if all(ic.outcome is not None for ic in page.icons):
            page.fully_explored = True
    save_state(state)
    _set_status("done")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_get("/screenshots/{filename}", handle_screenshots)
    app.router.add_get("/state", handle_state)
    app.router.add_post("/respond", handle_respond)
    app.router.add_post("/reset", handle_reset)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", PORT)
    await site.start()
    print(f"Open http://localhost:{PORT} in your browser")

    await run_bot()

    # Keep the server alive after traversal finishes (status will be "done")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())

