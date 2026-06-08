import asyncio
from aiohttp import web
from utils import Searcher, InfoFinder, ICON_SET_ROOT
from pathlib import Path
PORT = 8080

async def handle_root(request: web.Request) -> web.Response:
    return web.FileResponse("index.html")


def get_image(request: web.Request):
    filename = request.match_info["filename"]
    root_dir = Path("../")
    file_path = root_dir / filename
    if file_path.is_file():
        return web.FileResponse(file_path)
    else:
        return web.Response(status=404, text="File not found")
    
  
async def server():
    searcher = Searcher(ICON_SET_ROOT / "embedding")
    searcher.remove_icon_set("SYBX")
    searcher.remove_icon_set("MTC")
    searcher.remove_icon_set("PCS")
    searcher.remove_icon_set("WDGT")
    infoF = InfoFinder()
    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_get("/file/{filename:.+}", get_image)
    app.router.add_get("/search/{image}", searcher.search_cell_request)
    app.router.add_get("/search_text", searcher.search_text_request)
    app.router.add_get("/next", infoF.next_button_request)
    app.router.add_post("/submit", infoF.submit_info)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "localhost", PORT)
    await site.start()

    print(f"Open http://localhost:{PORT} in your browser")

    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(server())