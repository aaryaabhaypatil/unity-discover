from pathlib import Path
from aiohttp import web
from PIL import Image
from VEmbedBase import EmbeddingDatabase, Vit
from label import find_label, get_background_color
import json

ICON_SET_ROOT = Path("../IconSets")
CELL_ROOT = Path("../cell-images")
CELL_INFO_PATH = Path("../cells.json")


class Searcher:
    def __init__(self, embedding_dir):
        self.database = EmbeddingDatabase(embedding_dir)
        self.model = Vit('ViT-L-14', pretrained='laion2b_s32b_b82k')
        print(f"Loading embedding database from {ICON_SET_ROOT / 'embedding'}...")
        self.database.load_binary()
        self.model.load()
        self.eSet = self.database.get_embeddings(self.model)

        print(f"Loaded embedding database with {len(self.eSet.paths)} entries.")

    def remove_icon_set(self, icon_set_name):
        paths = self.eSet.paths
        idx_2_remove = [icon_set_name in p for p in paths]
        self.eSet.remove(idx_2_remove)

    def search_cell(self, cell_name, k=10):
        test_image = CELL_ROOT / cell_name
        label, _ = find_label(test_image)
        bg = get_background_color(Image.open(test_image))
        test_vec = self.model.embed_image(test_image)
        paths, scores  = self.eSet.top_k_combined(test_vec, label, k=k, alpha=0.1)
        return label, paths, scores, bg
    
    def search_text(self, text, k=10, alpha=0.5):
        test_vec = self.model.embed_text(text)
        paths, scores  = self.eSet.top_k_combined(test_vec, text, k=k, alpha=alpha)
        return paths, scores
    
    def search_text_request(self, request: web.Request):
        text = request.query.get("text", "")
        k = int(request.query.get("k", 10))
        alpha = float(request.query.get("alpha", 0.5))
        paths, scores = self.search_text(text, k, alpha)
        data = {
            'icons': [{'path': str(Path('file/IconSets') / p), 'score': s} for p, s in zip(paths, scores)],
        }
        return web.json_response(data)
        
    def search_cell_request(self, request: web.Request):
        cell_name = request.match_info["image"]
        # guard against missing/invalid selection from the frontend (e.g. 'null')
        if not cell_name or cell_name in ("null", "None"):
            return web.json_response({"error": "no image specified"}, status=400)

        k = int(request.query.get("k", 10))
        try:
            label, paths, scores, bg = self.search_cell(cell_name, k)
        except FileNotFoundError:
            return web.json_response({"error": f"file not found: {cell_name}"}, status=404)
        data = {
            'icons': [{'path': str(Path('file/IconSets') / p), 'score': s} for p, s in zip(paths, scores)],
            'label': label,
            'background': 'rgb(' + ','.join(map(str, bg)) + ')'
        }
        return web.json_response(data)

class InfoFinder:
    def __init__(self):
        self.cells = {}

        # open matching list
        if CELL_INFO_PATH.is_file():
            with open(CELL_INFO_PATH, "r") as f:
                self.cells = json.load(f)

        self.cell_paths = list(CELL_ROOT.glob("*.png"))
        self.cell_list = [str(p.relative_to(CELL_ROOT)) for p in self.cell_paths]

        self.que = self.get_button_queue()
        print(f"Cells remaining: {len(self.que)}")


    def get_button_queue(self):
        que = [(cell, path) for cell, path in zip(self.cell_list, self.cell_paths) if not cell in self.cells]
        if que == []:
            que = [(cell, path) for cell, path in zip(self.cell_list, self.cell_paths) if self.cells[cell].get("flagged", False)]
        return que
    
    def next_button(self):
        if self.que:
            return self.que.pop(0)
        return None, None
    
    def next_button_request(self, request: web.Request):
        cell, path = self.next_button()
        if cell is None:
            return web.json_response({"cell": None, "path": None})
        else:
            return web.json_response({"cell": cell, "path": str('file' / path.relative_to(Path("../")))})
    
    def add_info(self, cell, data):
        self.cells[cell] = data


    async def submit_info(self, request: web.Request):
        data = await request.json()
        cell = data["cell"]
        del data["cell"]
        self.add_info(cell, data)
        self.save()
        p, flag, no_flag, total = self.progress
        return web.json_response({"status": "success", "progress": p, "flag": flag, "no_flag": no_flag, "total": total})
    
    @property
    def progress(self):
        total = len(self.cell_list)
        no_flag = 0
        flag = 0
        for cell in self.cell_list:
            if cell in self.cells:
                if self.cells[cell].get("flagged", False):
                    flag += 1
                else:
                    no_flag += 1
        p = (no_flag + flag) / total

        return p, flag, no_flag, total
        

    def save(self):
        with open(CELL_INFO_PATH, "w") as f:
            json.dump(self.cells, f, indent=4)
            
