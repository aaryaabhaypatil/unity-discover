from test_candidate import test
from pathlib import Path    
from utils import Searcher, InfoFinder, ICON_SET_ROOT

# SCSH/S_0B.png

def auto():
    searcher = Searcher(ICON_SET_ROOT / "embedding")
    infoF = InfoFinder()
    searcher.remove_icon_set("SYBX")
    searcher.remove_icon_set("MTC")
    
    _, a, b, total =  infoF.progress

    remaining = total - b
    print(f"Progress: {a} correct, {b} flagged, {remaining} remaining")
    for i in range(remaining):
        cell_id, cell_path = infoF.next_button()
        label, paths, _,bg = searcher.search_cell(cell_id, k=30)

        fond = False
        if label is None:
            print(f"{i}/{remaining}: No label found for {cell_id}, flagging for review.")
            infoF.add_info(cell_id, {
                "label": None,
                "icon": None,
                "background": bg,
                "flagged": True
            })
            infoF.save()
            continue

        for j, p in enumerate(paths):
            try:
                diff_score = test(cell_path, ICON_SET_ROOT / p, label)
                if diff_score < 4:
                    print(f"{i}/{remaining}: Found match for {cell_id} (k = {j}) with score {diff_score}")
                    infoF.add_info(cell_id, {
                        "label": label,
                        "icon": p,
                        "background": bg,
                        "flagged": False
                    })
                    infoF.save()
                    fond = True
                    break
            except Exception as e:
                print(f"{i}/{remaining}: Error testing {p} against {cell_id}")
                print(e)
                continue

        if not fond:
            print(f"{i}/{remaining}: No good match found for {cell_id}, flagging for review.")
            infoF.add_info(cell_id, {
                "label": label,
                "icon": None,
                "background": bg,
                "flagged": True
            })

    
# def find_similar_icons(cell_id, k=10):
#     searcher = Searcher(ICON_SET_ROOT / "embedding")
#     searcher.remove_icon_set("SYBX")
#     searcher.remove_icon_set("MTC")
#     label, paths, scores, bg = searcher.search_cell(cell_id, k)
#     print(f"Label: {label}, Background: {bg}")
#     for p, s in zip(paths, scores):
#         print(f"Path: {p}, Score: {s}")

    
    



auto()
    


