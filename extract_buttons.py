
# load json 
import json
from PIL import Image
from extract_grid import split_image, get_background_color
from imagehash import phash


def is_image_blank(img, threshold=1):
    # Convert to grayscale and check if the average pixel value is below the threshold
    gray = img.convert("L")
    max_pixel_value = max(gray.getdata())
    min_pixel_value = min(gray.getdata())
    return max_pixel_value - min_pixel_value < threshold

def sub2ind(row, col, size = (4, 9)):
    return row * size[1] + col

def ind2sub(index, size = (4, 9)):
    row = index // size[1]
    col = index % size[1]
    return row, col

def page_icons_by_ind(page):
    icons = {}
    for icon in page['icons']:
        icons[sub2ind(icon["row"], icon["col"])] = icon
    return icons

def isolate_icons(json_path, cell_image_dir, output_json_path=None):
    icon_cell_hashes = {}
    with open(json_path, 'r') as f:
        data = json.load(f)
        pages = data['pages']

        for id, page in pages.items():
            print(f"Page: {id}")
            img = Image.open(page['screenshot_path'])

            # get icons for page
            icons = page_icons_by_ind(page)
            cells = split_image(img, 4, 9, 4);
            
            new_icons = []
            # for each cell, if it contains an icon
            for i, cell in enumerate(cells):
                if not is_image_blank(cell) and i in icons:
                    icon = icons[i]
                    icon['background'] = get_background_color(cell)
                    
                    # Hash the cell image
                    h = phash(cell)
                    icon['cell_image'] = f"{h}"

                    # Save the cell image if it hasn't been saved before
                    if h not in icon_cell_hashes:
                        cell.save(f"{cell_image_dir}/{h}.png")
                        icon_cell_hashes[h] = True
                    
                    new_icons.append(icon)

            page['icons'] = new_icons

        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(data, f, indent=4)



                        
                    
                
                
        

if __name__ == "__main__":
    isolate_icons('state.json', 'cell-images', "unity36.json")