import io
from dataclasses import dataclass
from PIL import Image
try:
    import Vision
    import objc
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing macOS Vision framework bindings. Install them with: "
        "pip install pyobjc pyobjc-framework-Vision"
    ) from exc
import numpy as np

def isolate_icon(img: Image.Image, bg_tol: float = 5) -> Image.Image:
    """Find bounding box of the main graphic by detecting non-background pixels.
    Background colour is sampled from the corners of the image.
    Background color is then set to transparent, and the image 
    is cropped to the bounding box of non-background pixels.
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba)

    # Sample background from corners
    corners = [arr[0,0], arr[0,-1], arr[-1,0], arr[-1,-1]]
    bg = np.mean(corners, axis=0)[:3]

    # Find pixels that differ from background
    diff = np.abs(arr[:, :, :3].astype(int) - bg).sum(axis=2)
    mask = diff > bg_tol

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return img
    
    cropped = img.crop((cols[0], rows[0], cols[-1]+1, rows[-1]+1))

    # Set background pixels to transparent
    cropped_arr = np.array(cropped.convert("RGBA"))
    cropped_arr[~mask[rows[0]:rows[-1]+1][:, cols[0]:cols[-1]+1]] = [0, 0, 0, 0]
    cropped = Image.fromarray(cropped_arr)  

    return cropped

    

@dataclass
class GridIcon:
    background: tuple[int, int, int]  # RGB background colour
    label: list[str] | None  # text label(s) detected on background
    icon: Image.Image | None  # cropped icon region
    row: int
    col: int


def get_background_color(img: Image.Image) -> tuple[int, int, int]:
    # Get the colors of the four corners
    corners = [img.getpixel((0, 0)), img.getpixel((img.width - 1, 0)),
               img.getpixel((0, img.height - 1)), img.getpixel((img.width - 1, img.height - 1))]
    # Return the most common color among the corners    
    return max(set(corners), key=corners.count) 

def is_text_on_background(img, coords, bg, tol=1):
    x,y,w,h = coords
    W, H = img.size
    # sample corners of text box
    samples = [
        img.getpixel((int(x*W), int((1-y)*H))),  # bottom-left
        img.getpixel((int((x+w)*W), int((1-y)*H))),  # bottom-right
        img.getpixel((int(x*W), int((1-y-h)*H))),  # top-left
        img.getpixel((int((x+w)*W), int((1-y-h)*H)))  # top-right
    ]
    # check if all corners are similar (indicating text on solid background)
    return all(np.linalg.norm(np.array(s) - np.array(bg)) < tol for s in samples)

def to_ns_data(img: Image.Image) -> objc.lookUpClass("NSData"):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return objc.lookUpClass("NSData").dataWithBytes_length_(buf.getvalue(), buf.tell())

def get_text_observations(img: Image.Image):
    """Returns list of (text, norm_bbox) where norm_bbox is (x, y, w, h) 
    in Vision coords (origin bottom-left, y=1 is top of image)."""
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(to_ns_data(img), {})
    handler.performRequests_error_([request], None)
    request.setUsesLanguageCorrection_(False)
    results = []
    for obs in request.results():
        bb = obs.boundingBox()
        top = obs.topCandidates_(1)[0]
        text = top.string()
        score = top.confidence()
        results.append((text, (bb.origin.x, bb.origin.y, bb.size.width, bb.size.height), score))
    return results

def parse_image(img: Image.Image) -> GridIcon:
    """
    label_zone: normalised y threshold above which text is considered the label.
                In Vision coords (bottom-left origin), y > label_zone means top of image.
    """
    w, h = img.size
    observations = get_text_observations(img)
    background = get_background_color(img)
    label = None
    icon = None

    obs = " | ".join([f"{o[0]}" for o in observations])
    print(f"Observations: {obs}, background: {background}")
    if not observations:
        icon = isolate_icon(img)
    else:
        observations = [(text, is_text_on_background(img, c, background), c) for text, c, _ in observations]
        labels = list(filter(lambda x: x[1], observations))  # only consider text on background for labels
        if labels:
            label = " ".join([l[0] for l in labels]) 
            min_label_y = min([c[2][1] for c in labels])
            icon_area = img.crop((0, int((1-min_label_y)*h), w, h))
            icon = isolate_icon(icon_area)
        else:
            icon = isolate_icon(img)

    return GridIcon(label=label, icon=icon, background=background, row=-1, col=-1)

def split_image(img, rows, cols, space):
    width, height = img.size
    cell_width = (width - (cols - 1) * space) // cols
    cell_height = (height - (rows - 1) * space) // rows

    images = []
    for row in range(rows):
        for col in range(cols):
            left = col * (cell_width + space)
            upper = row * (cell_height + space)
            right = left + cell_width
            lower = upper + cell_height
            images.append(img.crop((left, upper, right, lower)))
    return images

def img_is_white(img, threshold=254.9):
    # Convert to grayscale and check if the average pixel value is below the threshold
    gray = img.convert("L")
    avg_pixel_value = sum(gray.getdata()) / (gray.width * gray.height)
    return avg_pixel_value > threshold

async def parse_grid(img, crop: tuple = None) -> list[GridIcon]:
    icons = []
    if crop:
        img = img.crop(crop)  # (left, upper, right, lower)
    imgs = split_image(img, 4, 9, 4)
    for i, im in enumerate(imgs):
        row = i // 9
        col = i % 9
        # im.save(f"{filename}_{row}_{col}.{t}")
        if not img_is_white(im):
            icon = parse_image(im)
            icon.row = row
            icon.col = col
            icons.append(icon)
    return icons
