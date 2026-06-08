from PIL import Image
import Vision
import io
import objc
import numpy as np
from pathlib import Path


def get_background_color(img: Image.Image) -> tuple[int, int, int]:
    # Get the colors of the four corners
    corners = [img.getpixel((0, 0)), img.getpixel((img.width - 1, 0)),
               img.getpixel((0, img.height - 1)), img.getpixel((img.width - 1, img.height - 1))]
    # Return the most common color among the corners    
    return max(set(corners), key=corners.count) 


def is_text_on_background(img, coords, bg, tol=15, min_fraction=0.5):
    x, y, w, h = coords
    W, H = img.size
    region = img.crop((int(x*W), int((1-y-h)*H), int((x+w)*W), int((1-y)*H)))

    arr = np.array(region.convert("RGB"))

    diffs = np.linalg.norm(arr.astype(int) - np.array(bg[:3]), axis=2)
    return (diffs < tol).mean() > min_fraction

def to_ns_data(img: Image.Image):
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


def find_label(img: str) -> str:
    if isinstance(img, str) or isinstance(img, Path):
        img = Image.open(img)

    observations = get_text_observations(img)
    if not observations:
        return None, None
    
    # filter text thats too small or too large to be a label
    height_min = 0.07
    height_max = 0.135
    labels = list(filter(lambda x: x[1][3] > height_min and x[1][3] < height_max, observations))

    min_dist_from_center = 0.015
    labels = list(filter(lambda x: abs((x[1][0] + x[1][2]/2) - 0.5) < min_dist_from_center, labels))  # only consider text near horizontal center of image for labels

    bg = get_background_color(img)
    labels = list(filter(lambda x: is_text_on_background(img, x[1], bg), labels))  # only consider text on background for labels
    
    label = " ".join([l[0] for l in labels]) if labels else None
    return label, labels 

  