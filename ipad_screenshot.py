from pymobiledevice3.lockdown import create_using_usbmux
from pymobiledevice3.services.screenshot import ScreenshotService
from PIL import Image
import io
from extract_grid import parse_grid, GridIcon


async def screenshot():
    async with await create_using_usbmux() as lockdown:
        async with ScreenshotService(lockdown) as service:
            data = await service.take_screenshot()
            return Image.open(io.BytesIO(data))
    return None


async def capture_grid() -> list[GridIcon]:
    img = await screenshot()
    return parse_grid(img, (2, 196, 2356, 1608))
    

