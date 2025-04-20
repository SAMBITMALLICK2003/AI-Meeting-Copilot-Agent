"""
Image capture tools for the Gemini Agent Framework
"""

import asyncio
import base64
import io
import cv2
import PIL.Image
import mss
from ..tools.base_tool import BaseTool

class ImageCaptureTool(BaseTool):
    """Base class for image capture tools"""
    
    def __init__(self, name, description):
        super().__init__(name, description)


class CameraCaptureTool(ImageCaptureTool):
    """Tool for capturing images from the camera"""
    
    def __init__(self):
        super().__init__(
            name="CameraCapture", 
            description="Captures frames from the camera"
        )
        self.cap = None
        
    async def initialize(self):
        """Initialize the camera"""
        self.cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        
    async def execute(self):
        """Capture a frame from the camera"""
        if self.cap is None:
            await self.initialize()
            
        ret, frame = await asyncio.to_thread(self.cap.read)
        if not ret:
            return None

        # Convert BGR to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()


class ScreenCaptureTool(ImageCaptureTool):
    """Tool for capturing the screen"""
    
    def __init__(self):
        super().__init__(
            name="ScreenCapture", 
            description="Captures the current screen"
        )
        
    async def execute(self):
        """Capture the screen"""
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}