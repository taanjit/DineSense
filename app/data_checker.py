from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging

# import dine

app = FastAPI()

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        logger.warning("Uploaded file is not an image.")
        return JSONResponse(status_code=400, content={"error": "File is not an image"})

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        logger.info(f"Received image: {file.filename} ({image.format}, {image.size})")

        # Example: Return image metadata
        response = {
            "filename": file.filename,
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height
        }

        return response

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to process image"})
