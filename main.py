from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from ultralytics import YOLO
import cv2
from typing import Optional
from vectorizer_ai import VectorizerAI
import vtracer
from dotenv import load_dotenv
from PIL import Image
import torch
import io
from RealESRGAN import RealESRGAN
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Global variables for Real-ESRGAN model
device = None
realesrgan_model = None

def load_realesrgan_model():
    """Load the Real-ESRGAN model with x4plus_anime_6B weights"""
    global device, realesrgan_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model with 6 blocks for anime model
    realesrgan_model = RealESRGAN(device, scale=4, model_name='x4plus_anime_6B')
    
    # Load the anime model weights
    model_path = 'weights/RealESRGAN_x4plus_anime_6B.pth'
    if os.path.exists(model_path):
        realesrgan_model.load_weights(model_path, download=False)
        logger.info("Successfully loaded RealESRGAN_x4plus_anime_6B model")
    else:
        logger.warning(f"Model weights not found at {model_path}. Upscaling will be skipped.")
        realesrgan_model = None

app = FastAPI(
    title="YOLO API with Real-ESRGAN",
    description="API for object detection, image vectorization with upscaling, and image cropping"
)
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers.
)
# Configure folders
UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./outputs"
SVG_FOLDER = "./svg_outputs"

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, SVG_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load the YOLO model
model = YOLO("./model/best.pt", task='detect')

@app.on_event("startup")
async def startup_event():
    """Load Real-ESRGAN model on startup"""
    try:
        load_realesrgan_model()
    except Exception as e:
        logger.error(f"Failed to load Real-ESRGAN model: {e}")
        # Don't raise the error, just log it so the app can still start

@app.get("/")
def home():
    return {"message": "Welcome to the Vectify API with Real-ESRGAN upscaling....."}

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    """
    Detect objects in an image using YOLO model
    """
    # Check if image was uploaded
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # Save the uploaded image
    filename = image.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    # Perform object detection
    results = model(filepath)
    
    # Process results
    detections = []
    
    # Read the original image for drawing bounding boxes
    img = cv2.imread(filepath)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Get class name if available
            class_name = model.names[class_id] if hasattr(model, 'names') else f"class_{class_id}"
            
            # Convert coordinates to integers for drawing
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append({
                'bounding_box': {
                    'x1': round(x1, 2),
                    'y1': round(y1, 2),
                    'x2': round(x2, 2),
                    'y2': round(y2, 2)
                },
                'class_id': class_id,
                'class_name': class_name,
                'confidence': round(confidence, 2)
            })
    
    # Save the annotated image to the output folder
    output_filename = f"annotated_{filename}"
    output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_filepath, img)
    
    return {
        'success': True,
        'detections': detections,
        'annotated_image_path': output_filepath
    }

@app.post("/vectorize")
async def vectorize_image(image: UploadFile = File(...)):
    """
    Convert an image to SVG with upscaling and noise reduction
    """
    # Check if image was uploaded
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # Save the uploaded image
    filename = image.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    logger.info(f"Saving uploaded image to {filepath}")
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    # Generate output SVG filename and path
    svg_filename = os.path.splitext(filename)[0] + '.svg'
    svg_filepath = os.path.join(SVG_FOLDER, svg_filename)
    
    try:
        # # Step 1: Read and preprocess the image
        # img = cv2.imread(filepath)
        # if img is None:
        #     raise HTTPException(status_code=500, detail="Failed to read image")
        
        # # Apply denoising
        # denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # # Apply mean shift filtering for further noise reduction and edge preservation
        # filtered_img = cv2.pyrMeanShiftFiltering(denoised, sp=20, sr=40, maxLevel=2)
        
        # # Save the processed image
        # processed_filepath = os.path.join(UPLOAD_FOLDER, f"processed_{filename}")
        # cv2.imwrite(processed_filepath, filtered_img)
        
        # # Step 2: Upscale the processed image using Real-ESRGAN if model is available
        # if realesrgan_model is not None:
        #     logger.info("Upscaling processed image using Real-ESRGAN")
            
        #     # Read processed image with PIL for upscaling
        #     pil_image = Image.open(processed_filepath)
            
        #     # Validate image size for upscaling
        #     width, height = pil_image.size
        #     if width >= 5000 or height >= 5000:
        #         logger.warning("Image too large for upscaling, skipping upscaling step")
        #         upscaled_image = pil_image
        #     elif width < 10 or height < 10:
        #         logger.warning("Image too small for upscaling, skipping upscaling step")
        #         upscaled_image = pil_image
        #     else:
        #         # Convert to RGB if necessary
        #         if pil_image.mode != 'RGB':
        #             pil_image = pil_image.convert('RGB')
                
        #         logger.info(f"Upscaling image: {width}x{height} -> {width*4}x{height*4}")
        #         upscaled_image = realesrgan_model.predict(pil_image)
            
        #     # Save upscaled image
        #     upscaled_filename = f"upscaled_{filename}"
        #     upscaled_filepath = os.path.join(OUTPUT_FOLDER, upscaled_filename)
        #     upscaled_image.save(upscaled_filepath)
            
        #     # Use upscaled image for vectorization
        #     final_image_path = upscaled_filepath
        #     logger.info("Image upscaling completed successfully")
        # else:
        #     logger.info("Real-ESRGAN model not available, using processed image")
        #     # Use the processed (denoised) image for vectorization
        #     final_image_path = processed_filepath
        
        # Step 3: Vectorize using VectorizerAI
        logger.info("Starting vectorization process")
        client = VectorizerAI(
                api_id=os.getenv("VECTORIZER_API_ID"),
                api_secret=os.getenv("VECTORIZER_API_SECRET"),
                mode=os.getenv("VECTORIZER_MODE", "production")
            )
        # svg = client.vectorize(final_image_path)
        svg = client.vectorize(filepath)

          # Convert the processed image to SVG using VTracer
        # vtracer.convert_image_to_svg_py(
        #     processed_filepath, 
        #     svg_filepath,
        #     colormode="color",          # Full-color mode
        #     hierarchical="stacked",     # Stacked shapes for compact output
        #     mode="spline",              # Smooth curves for sharp edges
        #     filter_speckle=6,           # Remove small noise (adjustable)
        #     color_precision=7,          # Color accuracy (6-8 bits)
        #     layer_difference=16,        # Color layer separation
        #     corner_threshold=60,        # Angle to detect corners
        #     length_threshold=4.0,       # Min segment length
        #     max_iterations=10,          # Curve fitting iterations
        #     splice_threshold=45,        # Spline splicing angle
        #     path_precision=9            # Decimal precision in paths
        # )
          # Check if SVG was created successfully
        # if not os.path.exists(svg_filepath):
        #     raise HTTPException(status_code=500, detail="SVG conversion failed")
        
        # # Read the SVG content
        # with open(svg_filepath, 'r') as svg_file:
        #     svg_content = svg_file.read()
        
        # Return the SVG content as a Response with appropriate content type
        logger.info("Vectorization completed successfully")
        return Response(
            content=svg,
            media_type="image/svg+xml"
            )

    except Exception as e:
        logger.error(f"Error during vectorization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        'status': 'healthy', 
        'yolo_model': 'YOLO object detection',
        'realesrgan_model': 'available' if realesrgan_model is not None else 'not available',
        'device': str(device) if device is not None else 'unknown'
    }

@app.post("/crop")
async def crop_image(
    image: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...)
):
    """
    Crop an image using provided coordinates
    """
    # Check if image was uploaded
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # Validate coordinates
    if x1 >= x2 or y1 >= y2:
        raise HTTPException(status_code=400, detail="Invalid bounding box coordinates. Ensure x1 < x2 and y1 < y2")
    
    # Save the uploaded image
    filename = image.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    try:
        # Read the image
        img = cv2.imread(filepath)
        if img is None:
            raise HTTPException(status_code=500, detail="Failed to read image")
        
        # Convert coordinates to integers for cropping
        height, width = img.shape[:2]
        x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
        x2_int, y2_int = min(width, int(x2)), min(height, int(y2))
          # Crop the image
        cropped_image = img[y1_int:y2_int, x1_int:x2_int]
        
        # Save the cropped image
        cropped_filename = f"cropped_{filename}"
        cropped_filepath = os.path.join(OUTPUT_FOLDER, cropped_filename)
        cv2.imwrite(cropped_filepath, cropped_image)
        
        # Read the image file to return it directly
        with open(cropped_filepath, "rb") as image_file:
            image_content = image_file.read()
            
        # Determine the content type based on the file extension
        content_type = "image/jpeg"  # Default
        if filename.lower().endswith(".png"):
            content_type = "image/png"
        elif filename.lower().endswith(".gif"):
            content_type = "image/gif"
        
        # Return the cropped image directly as a response
        return Response(
            content=image_content,
            media_type=content_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image cropping failed: {str(e)}")

@app.post("/upscale")
async def upscale_image(image: UploadFile = File(...)):
    """
    Upscale an image using Real-ESRGAN x4plus_anime_6B model
    
    Args:
        image: Image file to upscale (supported formats: JPEG, PNG, WebP)
    
    Returns:
        Upscaled image as PNG
    """
    if realesrgan_model is None:
        raise HTTPException(status_code=500, detail="Real-ESRGAN model not loaded")
    
    # Check if image was uploaded
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if image.content_type not in allowed_types:
        # Also try to detect by file extension as fallback
        filename = image.filename.lower() if image.filename else ""
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {image.content_type}. Please upload JPEG, PNG, or WebP images."
            )
    
    try:
        # Save the uploaded image
        filename = image.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        logger.info(f"Saving uploaded image to {filepath}")
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Read and validate image
        pil_image = Image.open(filepath)
        
        # Check image size limitations
        width, height = pil_image.size
        if width >= 5000 or height >= 5000:
            raise HTTPException(
                status_code=400, 
                detail="Image too large. Maximum dimensions: 5000x5000 pixels."
            )
        
        if width < 10 or height < 10:
            raise HTTPException(
                status_code=400, 
                detail="Image too small. Minimum dimensions: 10x10 pixels."
            )
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        logger.info(f"Processing image: {width}x{height} -> {width*4}x{height*4}")
        
        # Perform upscaling
        upscaled_image = realesrgan_model.predict(pil_image)
        
        # Save upscaled image
        upscaled_filename = f"upscaled_{filename}"
        upscaled_filepath = os.path.join(OUTPUT_FOLDER, upscaled_filename)
        upscaled_image.save(upscaled_filepath)
        
        # Read the upscaled image file to return it directly
        with open(upscaled_filepath, "rb") as image_file:
            image_content = image_file.read()
        
        logger.info("Image upscaling completed successfully")
        
        # Return the upscaled image directly as a response
        return Response(
            content=image_content,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=upscaled_{filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
