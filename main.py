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
load_dotenv()
app = FastAPI(
    title="YOLO API",
    description="API for object detection, image vectorization, and image cropping"
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
@app.get("/")
def home():
    return {"message": "Welcome to the Vectify API....."}

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
    Convert an image to SVG using VTracer with noise reduction
    """
    # Check if image was uploaded
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # Save the uploaded image
    filename = image.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    print(f"Saving uploaded image to {filepath}")
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    # Generate output SVG filename and path
    svg_filename = os.path.splitext(filename)[0] + '.svg'
    svg_filepath = os.path.join(SVG_FOLDER, svg_filename)
    
    
    # Process the image to remove noise
    try:
        # Read the image
        img = cv2.imread(filepath)
        if img is None:
            raise HTTPException(status_code=500, detail="Failed to read image")
        
        # Step 1: Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Step 2: Apply mean shift filtering for further noise reduction and edge preservation
        filtered_img = cv2.pyrMeanShiftFiltering(denoised, sp=20, sr=40, maxLevel=2)
        
        # Save the processed image
        processed_filepath = os.path.join(UPLOAD_FOLDER, f"processed_{filename}")
        cv2.imwrite(processed_filepath, filtered_img)
        
        client = VectorizerAI(
                api_id=os.getenv("VECTORIZER_API_ID"),
                api_secret=os.getenv("VECTORIZER_API_SECRET"),
                mode=os.getenv("VECTORIZER_MODE", "production")
            )
        svg = client.vectorize(processed_filepath)

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
        # Using Response instead of JSONResponse because SVG isn't JSON
        return Response(
            content=svg,
            media_type="image/svg+xml"
            )

    except Exception as e:
        print(f"Error during vectorization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {'status': 'healthy', 'model': 'YOLO object detection'}

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
