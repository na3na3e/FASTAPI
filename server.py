# Import section
import io
import json
from PIL import Image, ImageDraw
from fastapi import File, FastAPI, Form
from fastapi.responses import FileResponse
import torch
import tempfile
import os
import time
import requests

#C:\Users\Hamza\Downloads\unnamed.jpg

# Global variable to keep track of the number of iterations
i = 0
previous_folder_name = ""

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"dam_det.pt")
model.conf = 0.5

# Create your API
app = FastAPI()


# Helper function to draw bounding boxes on the image with thicker lines and display the class name and confidence
def draw_boxes_with_thicker_lines(image, boxes, line_width=10):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x_min, y_min, x_max, y_max, conf, cls = box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=line_width)
        draw.text((x_min, y_min), f"{cls} - {conf:.2f}", fill=(255, 0, 0))

   
# Set up a temporary directory to save the image
temp_dir = tempfile.mkdtemp()


# Helper function to save the image with detections and thicker lines to the temporary directory
def save_image_with_detections(image, boxes, output_path):
    draw_boxes_with_thicker_lines(image, boxes, line_width=10)
    image.save(output_path, format="PNG")


# Define the mapping function for 'class' to 'Pièce'
def map_class_to_piece(class_value):
    pieces = {
        0: 'Dommage au pare-brise avant',
        1: 'Dommage aux phares',
        2: "Grosse bosse à l'arrière du pare-chocs",
        3: 'Dommage au pare-brise arrière',
        4: 'Bosse sur le marchepied',
        5: 'Dommage au rétroviseur',
        6: 'Dommage aux feux de signalisation',
        7: 'Dommage aux feux arrière',
        8: 'Bosse sur le capot',
        9: 'Bosse sur la porte extérieure',
        10: "Bosse sur l'aile",
        11: 'Bosse sur le pare-chocs avant',
        12: 'Bosse sur le panneau de carrosserie moyen',
        13: "Bosse sur le pilier",
        14: "Bosse sur l'aile",
        15: 'Bosse sur le pare-chocs arrière',
        16: 'Bosse sur le toit'
    }
    return pieces.get(class_value, 'Unknown')


# Set up your API and integrate your ML model
@app.post("/objectdetection/")
async def get_body(folder_name: str, image_path_or_link: str = Form(...)):
    global i  # Declare 'i' as a global variable

    # Load the image from the provided image_path_or_link (whether it's a local path or a URL)
    if image_path_or_link.startswith(("http://", "https://")):
        # If it's a URL, download the image from the URL
        response = requests.get(image_path_or_link)
        input_image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        # Otherwise, treat it as a local file path
        input_image = Image.open(image_path_or_link).convert("RGB")

    results = model(input_image)
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

    # Map 'class' to 'Pièce' descriptions
    for result in results_json:
        class_value = result["class"]
        result["Pièce"] = map_class_to_piece(class_value)

    # Save the image with drawn boxes to the temporary directory
    timestamp = int(time.time())  # Get the current timestamp
    image_with_detections_path = os.path.join(temp_dir, f"{folder_name}_detection_{timestamp}.png")
    save_image_with_detections(input_image, results.pred[0].detach().cpu().numpy(), image_with_detections_path)

    # Define the path to the public directory where you want to save the image
    public_image_directory = "C:\\Users\\Hamza\\OneDrive\\Images\\f"

    # Move the image to the publicly accessible location on your server
    public_image_path = os.path.join(public_image_directory, f"{folder_name}_detection_{timestamp}.png")
    os.rename(image_with_detections_path, public_image_path)

    # Generate the URL for the image
    image_url = f"C:\\Users\\Hamza\\OneDrive\\Images\\f/{folder_name}_detection_{timestamp}.png"

    return {"result": results_json, "image_with_detections_url": image_url}
