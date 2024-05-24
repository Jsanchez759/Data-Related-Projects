import requests
import warnings
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

warnings.filterwarnings('ignore') 

# Load the pre-trained model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load and process the image
img_path = "Foto.jpg"
image = Image.open(img_path).convert('RGB')

# Process the output
text = "the image of"
inputs = processor(images = image, text = text, return_tensors = "pt")

output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens = True)

print(caption)