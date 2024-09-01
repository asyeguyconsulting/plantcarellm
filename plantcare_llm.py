import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import PyPDF2
import anthropic

# Initialize the Claude AI client with the provided API key
api_key = st.secrets['api_key']
client = anthropic.Anthropic(api_key=api_key)

# Load the plant identification model and feature extractor
model_name = "umutbozdag/plant-identity"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Function to extract text from an online PDF
def extract_pdf_text(pdf_url):
    try:
        # Fetch the PDF content from the URL
        response = requests.get(pdf_url)
        response.raise_for_status()  # Check if the request was successful
        
        # Ensure the content type is 'application/pdf'
        if response.headers['Content-Type'] != 'application/pdf':
            raise ValueError("URL did not point to a PDF file")

        # Read the PDF content using PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the PDF file: {e}")
        return ""
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"Error reading the PDF file: {e}")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return ""

# Function to find plant information in the PDF text
def find_plant_info(plant_name, pdf_text):
    # Basic keyword search; you can enhance this with NLP techniques
    if plant_name.lower() in pdf_text.lower():
        start = pdf_text.lower().find(plant_name.lower())
        end = pdf_text.find("\n", start + 500)  # Assuming relevant info is within 500 chars
        return pdf_text[start:end].strip()
    return "No specific information found for this plant."

# Streamlit app title and description
st.title("Plant Identification and Care Assistant")
st.write("Upload an image to identify the plant and get detailed care instructions.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Move model and inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference to identify the plant
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class index
    predicted_class_idx = logits.argmax(-1).item()

    # Access the labels directly from the model
    plant_name = model.config.id2label[predicted_class_idx]

    # Display the result
    st.write(f"**Identified Plant:** {plant_name}")

    # Display the image with the predicted label using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Identified Plant: {plant_name}", fontsize=16)
    ax.axis('off')
    st.pyplot(fig)

    # URL of the PDF containing care information
    pdf_url = "https://www.kellogggarden.com/wp-content/uploads/2020/05/Monthly-Flower-Gardening-Guide.pdf"

    # Extract text from the PDF
    pdf_text = extract_pdf_text(pdf_url)

    # Retrieve and display plant care information using Claude AI
    if pdf_text:
        plant_info = find_plant_info(plant_name, pdf_text)
        
        if plant_info != "No specific information found for this plant.":
            # Create the prompt for Claude AI
            prompt = f"I have the following information about the plant {plant_name}:\n\n{plant_info}\n\nPlease provide detailed care instructions based on this information."
            
            # Call Claude AI API
            try:
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=250,
                    temperature=0.7,
                    system="Use the information about the plant in the prompt to give tips on plant care, specifically water, soil and any other information",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                # Extract the useful text from the response
                useful_text = message['completion'] if 'completion' in message else "No response received."
                
                # Display the useful text in the text area
                st.text_area("Plant Care Instructions", value=useful_text, height=300)
            except Exception as e:
                st.error(f"An error occurred while calling Claude AI: {e}")
        else:
            st.warning(plant_info)
    else:
        st.warning("Failed to extract text from the PDF.")
