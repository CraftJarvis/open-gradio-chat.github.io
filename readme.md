Multimodal Vision-Language Chatbot Interface

This repository provides a chatbot interface designed to interact with multimodal data (images and text) using vision-language models. The interface uses Gradio for a web-based front-end and communicates with models deployed via OpenAI-compatible APIs. It supports selecting models from multiple deployment endpoints, generating responses, and providing interactive feedback for image and text inputs.

Features

	1.	Model Selection:
	•	Reads a YAML file to list available models deployed on specified IPs and ports. (TODO)
	•	Automatically detects available models using OpenAI-compatible APIs, creating a dictionary of accessible models. (DONE)
	•	Provides a dropdown in the Gradio interface to select models. (TODO)
	2.	Multimodal Input:
	•	Accepts both text and image inputs via a MultimodalTextbox. Users can interact with the chatbot by combining images and textual queries.
	•	Allows for multiple image uploads in a single input. (DONE)
	3.	Error Handling:
	•	Displays error messages in the front-end when model interactions fail. (TODO)
	4.	Interactive Parameters:
	•	Users can adjust parameters like temperature and maximum output tokens directly in the interface.
	5.	Example Inputs:
	•	Predefined examples guide users on how to interact with the interface using images and text.

Installation

Prerequisites

	•	Python 3.8 or later
	•	Required Python packages:

pip install gradio openai pyyaml pillow



Clone the Repository

git clone https://github.com/your-repo/multimodal-chatbot.git
cd multimodal-chatbot

Configuration

YAML File for Model Configuration

To manage models, define their IPs and ports in a YAML file. For example:

model:
  llama-3.1-8b:
    - ip: "localhost"
      port: 18001
    - ip: "100.1.100.122"
      port: 8001
  llama-3.1-70b: []

(Reading and using this file is currently marked as a TODO.)

Usage

Start the Server

Run the following command to launch the Gradio interface:

python chatbot.py --host localhost --port 19000

Access the Interface

Open your browser and navigate to:

http://localhost:19000

Interface Overview

Model Configuration

	•	Model Name: Select the model to interact with.
	•	Model URL: Specify the base URL of the model deployment.
	•	API Key: Provide the API key for authentication.

Parameters

	•	Temperature: Controls the randomness of the model’s responses.
	•	Max Output Tokens: Limits the maximum number of tokens generated in the response.

Example Inputs

	•	Example inputs guide users on how to interact with the chatbot using text and images.

File Structure

├── app.py               # Main script for running the Gradio interface
├── configs/
│   └── gradio/
│       └── vision_language_model.yaml  # Example YAML configuration for model endpoints
├── data/
│   └── images/              # Example images for predefined examples
└── README.md                # Documentation for the repository

Future Enhancements

	1.	Read Model Endpoints from YAML:
Automatically populate the dropdown with available models listed in the YAML file.
	2.	Error Display in Front-end:
Show detailed error messages in the Gradio interface when model interactions fail.
	3.	Improved User Experience:
Enhance the visual design and interactivity of the interface.

References

	•	Gradio Documentation
	•	Hugging Face LLaVA Example
	•	VLLM Chatbot Example

License

This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to contribute to the repository by submitting issues or pull requests! 🚀