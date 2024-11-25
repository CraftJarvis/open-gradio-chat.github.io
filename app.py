import yaml
import argparse
import gradio as gr
from openai import OpenAI
from PIL import Image

web_prefix = ""
def message_format(message):
    result = []
    if message["text"] is not None:
        result.append({"type": "text","text": message["text"]})
    for image_file in message["files"]:
        result.append({"type": "image_url","image_url": {"url": image_file['url']}})
        global web_prefix
        if web_prefix == "":
            web_prefix = image_file['url'].split("=")[0] + "="
    return result

# with open("configs/gradio/vision_language_model.yaml", "r") as file:
#     models = yaml.safe_load(file)['model']

# model_names = list(models.keys())

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=10001)

# Parse the arguments
args = parser.parse_args()

def predict(message, history, model, model_url, api_key, temp, max_output_tokens):
    client = OpenAI(
        # api_key="EMPTY",
        # base_url=f"http://{models[model]['model_url']}:{models[model]['model_port']}/v1",
        api_key=api_key,
        base_url=model_url,
    )
    # Convert chat history to OpenAI format
    history_openai_format = [
        # {
        #     "role": "system",
        #     "content": "You are a great ai assistant."
        # }
    ]
    for human, assistant in history:
        if assistant is not None:
            history_openai_format.append({
                "role": "assistant",
                "content": assistant
            })

        if type(human) is str:
            history_openai_format.append({"role": "user", "content": human})
        else:
            content = []
            for image_url in human:
                global web_prefix
                content.append({"type": "image_url","image_url": {"url": f"{web_prefix}{image_url}"}})
            history_openai_format.append({"role": "user", "content": content})

    message = message_format(message)
    history_openai_format.append({"role": "user", "content": message})


    # Create a chat completion request and send it to the API server
    try:
        stream = client.chat.completions.create(
            model=model,  # Model name to use
            messages=history_openai_format,  # Chat history
            temperature=temp,  # Temperature for text generation
            stream=True,  # Stream response
            max_tokens = max_output_tokens,
            # extra_body={
            #     'repetition_penalty':
            #     1,
            #     'stop_token_ids': [
            #         int(id.strip()) for id in args.stop_token_ids.split(',')
            #         if id.strip()
            #     ] if args.stop_token_ids else []}
            )

        # Read and return generated text from response stream
        partial_message = ""
        for chunk in stream:
            try:
                partial_message += (chunk.choices[0].delta.content or "")
            except:
                pass
            yield partial_message
    except Exception as e:
        raise gr.Error(str(e))


multimodaltextbox = gr.MultimodalTextbox()
with gr.Blocks(fill_height=True) as demo:
    with gr.Row():
        with gr.Column(scale=3):
            # model = gr.Dropdown(
            #     choices=model_names,
            #     value=model_names[0] if len(model_names) > 0 else "",
            #     interactive=True,
            #     label="Model")
            model = gr.Textbox(value="gpt-4o", label="Model name")
            model_url = gr.Textbox(value="https://api.openai.com/v1", label="Model URL")
            api_key = gr.Textbox(value="EMPTY", label="API Key", type="password")

            with gr.Accordion("Parameters", open=False) as parameter_row:
                temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Temperature",)
                max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            gr.Examples(examples=[
                {"files":["data/images/007-dark_forest.png"], "text": "What are the red structures visible in the background?"},
                {"files":["data/images/030-villager.png"], "text": "What is the role of the villager seen in the image?"},
                {"files":["data/images/038-inventory.png"], "text": "What type of armor is the player wearing?"},
                {"files":[], "text": "Give you nothing in the inventory, generate a step-by-step plan to obtain diamonds."},
            ], inputs=multimodaltextbox)

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(height=650)
            multimodaltextbox.render()
            gr.ChatInterface(predict, chatbot=chatbot, textbox=multimodaltextbox, multimodal=True,
                            additional_inputs=[model, model_url, api_key, temp, max_output_tokens], fill_height=True)

demo.queue().launch(server_name=args.host, server_port=args.port, share=True)