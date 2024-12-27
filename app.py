import yaml
import argparse
import gradio as gr
from openai import OpenAI
import requests
import base64
from pathlib import Path

from rich.console import Console
from rich.table import Table

from datetime import datetime
import json 
import os 

from gradio import ChatMessage

console = Console()

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

def message_format(message):
    message_content = []
    # print("message:", message)
    if 'files' in message.keys() and message['files']:
        for msg_file in message['files']:
            image_base64 = encode_image_base64(msg_file)
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            # # choice 1: load local image path into base64
            # if 'path' in msg_file.keys() and msg_file['path']:
            #     image_base64 = encode_image_base64(msg_file['path'])
            #     message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            # # choice 2: load url, which is not encouraged
            # elif 'url' in msg_file.keys() and msg_file['url']:
            #     # image_base64 = encode_image_base64(msg_file['url'])
            #     # message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            #     message_content.append({"type": "image_url", "image_url": {"url": msg_file['url']}})
            # else:
            #     raise ValueError("Invalid file format. Must have a valid path or url.")
    if 'text' in message.keys() and message['text']:
        message_content.append({"type": "text", "text": message['text']})
    return message_content

def append_history(history, message):
    # output_history: [{'role': 'user', 'metadata': {'title': None}, 'content': ('/tmp/gradio/e508fa9046b1c35d5cec52b46fc3d1dab59960d4c69f2838491414abd55c0a35/007-dark_forest.png',), 'options': None}, {'role': 'user', 'metadata': {'title': None}, 'content': 'What are the red structures visible in the background?', 'options': None}]
    # message: {'text': 'What are the red structures visible in the background?', 'files': ['/tmp/gradio/e508fa9046b1c35d5cec52b46fc3d1dab59960d4c69f2838491414abd55c0a35/007-dark_forest.png']}
    if 'files' in message.keys() and message['files']:
        for msg_file in message['files']:
            history.append({"role": "user", "content": (msg_file,)})
    if 'text' in message.keys() and message['text']:
        history.append({"role": "user", "content": message['text']})
    return history

def history_format(history):
    """
    将对话转换为 OpenAI 支持的格式。
    连续的两个 user 消息会合并。
    
    Args:
        conversation (list): 原始对话列表，格式如 [(user_msg, assistant_msg), ...]

    Returns:
        list: 转换后的 OpenAI 对话格式
    """
    formatted_conversation = []
    conv_buffer = []

    for turn in history:
        # user_message, assistant_message = turn

        # 合并连续的用户消息
        if turn['role'] == 'user':
            user_message = turn['content'] 
            if isinstance(user_message, str):
                conv_buffer.append({"role": "user", "content": [{"type": "text", "text": user_message}]})
            elif Path(user_message[0]).is_file():
                image_base64 = encode_image_base64(user_message[0])
                conv_buffer.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]})
            else:
                raise gr.Error("Can not parse user messages in history! Invalid input string. Must be a valid URL or file path.")
        if turn['role'] == 'assistant':
            assistant_message = turn['content']
            if isinstance(assistant_message, str):
                conv_buffer.append({"role": "assistant", "content": assistant_message})
            elif Path(assistant_message[0]).is_file():
                continue # skip the assistant image message, the returned image only need to show image 
            else:
                raise gr.Error("Can not parse assistant messages in history! Invalid input string. Must be a string.")
    
    prev_role = None
    for conv in conv_buffer:
        if conv['role'] == prev_role:
            formatted_conversation[-1]['content'] += conv['content']
        else:
            formatted_conversation.append(conv)
        prev_role = conv['role']
    return formatted_conversation


def encode_image_base64(image_input) -> str:
    """
    Encode an image to base64 format.
    Supports: URL, local file path, 
    Args:
        image_input (str | np.ndarray | PIL.Image.Image): Input image in different formats.

    Returns:
        str: Base64-encoded string of the image.
    """
    # Case 1: If the input is a URL (str)
    if isinstance(image_input, str):
        if image_input.startswith('http://') or image_input.startswith('https://'):
            try:
                response = requests.get(image_input)
                response.raise_for_status()
                return base64.b64encode(response.content).decode('utf-8')
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to retrieve the image from the URL: {e}")
        elif Path(image_input).is_file():  # Local file path
            try:
                with open(image_input, 'rb') as file:
                    return base64.b64encode(file.read()).decode('utf-8')
            except Exception as e:
                raise ValueError(f"Failed to read image file: {e}")
        else:
            raise ValueError("Invalid input string. Must be a valid URL or file path.")
    # Raise an error if the input type is unsupported
    else:
        raise ValueError("Unsupported input type. Must be a URL (str) or a local file path (str).")

def predict(message, history, model, model_url, api_key, temp, max_output_tokens, stream):
    original_history = history.copy()
    print("message:", message) # {'text': 'What is the role of the villager seen in the image?', 'files': [{'path': '/tmp/gradio/bd3ef8883b88f81857dfdb68ebbc757024d4fa718e1e0a138e805f27c1cd245a/030-villager.png', 'url': 'https://72721a834ae34c0685.gradio.live/file=/tmp/gradio/bd3ef8883b88f81857dfdb68ebbc757024d4fa718e1e0a138e805f27c1cd245a/030-villager.png', 'size': None, 'orig_name': '030-villager.png', 'mime_type': 'image/png', 'is_stream': False, 'meta': {'_type': 'gradio.FileData'}}]}
    print("history:", history) # [[('/tmp/gradio/6d6fecf474fc8192b4738918f0162bc731dfdf04eaf060402aa9a9c5ffe9051d/007-dark_forest.png',), None], ['What are the red structures visible in the background?', 'The red structures in the background are giant mushrooms, commonly found in the Roofed Forest biome in Minecraft.']]
    client = OpenAI(
        api_key=api_key,
        base_url=model_url,
    )
    # Convert chat history to OpenAI format
    post_conv = [
        # {"role": "system", "content": "You are a great ai assistant."}
    ]
    post_conv += history_format(history)
    # post_conv = history

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Time", style="dim", width=20)
    table.add_column("Model", style="dim", width=10)
    table.add_column("History", style="dim", width=30)
    table.add_column("Message", style="dim", width=20)
    table.add_column("Generation", style="dim", width=20)

    formatted_message = message_format(message)
    post_conv.append({"role": "user", "content": formatted_message})

    # print("post conversation:", post_conv)

    # official openai chat template is 
    # [{"role":"user", "content": [{"type": "text", "text": prompt}, {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}]
    # Create a chat completion request and send it to the API server

    history = append_history(history, message)

    try:
        response = client.chat.completions.create(
            model=model,  # Model name to use
            messages=post_conv,  # Chat history
            temperature=temp,  # Temperature for text generation
            stream=stream,  # Stream response
            max_tokens = max_output_tokens,
            # extra_body={
            #     'repetition_penalty':
            #     1,
            #     'stop_token_ids': [
            #         int(id.strip()) for id in args.stop_token_ids.split(',')
            #         if id.strip()
            #     ] if args.stop_token_ids else []}
        )
        if stream:
            # Read and return generated text from response stream
            partial_message = ""
            history.append({"role": "assistant", "content": partial_message})
            for chunk in response:
                try:
                    partial_message += (chunk.choices[0].delta.content or "")
                except:
                    pass
                history[-1]["content"] = partial_message
                yield "", history
                yield partial_message
                # yield partial_message
                # return_msg["content"] = {"type": "text", "text": partial_message}
                # yield return_msg
                # return_msg.append({"type": "text", "text": partial_message})
        else:
            partial_message = response.choices[0].message.content
            history.append({"role":"assistant", "content": partial_message})
            yield "", history
            # Read and return generated text from response stream
            # yield partial_message
            # return_msg["content"] = {"type": "text", "text": partial_message}
            # yield return_msg
            # history.append(
            #     ChatMessage(
            #         role="assistant", content=partial_message
            #     )
            # )
            # yield history   
            # return_msg.append({"type": "text", "text": partial_message})
    except Exception as e:
        raise gr.Error(str(e))

    post_conv.append({"role": "assistant", "content": partial_message})
    
    # test show tool or thought
    # if True:
    #     history.append({
    #         "role":"assistant",
    #         "content":"The weather API says it is 20 degrees Celcius in New York.",
    #         "metadata":{"title": "🛠️ Used tool Weather API"}
    #     })
    #     yield "", history

    # test embed image
    # if True:
    #     # Embed the quaterly sales report in the chat
    #     history.append(
    #         {"role": "assistant", "content":{"path": "data/images/007-dark_forest.png", "alt_text": "dark forest"}}
    #     )
    #     yield "", history

    # # test embed some file 
    # if True:
    #     history.append(
    #         {"role": "assistant", "content":{"path": "todo.md", "alt_text": "To DO Markdown"}}
    #     )
    #     yield "", history
    
    # save the conversation to a json file
    folder = os.path.join("logs", datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".json")
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message, 
        "history": original_history, 
        "model": model, 
        "model_url": model_url, 
        # "api_key": api_key,  # hide the api keys
        "temperature": temp, 
        "max_output_tokens": max_output_tokens,
        "generation": partial_message,
        "post_conv": post_conv
    }
    with open(file_path, "w") as f:
        json.dump(log_data, f, indent=2)

    # 添加数据行
    table.add_row(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model, str(original_history), str(message), partial_message)
    console.print(table)
    # print("history:", history)
    # import ipdb; ipdb.set_trace()
    return "", history

def like(evt: gr.LikeData):
    print("User liked the response")
    print(evt.index, evt.liked, evt.value)

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
                max_output_tokens = gr.Slider(minimum=0, maximum=8196, value=2048, step=128, interactive=True, label="Max output tokens",)
                # gradio checkbox for stream mode or not 
                stream = gr.Checkbox(label="Streaming", value = False)


            gr.Examples(examples=[
                {"files":["data/images/007-dark_forest.png"], "text": "What are the red structures visible in the background?"},
                {"files":["data/images/030-villager.png"], "text": "What is the role of the villager seen in the image?"},
                {"files":["data/images/038-inventory.png"], "text": "What type of armor is the player wearing?"},
                {"files":[], "text": "Give you nothing in the inventory, generate a step-by-step plan to obtain diamonds."},
            ], inputs=multimodaltextbox)

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                height=650, 
                type="messages", 
                show_copy_button=True,
                # additional_inputs=[model, model_url, api_key, temp, max_output_tokens, stream],
                # fill_height=True
                )
            multimodaltextbox.render()
            # gr.ChatInterface(predict, type="messages", chatbot=chatbot, textbox=multimodaltextbox, multimodal=True, additional_inputs=[model, model_url, api_key, temp, max_output_tokens, stream], fill_height=True)
            multimodaltextbox.submit(predict, [multimodaltextbox, chatbot, model, model_url, api_key, temp, max_output_tokens, stream], [multimodaltextbox, chatbot])
            chatbot.like(like)

demo.queue().launch(server_name=args.host, server_port=args.port, share=True)