import base64 
import requests
from pathlib import Path
import re
import uuid
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFilter, Image
import math

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

def extract_qwen_object_and_box(text):
    import re
    # 定义模式
    object_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
    points_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"

    # 示例文本
    # text = "<|object_ref_start|>birch tree<|object_ref_end|><|box_start|>(530,351),(602,516)<|box_end|><|box_start|>(176,106),(232,160)<|box_end|>"

    # 提取对象
    object_match = re.search(object_pattern, text)
    object_name = object_match.group(1) if object_match else None

    # 提取所有的 box
    points_matches = re.findall(points_pattern, text)
    all_points = []
    for points_str in points_matches:
        # 清理字符串并将点转换为整数元组
        points = [tuple(map(int, p.replace('(', '').replace(')', '').split(','))) for p in points_str.split('),(')]
        all_points += (points)

    # print("Object Name:", object_name)
    # print("All Boxes:", all_points)
    return object_name, all_points

def extract_qwen_object_and_points(text):
    # Regex patterns for object and points
    object_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
    points_pattern = r"<\|point_start\|>(.*?)<\|point_end\|>"
    
    # Extract the object
    object_match = re.search(object_pattern, text)
    object_name = object_match.group(1) if object_match else None
    
    # Extract the points
    points_match = re.search(points_pattern, text)
    points = []
    if points_match:
        points_str = points_match.group(1)
        # Clean the string to remove parentheses and split points
        points = [tuple(map(int, p.replace('(', '').replace(')', '').split(','))) for p in points_str.split('),(')]
    return object_name, points

def denormarlize_qwen_points(image, points):
    # image_width, image_height = image.size
    # extract image size from image
    # print(image.shape)
    # 向上取整
    image_width, image_height = image.shape[1], image.shape[0]
    points = [(math.ceil((x) * image_width / 999.0), math.ceil((y) * image_height / 999.0)) for x, y in points]
    return points

def show_point(model, history):
    if 'molmo' in model:
        ...
    elif 'qwen' in model:
        message = history[-1]["content"]
        obj_name, points = extract_qwen_object_and_points(message)
        print(f"obj_name: {obj_name}, raw points: {points}")
        if obj_name is None or len(points) == 0:
            return None, None, None
        else:
            image = None
            for turn in history[::-1]:
            # user_message, assistant_message = turn
                # 合并连续的用户消息
                if turn['role'] == 'user':
                    user_message = turn['content'] 
                    if Path(user_message[0]).is_file():
                        # image_base64 = encode_image_base64(user_message[0])
                        # conv_buffer.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]})
                        image = plt.imread(user_message[0])
                        break
            # import ipdb; ipdb.set_trace()
            if image is None:
                print("No image found in history")
                return None, None, None
            show_points = denormarlize_qwen_points(image, points)
            print(f"show_points: {show_points}")

            plt.figure()
            plt.imshow(image)
            plt.imshow(np.full_like(image, 0, dtype=np.uint8), alpha=0.5, cmap='gray')  # 半透明灰色覆盖
            # plt.scatter(show_points[:, 0], show_points[:, 1], c='red', marker='o')
            for point in show_points:
                # plt.scatter(point[0], point[1], c='red', marker='o', s=20)
                plt.scatter(point[0], point[1], c='red', marker='o', s=40, edgecolors='black', linewidths=1)
            # plt.title(f"Object: {obj_name}")
            plt.axis('off')
            # plt.show()
            image_path = f"output/images/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{obj_name}-{str(uuid.uuid4())[:8]}.png"
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            return obj_name, show_points, image_path
    else:
        # raise ValueError("Unsupported model")
        print("Unsupported model")
        return None, None, None

def show_box(model, history):
    if 'molmo' in model:
        ...
    elif 'qwen' in model:
        message = history[-1]["content"]
        obj_name, points = extract_qwen_object_and_box(message)
        print(f"obj_name: {obj_name}, raw box: {points}")
        if obj_name is None or len(points) % 2 == 1 or len(points) == 0:
            return None, None, None
        else:
            image = None
            for turn in history[::-1]:
            # user_message, assistant_message = turn
                # 合并连续的用户消息
                if turn['role'] == 'user':
                    user_message = turn['content'] 
                    if Path(user_message[0]).is_file():
                        # image_base64 = encode_image_base64(user_message[0])
                        # conv_buffer.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]})
                        image = plt.imread(user_message[0])
                        break
            # import ipdb; ipdb.set_trace()
            if image is None:
                print("No image found in history")
                return None, None, None
            show_points = denormarlize_qwen_points(image, points)
            # print(f"show boxes: ", show_points)
            show_boxes = []
            for i in range(len(show_points)//2):
                x_min, y_min, width, height = show_points[i*2][0], show_points[i*2][1], show_points[i*2+1][0]-show_points[i*2][0], show_points[i*2+1][1]-show_points[i*2][1]
                show_boxes.append((x_min, y_min, width, height))
            print(f"show_boxes: {show_boxes}")

            #
            # pil_image = Image.fromarray(image.astype(np.uint8))
            # shadow_image = pil_image.filter(ImageFilter.GaussianBlur(radius=10))  # Add shadow effect
            # # Blend the original and shadowed image to simulate a background shadow effect
            # blend_ratio = 0.1
            # blended_image = Image.blend(pil_image, shadow_image, alpha=blend_ratio)
            # image = blended_image

            plt.figure()
            plt.imshow(image)
            plt.imshow(np.full_like(image, 0, dtype=np.uint8), alpha=0.5, cmap='gray')  # 半透明灰色覆盖
            # plt.scatter(show_points[:, 0], show_points[:, 1], c='red', marker='o')
            # for point in show_points:
            #     # plt.scatter(point[0], point[1], c='red', marker='o', s=20)
            #     plt.scatter(point[0], point[1], c='red', marker='o', s=40, edgecolors='black', linewidths=1)
            # plt.title(f"Object: {obj_name}")
            # 绘制框
            for box in show_boxes:
                x_min, y_min, width, height = box
                rect = plt.Rectangle(
                    (x_min, y_min),  # 左上角
                    width,  # 宽
                    height,  # 高
                    linewidth=2,
                    edgecolor="blue",  # 边框颜色
                    facecolor="none",  # 无填充
                )
                plt.gca().add_patch(rect)  # 添加到当前的绘图区域
            plt.axis('off')
            # plt.show()
            image_path = f"output/images/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{obj_name}-{str(uuid.uuid4())[:8]}.png"
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            return obj_name, show_boxes, image_path
    else:
        # raise ValueError("Unsupported model")
        print("Unsupported model")
        return None, None, None
