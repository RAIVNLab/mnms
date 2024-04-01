import os
import ast
import torch
import textract
import requests
from PIL import Image
from .config import *
from transformers import pipeline
from typing import Dict, List, Union

root_path = CACHE_ROOT_PATH
os.environ["TORCH_HOME"] = f"{root_path}/torch"
os.environ["TRANSFORMERS_CACHE"] = f"{root_path}/huggingface"
os.environ["DIFFUSERS_CACHE"] = f"{root_path}/huggingface"
os.environ["HF_HOME"] = f"{root_path}/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{root_path}/huggingface"

def get_full_path_data(filename):
    base_path = DATA_PATH
    return os.path.join(base_path, filename)

def text_processing(file_path):
    # Check the file extension
    if file_path.endswith(".txt"):
        with open(file_path, "r") as file:
            content = file.read()
    elif file_path.endswith(".doc") or file_path.endswith(".docx"):
        # Use textract to extract text from doc and docx files
        content = textract.process(file_path).decode("utf-8")
    else:
        # if the file is not .txt .doc .docx, then it is a string, directly return the stirng
        return file_path
    return content

def object_processing(file_path):
    import pickle

    with open(file_path, "rb") as f:
        objs = pickle.load(f)
    return objs

def image_processing(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    elif isinstance(img, str):
        if os.path.exists(img):
            return Image.open(img).convert("RGB")
        else:
            full_path = get_full_path_data(img)
            if os.path.exists(full_path):
                return Image.open(full_path).convert("RGB")
            else:
                raise FileNotFoundError

def audio_processing(audio_file):
    if os.path.exists(audio_file):
        return audio_file
    else:
        full_path = get_full_path_data(audio_file)
        return full_path

# -------------------------- Tool Functions --------------------------

def text_generation(text: str, ckpt=MODEL_SELECTION["text_generation"]):
    """
    models:
        "gpt-4-1106-preview": gpt-4 turbo, tested
        "gpt-4" : gpt-4, tested
        "gpt-3.5-turbo-1106" : gpt-3.5 turbo, tested
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    text = text_processing(text)

    response = client.chat.completions.create(
        model=ckpt,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{text}"},
                ],
            }
        ],
        max_tokens=300,
    )
    result_formatted = {"text": response.choices[0].message.content}

    return result_formatted


def text_summarization(text: str, ckpt=MODEL_SELECTION["text_summarization"]): 
    # could change it to return its original text if the number of words is less than 130
    text = text_processing(text)
    pipe = pipeline("summarization", model=ckpt)
    # default parameters: max_length=130, min_length=30, do_sample=False
    result_raw = pipe(
        text,
        max_length=130,
        min_length=30,
        do_sample=False,
    ) 
    result_formatted = {"text": result_raw[0]["summary_text"]}

    return result_formatted


def text_classification(text: str, ckpt=MODEL_SELECTION["text_classification"]):
    text = text_processing(text)

    pipe = pipeline("text-classification", model=ckpt)
    result_raw = pipe(text)  # [{'label': 'POSITIVE', 'score': 0.9998867511749268}]
    result_formatted = {"text": result_raw[0]["label"]}

    return result_formatted


def question_answering(
    question: str, text: str, ckpt=MODEL_SELECTION["question_answering"]
):  # alternative: "deepset/roberta-base-squad2"
    question = text_processing(question)
    text = text_processing(text)
    pipe = pipeline("question-answering", model=ckpt)
    result_raw = pipe(
        question=question, context=text
    )  # {'score': 0.01082150824368, 'start': 0, 'end': 10, 'answer': 'My name is'}
    result_formatted = {"text": result_raw["answer"]}

    return result_formatted


def automatic_speech_recognition(
    audio: str, ckpt=MODEL_SELECTION["automatic_speech_recognition"]
):
    audio = audio_processing(audio)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt == "openai/whisper-large-v2":
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import librosa

        # load model and processor
        processor = WhisperProcessor.from_pretrained(ckpt)
        model = WhisperForConditionalGeneration.from_pretrained(ckpt)
        model.forced_decoder_ids = None

        # load dummy dataset and read audio files
        sample = {}
        sample["array"], sample["sampling_rate"] = librosa.load(audio, sr=16_000)
        input_features = processor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ).input_features

        # generate token ids and decoding
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        ) 
        result_formatted = {"text": transcription[0]}

        return result_formatted


def image_generation(text: str, ckpt=MODEL_SELECTION["image_generation"]):
    if ckpt == "stabilityai/stable-diffusion-xl-base-1.0":

        text = text_processing(text)
        if len(text) >= 75:
            from openai import OpenAI
            import io
            client = OpenAI(
                api_key=OPENAI_API_KEY
            )

            response = client.images.generate(
                model="dall-e-3",
                prompt=text,
                size="1024x1024",
                quality="hd",
                n=1,
            )
            result_url = response.data[0].url
            response = requests.get(result_url)

            # download the image from Openai
            if response.status_code == 200:
                image_data = io.BytesIO(response.content)
                result_image = Image.open(image_data)
            result_formatted = {"image": result_image}
            return result_formatted
        else:
            from diffusers import DiffusionPipeline

            generator = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=ckpt,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            generator.to("cuda")

            result = generator(prompt=text).images[0]
            result_formatted = {"image": result}

            return result_formatted


def image_captioning(
    image: Union[str, Image.Image], ckpt=MODEL_SELECTION["image_captioning"]
):  # alternative: nlpconnect/vit-gpt2-image-captioning (testing, blip is better than vit-gpt2)z

    image = image_processing(image)

    pipe = pipeline("image-to-text", model=ckpt)

    result = pipe(
        image
    )  # [{'generated_text': 'there is a small white dog sitting next to a cell phone'}]
    result_formatted = {"text": result[0]["generated_text"]}

    return result_formatted


def image_editing(
    image: Union[str, Image.Image], prompt: str, ckpt=MODEL_SELECTION["image_editing"]
):  # alternative: "timbrooks/instruct-pix2pix"
    if (
        ckpt == "stabilityai/stable-diffusion-xl-refiner-1.0"
    ):  # the result of this model isn't good enough
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline
        from diffusers.utils import load_image

        image = image_processing(image)
        prompt = text_processing(prompt)

        init_image = image

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")

        result = pipe(prompt, image=init_image).images[0]
        result_formatted = {"image": result}

        return result_formatted

    if ckpt == "timbrooks/instruct-pix2pix":
        import torch
        from diffusers import (
            StableDiffusionInstructPix2PixPipeline,
            EulerAncestralDiscreteScheduler,
        )
        from diffusers.utils import load_image

        image = image_processing(image)
        prompt = text_processing(prompt)
        init_image = image

        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        )
        pipe.to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        result = pipe(
            prompt, image=init_image, num_inference_steps=10, image_guidance_scale=1
        ).images[0]
        result_formatted = {"image": result}

        return result_formatted


def image_classification(
    image: Union[str, Image.Image], ckpt=MODEL_SELECTION["image_classification"]
):  # alternative: "microsoft/resnet-50"
    from transformers import ViTImageProcessor, ViTForImageClassification
    from PIL import Image

    image = image_processing(image)

    processor = ViTImageProcessor.from_pretrained(ckpt)
    model = ViTForImageClassification.from_pretrained(ckpt)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    result_formatted = {"text": model.config.id2label[predicted_class_idx]}

    return result_formatted


def visual_question_answering(
    image: Union[str, Image.Image], question, ckpt=MODEL_SELECTION["visual_question_answering"]
):  # alternative: "dandelin/vilt-b32-finetuned-vqa"
    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForQuestionAnswering

    image = image_processing(image)
    question = text_processing(question)

    processor = BlipProcessor.from_pretrained(ckpt)
    model = BlipForQuestionAnswering.from_pretrained(
        ckpt, torch_dtype=torch.float16
    ).to("cuda")

    raw_image = image

    inputs = processor(raw_image, question, return_tensors="pt").to(
        "cuda", torch.float16
    )
    out = model.generate(**inputs)
    result_formatted = {"text": processor.decode(out[0], skip_special_tokens=True)}

    return result_formatted


def object_detection(
    image: Union[str, Image.Image], ckpt=MODEL_SELECTION["object_detection"]
):  # alternative: "facebook/detr-resnet-50" # can not detect cartoon figure, might be due to the model
    from transformers import DetrImageProcessor, DetrForObjectDetection
    import torch
    from PIL import Image

    image = image_processing(image)

    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained(ckpt, revision="no_timm")
    model = DetrForObjectDetection.from_pretrained(ckpt, revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]
    boxes = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        boxes.append({"bbox": box, "label": model.config.id2label[label.item()]})

    results_formatted = {"image": image, "objects": boxes}

    return results_formatted


def image_segmentation(image: Union[str, Image.Image], ckpt=MODEL_SELECTION["image_segmentation"]):
    import torch
    import numpy as np
    from PIL import Image
    import transformers

    img = image_processing(image)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    feature_extractor = transformers.MaskFormerFeatureExtractor.from_pretrained(ckpt)
    model = transformers.MaskFormerForInstanceSegmentation.from_pretrained(ckpt).to(
        device
    )
    model.eval()

    inputs = feature_extractor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    outputs = feature_extractor.post_process_panoptic_segmentation(outputs)[0]
    instance_map = outputs["segmentation"].cpu().numpy()
    objs = []
    for seg in outputs["segments_info"]:
        inst_id = seg["id"]
        label_id = seg["label_id"]
        category = model.config.id2label[label_id]
        mask = (instance_map == inst_id).astype(float)
        resized_mask = np.array(
            Image.fromarray(mask).resize(img.size, resample=Image.BILINEAR)
        )
        Y, X = np.where(resized_mask > 0.5)
        x1, x2 = np.min(X), np.max(X)
        y1, y2 = np.min(Y), np.max(Y)
        num_pixels = np.sum(mask)
        objs.append(
            dict(
                mask=resized_mask,
                label=category,
                bbox=[x1, y1, x2, y2],
                inst_id=inst_id,
            )
        )

    results_formatted = {"image": img, "objects": objs}

    return results_formatted


def optical_character_recognition(
    image: Union[str, Image.Image], ckpt=MODEL_SELECTION["optical_character_recognition"]
):
    import easyocr
    import io

    reader = easyocr.Reader(["en"])  # Load the OCR model into memory

    if isinstance(image, str):
        # If image is a path, use it directly
        image_path_or_bytes = (
            image if os.path.exists(image) else get_full_path_data(image)
        )
    else:
        # If image is an Image object, convert it to a bytes stream
        buffer = io.BytesIO()
        image = image_processing(image)  # Process the image if needed
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        image_path_or_bytes = buffer

    # Read text from the image or image path
    result = reader.readtext(image_path_or_bytes)

    # Extract only the text from the result
    result_text = [text for _, text, _ in result]

    # Format the result
    result_formatted = {"text": ", ".join(result_text)}

    return result_formatted


def image_crop(image: Union[str, Image.Image], object: Dict, ckpt=None):
    img = image_processing(image)
    if 'bbox' in object:
        bbox = object['bbox']
        
        if isinstance(bbox, str):
            try:
                bbox = ast.literal_eval(bbox)
            except:
                bbox = []

        if len(bbox) == 4:
            use_percent = (all(x < 1.0 for x in bbox))
            if use_percent:
                W, H = img.size
                bbox = [bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H]
            out_img = img.crop(bbox)
        else:
            out_img = img
    else:
        out_img = img

    result_formatted = {"image": out_img}
    return result_formatted


def image_crop_left(image: Union[str, Image.Image], ckpt=None):
    img = image_processing(image)

    w, h = img.size
    left_box = [0, 0, int(w / 2), h - 1]

    out_img = img.crop(left_box)

    result_formatted = {"image": out_img}
    return result_formatted


def image_crop_right(image: Union[str, Image.Image], ckpt=None):
    img = image_processing(image)

    w, h = img.size
    right_box = [int(w / 2), 0, w - 1, h - 1]

    out_img = img.crop(right_box)

    result_formatted = {"image": out_img}

    return result_formatted


def image_crop_top(image: Union[str, Image.Image], ckpt=None):
    img = image_processing(image)

    w, h = img.size
    above_box = [0, 0, w - 1, int(h / 2)]

    out_img = img.crop(above_box)

    result_formatted = {"image": out_img}

    return result_formatted


def image_crop_bottom(image: Union[str, Image.Image], ckpt=None):
    img = image_processing(image)

    w, h = img.size
    below_box = [0, int(h / 2), w - 1, h - 1]

    out_img = img.crop(below_box)

    result_formatted = {"image": out_img}

    return result_formatted


def background_blur(image: Union[str, Image.Image], object: Dict, ckpt=None):
    import numpy as np
    import cv2
    from PIL import Image, ImageFilter

    def refine_mask(img, mask):
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask, _, _ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
        return mask.astype(float)

    def smoothen_mask(mask):
        mask = Image.fromarray(255 * mask.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=5)
        )
        return np.array(mask).astype(float) / 255

    obj = object
    img = image_processing(image)

    bgimg = img.copy()
    bgimg = bgimg.filter(ImageFilter.GaussianBlur(radius=2))
    bgimg = np.array(bgimg).astype(float)
    img = np.array(img).astype(float)
    if "mask" in obj:
        # blur only the background of an object if its mask is provided, else blur the whole image
        refined_mask = refine_mask(img, obj["mask"])
        mask = np.tile(refined_mask[:, :, np.newaxis], (1, 1, 3))
        mask = smoothen_mask(mask)
        bgimg = mask * img + (1 - mask) * bgimg

    bgimg = np.array(bgimg).astype(np.uint8)
    bgimg = Image.fromarray(bgimg)

    result_formatted = {"image": bgimg}
    return result_formatted


def color_pop(image: Union[str, Image.Image], object: Dict, ckpt=None):
    import numpy as np
    import cv2
    from PIL import Image, ImageFilter

    def refine_mask(img, mask):
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask, _, _ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
        return mask.astype(float)

    obj = object
    img = image_processing(image)

    gimg = img.copy()
    gimg = gimg.convert("L").convert("RGB")
    gimg = np.array(gimg).astype(float)
    img = np.array(img).astype(float)
    if "mask" in obj:
        # make a color pop if an object mask is provided, else return a black and white image
        refined_mask = refine_mask(img, obj["mask"])
        mask = np.tile(refined_mask[:, :, np.newaxis], (1, 1, 3))
        gimg = mask * img + (1 - mask) * gimg

    gimg = np.array(gimg).astype(np.uint8)
    gimg = Image.fromarray(gimg)

    result_formatted = {"image": gimg}
    return result_formatted


def count(objects: List[Dict], ckpt=None):
    objs = objects
    result_formatted = {"number": len(objs)}

    return result_formatted


def tag(image: Union[str, Image.Image], objects: List[Dict], ckpt=None):
    from PIL import Image, ImageDraw, ImageFont

    objs = objects
    img = image_processing(image)

    W, H = img.size
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for i, obj in enumerate(objs):
        box = obj["bbox"]
        draw.rectangle(box, outline="green", width=4)
        x1, y1, x2, y2 = box
        label = obj["label"]

        font = ImageFont.load_default()
        font_box = font.getbbox(label)
        text_width = font_box[2] - font_box[0]
        text_height = font_box[3] - font_box[1]

        if x1 + text_width > W:
            x1 = x1 - text_width
        if y1 + text_height > H:
            y1 = y1 - text_height

        draw.rectangle((x1, y1 - text_height, x1 + text_width, y1), fill="green")
        draw.text((x1, y1 - text_height), label, fill="white", font=font)

    result_formatted = {"image": img1}

    return result_formatted


def emoji(image: Union[str, Image.Image], object: Dict, emoji: str, ckpt=None):
    import os
    from PIL import Image
    import augly.image as imaugs
    from augly.utils.base_paths import EMOJI_DIR

    def add_emoji(objs, emoji_name, img):
        W, H = img.size
        emojipth = os.path.join(EMOJI_DIR, f"smileys/{emoji_name}.png")

        for obj in objs:
            if "bbox" in obj:
                x1, y1, x2, y2 = obj["bbox"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                s = (y2 - y1) / 1.5
                x_pos = (cx - 0.5 * s) / W
                y_pos = (cy - 0.5 * s) / H
                emoji_size = s / H
                emoji_aug = imaugs.OverlayEmoji(
                    emoji_path=emojipth, emoji_size=emoji_size, x_pos=x_pos, y_pos=y_pos
                )
                img = emoji_aug(img)

        return img

    img = image_processing(image)
    objs = [object]
    emoji_name = emoji

    emojipth = os.path.join(EMOJI_DIR, f"smileys/{emoji_name}.png")
    emoji_name = emoji_name if os.path.exists(emojipth) else "smiling_face"

    img = add_emoji(objs, emoji_name, img)

    result_formatted = {"image": img}
    return result_formatted


def select_object(objects: List[Dict], object_name: str, ckpt=None):
    objs = objects
    selected_obj = {}
    for obj in objs:
        if obj['label'].find(object_name) > -1:
            selected_obj = obj
            break
        else:
            label_words = obj['label'].split()
            obj_words = object_name.split()
            common_words = set(label_words).intersection(set(obj_words))
            if len(common_words) > 0:
                selected_obj = obj
                break
            
    result_formatted = {"object": selected_obj}

    return result_formatted


def get_date_fact(date: str, ckpt=None):
    from dateutil import parser

    date_str = date
    dt = parser.parse(date_str)

    formatted_date = dt.strftime("%m/%d")

    url = f"https://numbersapi.p.rapidapi.com/{formatted_date}/date"
    params = {"fragment": "true", "json": "true"}
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "numbersapi.p.rapidapi.com",
    }
    response = requests.get(url, headers=headers, params=params)
    result_formatted = response.json()

    return result_formatted


def get_year_fact(year: Union[str, int], ckpt=None):
    url = f"https://numbersapi.p.rapidapi.com/{year}/year"
    params = {"fragment": "true", "json": "true"}
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "numbersapi.p.rapidapi.com",
    }
    response = requests.get(url, headers=headers, params=params)
    result_formatted = response.json()

    return result_formatted


def get_math_fact(number: Union[str, int], ckpt=None):
    url = f"https://numbersapi.p.rapidapi.com/{number}/math"
    params = {"fragment": "true", "json": "true"}
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "numbersapi.p.rapidapi.com",
    }
    response = requests.get(url, headers=headers, params=params)
    result_formatted = response.json()

    return result_formatted


def get_trivia_fact(number: Union[str, int], ckpt=None):
    url = f"https://numbersapi.p.rapidapi.com/{number}/trivia"
    params = {"fragment": "true", "json": "true"}
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "numbersapi.p.rapidapi.com",
    }
    response = requests.get(url, headers=headers, params=params)
    result_formatted = response.json()

    return result_formatted


def love_calculator(first_name: str, second_name: str, ckpt=None):
    url = "https://love-calculator.p.rapidapi.com/getPercentage"
    params = {"fname": first_name, "sname": second_name}
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "love-calculator.p.rapidapi.com",
    }
    response = requests.get(url, headers=headers, params=params)
    result_formatted = response.json()
    result_formatted = {
        "number": result_formatted["percentage"],
        "message": result_formatted["result"],
    }

    return result_formatted


def get_location(city: str, ckpt=None):
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        'User-Agent': 'm&ms'
    }
    params = {"q": city, "format": "json"}
    response = requests.get(url, params=params, headers=headers)
    result_formatted = response.json()[0]
    return result_formatted


def search_movie(movie_title: str, movie_year: Union[str, int], ckpt=None):
    url = "http://www.omdbapi.com/"
    params = {
        "t": movie_title,
        "y": movie_year,
        "plot": "short",
        "r": "json",
        "apikey": OMDB_API_KEY,
    }
    response = requests.get(url, params=params)
    result_formatted = response.json()

    selected_keys = ["Title", "Year", "Genre", "Director", "Plot"]
    desc = ""
    if len(result_formatted) > 0:
        for k, v in result_formatted.items():
            if k in selected_keys and len(v) > 0:
                desc += f"{k}: {v}\n"
    else:
        desc = "Movie not found!"
    result_formatted = {"text": desc}

    return result_formatted


def get_weather(lon: Union[str, float], lat: Union[str, float], ckpt=None):
    url = "http://www.7timer.info/bin/api.pl"
    params = {"lon": lon, "lat": lat, "product": "civil", "output": "json"}
    response = requests.get(url, params=params)
    result_formatted = {"objects": response.json()["dataseries"]}

    return result_formatted


def wikipedia_simple_search(text: str, ckpt=None):
    max_len = 300
    query = text[:max_len]
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": "1",
        "srlimit": "1",
    }
    response = requests.get(url, params=params)
    try:
        text = response.json()["query"]["search"][0]["snippet"]
        text = text.replace('<span class="searchmatch">', "")
        text = text.replace("</span>", "")
    except:
        text = "" 
    result_formatted = {"text": text}

    return result_formatted