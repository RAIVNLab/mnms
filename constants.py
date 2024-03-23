TOOL_METADATA = {
  "text generation": {
      "input": {
          "modality": ["text"],
          "desc": ["a text string (usually in English) to be generated from"],
          "arg_name": ["text"],
      }, 
      "output": {
          "modality": ["text"],
          "desc": ["the continuated text string"],
          "arg_name": ["text"],
      },
      "description": "It takes an input text prompt and outputs a text that is most likely to follow the input text.",
      "next_tools": ["text translation", "text summarization", "text classification", "question answering", "text to audio", "image generation",  "get today news"], # "wikipedia simple search",
      "data_file": "txt_gen_samples.csv",
      "ann_key": "text",
      "category": "ml model"
  },
  "text summarization": { 
      "input": {
          "modality": ["text"],
          "desc": ["a text string (usually in English) to be summarized"],
          "arg_name": ["text"],
      }, 
      "output": {
          "modality": ["text"],
          "desc": ["a text string after summarization"],
          "arg_name": ["text"],
      },
      "description": "it takes a paragraph of text and summarizes into a few sentences.",
      "next_tools": ["text generation", "text translation", "text classification", "question answering", "text to audio", "wikipedia simple search", "get today news", "image generation"], 
      "data_file": "",
      "ann_key": "summary",
      "category": "ml model"
  },
  "text classification": { 
      "input": {
          "modality": ["text"],
          "desc": ["a text string (usually in English) to be classified"],
          "arg_name": ["text"],
      }, 
      "output": {
          "modality": ["text"],
          "desc": ["the text label for the class (model specific)"],
          "arg_name": ["text"],
      },
      "description": "It takes a text and classifies it into a category in the model's vocaburary (e.g. positive or negative based on its sentiment).",
      "next_tools": ["text translation", "text to audio",  "get today news"], # text summarization, question answering, "image generation", "text generation",
      "data_file": "sst2_samples.csv",
      "ann_key": "label",
      "category": "ml model"
  },
  "question answering": { 
      "input": {
          "modality": ["text", "text"],
          "desc": ["a context text string such as a paragraph", "a question about the context"],
          "arg_name": ["text", "question"],
      },  
      "output": {
          "modality": ["text"],
          "desc": ["a string that's the answer within the context text"],
          "arg_name": ["text"],
      }, 
      "description": "It takes a text and a question, and outputs an answer to that question based on the text.",
      "next_tools": ["text generation", "text translation", "text to audio", "text classification", "image generation", "wikipedia simple search", "get today news"], # "text summarization",
      "data_file": "squad_samples.csv",
      "ann_key": "answers",
      "category": "ml model"
  },
  "image generation": { 
      "input": {
          "modality": ["text"],
          "desc": ["a text string to generate an image from"],
          "arg_name": ["text"],
      },  
      "output": {
          "modality": ["image"],
          "desc": ["the generated image"],
          "arg_name": ["image"],
      },
      "description": "It takes a text prompt and generates an image that matches the text description.",
      "next_tools": [ "image editing", "image captioning", "object detection","image classification", "image segmentation", "image crop left", "image crop right", "image crop top", "image crop bottom", "get today news", "optical character recognition"], 
      "data_file": "vg_coco_gqa_val.csv",
      "ann_key": "imageId",
      "category": "ml model"
  },
  "image captioning": { 
      "input": {
          "modality": ["image"],
          "desc": ["an image to be captioned"],
          "arg_name": ["image"],
      },  
      "output": {
          "modality": ["text"],
          "desc": ["the generated caption"],
          "arg_name": ["text"],
      },
      "description": "It takes an image and generates a text caption of the image.",
      "next_tools": ["text generation", "text translation", "text summarization", "text classification", "question answering", "text to audio", "get today news",  "image generation"], # "wikipedia simple search",
      "data_file": "vg_coco_gqa_val.csv",
      "ann_key": "captions",
      "category": "ml model"
    },
  "optical character recognition": { 
      "input": {
          "modality": ["image"],
          "desc": ["an image to be recognized"],
          "arg_name": ["image"],
      },  
      "output": {
          "modality": ["text"],
          "desc": ["the text in the image"],
          "arg_name": ["text"],
      },
      "description": "It takes an image and outputs recognized texts in the image.",
      "next_tools": ["text generation", "text translation", "text classification", "text to audio", "image generation", "wikipedia simple search", "get today news"],  # "text summarization", "question answering",
      "data_file": "ocr_samples.csv",
      "ann_key": "anns",
      "category": "ml model"
    },
  "image classification": { 
      "input": {
          "modality": ["image"],
          "desc": ["an image to be classified"],
          "arg_name": ["image"],
      },  
      "output": {
          "modality": ["text"],
          "desc": ["the text label for the class (model specific)"], 
          "arg_name": ["text"],
      },
      "description": "It takes an image and classifies the subject in the image into a category such as cat or dog.",
      "next_tools": ["text generation", "text translation", "text to audio", "image generation", "wikipedia simple search", "get today news"], # "text summarization", "text classification", "question answering"
      "data_file": "img_cls_samples.csv",
      "ann_key": "label",
      "category": "ml model"
    },
  "image editing": {
      "input": {
          "modality": ["image", "text"],
          "desc": ["the initial image to be edited", "a text instruction specifying how the image should be edited"],
          "arg_name": ["image", "prompt"],
      },  
      "output": {
          "modality": ["image"],
          "desc": ["the edited image"],
          "arg_name": ["image"],
      },
      "description": "It takes an image and a text prompt and outputs a new image based on the text.",
      "next_tools": ["image captioning", "object detection", "image classification", "optical character recognition", "image segmentation", "image crop left", "image crop right", "image crop top", "image crop bottom", "get today news"],
      "data_file": "img_edit_samples.csv",
      "ann_key": "output",
      "category": "ml model"
    },
  "object detection": { 
      "input": {
          "modality": ["image"],
          "desc": ["an image in which to detect objects"],
          "arg_name": ["image"],
      },  
      "output": {
          "modality": ["image", "list[dict]"], 
          "desc": ["the original image", "a list of objects where each object is a dictionary that contains the keys 'bbox' and 'label'."],
          "arg_name": ["image", "objects"] 
      },
      "description": "It takes an image and outputs rectangular bounding boxes of objects detected in the image.",
      "next_tools": ["count", "tag", "get today news", "select object"], # "image editing", "image captioning", "emoji"
      "data_file": "vg_coco_gqa_val.csv",
      "ann_key": "category_name",
      "category": "ml model"
    },
  "image segmentation": { 
      "input": {
          "modality": ["image"],
          "desc": ["an image to be segmented into masks"],
          "arg_name": ["image"],
      },  
      "output": {
          "modality": ["image", "list[dict]"],
          "desc": ["the original image", "a list of objects where each object is a dictionary that contains the keys 'mask' and 'label'."], 
          "arg_name": ["image", "objects"]
      },
      "description": "It takes an image, segments it into different parts, and outputs segmentation masks of any shape for the parts.",
      "next_tools": ["count", "select object", "get today news"], # "background blur", "color pop", "emoji", "image editing", "image captioning",
      "data_file": "vg_coco_gqa_val.csv",
      "ann_key": "category_name",
      "category": "ml model"
    },
  "automatic speech recognition": { 
      "input": {
          "modality": ["audio"],
          "desc": ["an audio file"],
          "arg_name": ["audio"],
      },  
      "output": {
          "modality": ["text"],
          "desc": ["the text that was recognized from the audio of a speech"],
          "arg_name": ["text"],
      },
      "description": "It takes an audio file and produces a transcription of the audio.",
      "next_tools": ["text generation", "text translation", "text classification", "text summarization", "question answering", "image generation", "get today news"], # text to audio, "wikipedia simple search"
      "data_file": "asr_samples.csv",
      "ann_key": "audioText",
      "category": "ml model"
    },
  "visual question answering": { 
      "input": {
          "modality": ["image", "text"],
          "desc": ["an image", "a qeustion about the image"],
          "arg_name": ["image", "question"],
      }, 
      "output": {
          "modality": ["text"],
          "desc": ["a string that's the answer to a question about the image"],
          "arg_name": ["text"],
      },
      "description": "It takes an image and a question about the image, and generates an answer to the question.",
      "next_tools": ["text generation", "text translation",  "text to audio", "wikipedia simple search", "get today news", "image generation"], # "text classification", "question answering", "text summarization",
      "data_file": "vg_coco_gqa_val.csv",
      "ann_key": "answer",
      "category": "ml model"
    },
    "image crop": { 
      "input": {
          "modality": ["image", "object"],
          "desc": ["an image to be cropped", "the bounding box coordinates of an object to be cropped"],
          "arg_name": ["image", "object"],
      }, 
      "output": {
          "modality": ["image"],
          "desc": ["a cropped image"],
          "arg_name": ["image"],
      },
      "description": "It takes an image and 4 numbers representing the coordinates of a bounding box and crops the image to the region within the box.",
      "next_tools": ["image captioning", "optical character recognition", "image classification", "object detection", "image segmentation", "image crop left", "image crop right", "image crop top", "image crop bottom", "get today news"],
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "image crop left": { 
      "input": {
          "modality": ["image"],
          "desc": ["an image whose left part is to be cropped"],
          "arg_name": ["image"],
      }, 
      "output": {
          "modality": ["image"],
          "desc": ["a cropped image"],
          "arg_name": ["image"],
      },
      "description": "It takes an image, crops and keeps the left part of the image.",
      "next_tools": ["image captioning", "optical character recognition", "image classification", "object detection", "image segmentation", "get today news"], # "image crop right", "image crop top", "image crop bottom",
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "image crop right": { 
      "input": {
          "modality": ["image"],
          "desc": ["an image whose right part is to be cropped"],
          "arg_name": ["image"],
      }, 
      "output": {
          "modality": ["image"],
          "desc": ["a cropped image"],
          "arg_name": ["image"],
      },
      "description": "It takes an image, crops and keeps the right part of the image.",
      "next_tools": ["image captioning", "optical character recognition", "image classification", "object detection", "image segmentation", "get today news"], # "image crop left", "image crop top", "image crop bottom",
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "image crop top": { 
      "input": {
          "modality": ["image"],
          "desc": ["an image whose top part is to be cropped"],
          "arg_name": ["image"],
      }, 
      "output": {
          "modality": ["image"],
          "desc": ["a cropped image"],
          "arg_name": ["image"],
      },
      "description": "It takes an image, crops and keeps the top part of the image.",
      "next_tools": ["image captioning", "optical character recognition", "image classification", "object detection", "image segmentation", "get today news"], # "image crop left", "image crop right", "image crop bottom",
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "image crop bottom": { 
      "input": {
          "modality": ["image"],
          "desc": ["an image whose bottom part is to be cropped"],
          "arg_name": ["image"],
      }, 
      "output": {
          "modality": ["image"],
          "desc": ["a cropped image"],
          "arg_name": ["image"],
      },
      "description": "It takes an image, crops and keeps the bottom part of the image.",
      "next_tools": ["image captioning", "optical character recognition", "image classification", "object detection", "image segmentation", "get today news"], # "image crop left", "image crop right", "image crop top",
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "background blur": {
        "input": {
          "modality": ["image", "object"],
          "desc": ["the image where the background is to be blurred", "the object in the foreground"],
          "arg_name": ["image", "object"],
      }, 
      "output": {
          "modality": ["image"],
          "desc": ["a blurred image"],
          "arg_name": ["image"],
      },
      "description": "It takes an image and one or multiple objects in the foreground, and returns an image where the backgroud is blurred.",
      "next_tools": ["image captioning", "optical character recognition", "image classification", "object detection", "image segmentation", "image crop left", "image crop right", "image crop top", "image crop bottom", "get today news"],
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "color pop": {
        "input": {
          "modality": ["image", "object"],
          "desc": ["the image of which to create a color pop", "the object to be colored"],
          "arg_name": ["image", "object"],
      }, 
      "output": {
          "modality": ["image"],
          "desc": ["a color pop of the original image"],
          "arg_name": ["image"],
      },
      "description": "It takes an image and one or multiple objects, and returns an image where only the object is colored and the rest is black and white.",
      "next_tools": ["image captioning", "optical character recognition", "image classification", "object detection", "image segmentation", "image crop left", "image crop right", "image crop top", "image crop bottom", "get today news"],
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "count": {
        "input": {
          "modality": ["list[dict]"],
          "desc": ["A list of objects to be counted"],
          "arg_name": ["objects"],
      },
      "output": {
          "modality": ["integer"],
          "desc": ["The total count of the input objects"],
          "arg_name": ["number"],
      },
      "description": "It takes a list of objects and returns the count of the objects.",
      "next_tools": ["get math fact", "get trivia fact", "get today news"],
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "tag": {
        "input": {
          "modality": ["image", "list[dict]"],
          "desc": ["an image to be tagged", "a list of objects, where each object is a dictionary with keys 'bbox' and 'label'"],
          "arg_name": ["image", "objects"],
      }, 
      "output": {
          "modality": ["image"],
          "desc": ["an image where objects are tagged"],
          "arg_name": ["image"],
      },
      "description": "It takes an image and a list of objects with their bounding boxes and classes, and tags all the objects",
      "next_tools": ["optical character recognition"], #"image crop bottom", "image crop right", "image crop top", "image crop left"
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "select object": {
        "input": {
          "modality": ["list[dict]", "text"],
          "desc": ["a list of objects", "the object to be selected and returned"],
          "arg_name": ["objects", "object_name"],
      }, 
      "output": {
          "modality": ["object"],
          "desc": ["a dict which contains the 'box' coordinates of the object and the 'label' of the object"],
          "arg_name": ["object"],
      },
      "description": "It takes a list of objects, and selects the object based on the input object name.",
      "next_tools": ["get today news", "background blur", "color pop", "emoji", "image crop"], 
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "emoji": {
        "input": {
          "modality": ["image", "object", "text"],
          "desc": ["an image to be editied", "a list of coordinates representing the bounding box of an object", "a text string representing an emoji"],
          "arg_name": ["image", "object", "emoji"],
      }, 
      "output": {
          "modality": ["image"],
          "desc": ["an image where the selected object is replaced by the specified emoji"],
          "arg_name": ["image"],
      },
      "description": "It takes an image and the bounding box coordinates of one or multiple objects, and replaces the object with an emoji (e.g. angry/flushed/crying/dizzy/sleepy/grimacing/kissing/smiling_face, alien, ghost, goblin etc).",
      "next_tools": ["image captioning", "optical character recognition", "image classification", "object detection", "image segmentation", "image crop left", "image crop right", "image crop top", "image crop bottom", "get today news"],
      "data_file": "",
      "ann_key": "",
      "category": "data processing"
    },
    "get date fact": {
        "description": 'It provides interesting facts about dates.',
        "input": {
          "modality": ["text"],
          "desc": ["a string representing a date in the format of month/date e.g. 2/7"],
          "arg_name": ["date"],
      }, 
      "output": {
          "modality": ["text"],
          "desc": ["a fact about the date"],
          "arg_name": ["text"],
      },
      "next_tools": ["text generation", "text summarization", "text classification", "image generation", "get today news"], # "wikipedia simple search",
      "data_file": "api_samples.csv",
      "ann_key": "",
      "category": "public api"
    },
    "get year fact": {
        "description": 'It provides interesting facts about years.',
        "input": {
          "modality": ["integer"],
          "desc": ["an integer representing a year"],
          "arg_name": ["year"],
      }, 
      "output": {
          "modality": ["text"],
          "desc": ["a fact about the year"],
          "arg_name": ["text"],
      },
      "next_tools": ["text generation", "text summarization", "text classification", "image generation", "get today news"], # "wikipedia simple search", 
      "data_file": "api_samples.csv",
      "ann_key": "",
      "category": "public api"
    },
    "get math fact": {
        "description": 'It provides interesting math facts about numbers.',
        "input": {
          "modality": ["integer"],
          "desc": ["an integer number"],
          "arg_name": ["number"],
      }, 
      "output": {
          "modality": ["text"],
          "desc": ["a fact about the number"],
          "arg_name": ["text"],
      },
      "next_tools": ["text generation", "text summarization", "text classification", "image generation", "get today news"], # "wikipedia simple search",
      "data_file": "api_samples.csv",
      "ann_key": "",
      "category": "public api"
    },
    "get trivia fact": {
        "description": 'It provides interesting trivia facts about number.',
        "input": {
          "modality": ["integer"],
          "desc": ["an integer of which to get a random trivia fact"],
          "arg_name": ["number"],
      }, 
      "output": {
          "modality": ["text"],
          "desc": ["a random trivia fact about the number"],
          "arg_name": ["text"],
      },
      "next_tools": ["text generation", "text summarization", "text classification", "image generation",  "get today news"], # "wikipedia simple search",
      "data_file": "api_samples.csv",
      "ann_key": "",
      "category": "public api"
    },
    "love calculator": {
        "description": "Enter your name and the name of your partner/lover/crush to find Love compatibility & chances of successful love relationship.",
        "input": {
          "modality": ["text", "text"],
          "desc": ["a string representing a name", "a string representing a second name"],
          "arg_name": ["first_name", "second_name"],
      }, 
      "output": {
          "modality": ["integer"],
          "desc": ["a number representing the percentage of love compatibility & chances of successful love relationship."],
          "arg_name": ["number"],
      },
      "next_tools": ["get math fact", "get trivia fact", "get today news"],
      "data_file": "api_samples.csv",
      "ann_key": "",
      "category": "public api"
    },
    "get location": {
        "description": "Convert a city name or address to geographical coordinates using OpenStreetMap's Nominatim API.",
        "input": {
          "modality": ["text"],
          "desc": ["a string representing a city's name or address"],
          "arg_name": ["city"],
      }, 
      "output": {
          "modality": ["text", "text"],
          "desc": ["the longitude of the city.", "the latitude of the city"],
          "arg_name": ["lon", "lat"],
      },
      "next_tools": ["get weather", "get today news"], # "wikipedia simple search"
      "data_file": "api_samples.csv",
      "ann_key": "",
      "category": "public api"
    },
    "search movie": {
        "description": "Retrieve basic movie information, including title, year, genre, and director.",
        "input": {
          "modality": ["text", "text"],
          "desc": ["the title of the movie", "the year of the movie"],
          "arg_name": ["movie_title", "movie_year"],
      }, 
      "output": {
          "modality": ["text"], 
          "desc": ["A detailed description of this movie."], 
          "arg_name": ["text"],
      },
      "next_tools": ["text generation", "text summarization", "text classification", "question answering", "image generation", "wikipedia simple search", "get today news"],
      "data_file": "api_samples.csv",
      "ann_key": "",
      "category": "public api"
    },
    "get weather": {
        "description": "Provides weather forecast data based on specific geographical coordinates.",
        "input": {
          "modality": ["text", "text"],
          "desc": ["the longitude of the city.", "the latitude of the city"],
          "arg_name": ["lon", "lat"],
      }, 
      "output": {
          "modality": ["list[dict]"],
          "desc": ["a list of dictionary with weather details"],
          "arg_name": ["objects"],
      },
      "next_tools": ["count", "get today news"],
      "data_file": "api_samples.csv",
      "ann_key": "",
      "category": "public api"
    },
    "wikipedia simple search": {
        "description": "Perform a basic search query on Wikipedia to retrieve a summary of the most relevant page.",
        "input": {
          "modality": ["text"],
          "desc": ["the query to search answers for"],
          "arg_name": ["text"],
      },
      "output": {
          "modality": ["text"],
          "desc": ["a brief summary of the wikipedia page most relevant to the search query"],
          "arg_name": ["text"],
      },
      "next_tools": ["get today news", "text generation", "text summarization", "text classification"], # "image generation"
      "data_file": "api_samples.csv",
      "ann_key": "",
      "category": "public api"
    },
}