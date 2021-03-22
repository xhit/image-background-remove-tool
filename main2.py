#!/usr/bin/python3
"""
Name: Background removal tool.
Description: This file contains the CLI interface.
Version: [release][3.3]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
License:
   Copyright 2020 OPHoperHPO

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
# Built-in libraries
import gc
import argparse
import logging
from pathlib import Path


# 3rd party libraries
import tqdm
import numpy as np
from PIL import Image

# Libraries of this project
from libs.strings import *
import libs.networks as networks
import libs.preprocessing as preprocessing
import libs.postprocessing as postprocessing

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# crop image default from http_api.py
def autocrop_image(image):
    image = image.crop(image.getbbox())
    # in a future, margin would be parametrized
    margin = 0
    return add_margin(image, int(image.size[1] * margin / 100),
                            int(image.size[0] * margin / 100),
                            int(image.size[1] * margin / 100),
                            int(image.size[0] * margin / 100), (0, 0, 0, 0))

# crop image 2, is not used, but left here
# def autocrop_image2(image):
#     image_data = np.asarray(image)
#     image_data_bw = image_data.max(axis=2)
#     non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
#     non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
#     cropBox = (min(non_empty_rows), max(non_empty_rows),
#                min(non_empty_columns), max(non_empty_columns))

#     image_data_new = image_data[cropBox[0]:cropBox[
#         1] + 1, cropBox[2]:cropBox[3] + 1, :]

#     new_image = Image.fromarray(image_data_new)
#     return new_image

def add_margin(pil_img, top, right, bottom, left, color):
    """
    Adds fields to the image.
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def __save_image_file__(img, file_path: Path, output_path: Path, crop_image, image_size):
    """
    Saves the PIL image to a file
    :param img: PIL image
    :param file_path: File path object
    :param output_path: Output path object
    """

    # crop image
    if crop_image:
        img = autocrop_image(img)
    
    # size of image
    if image_size == "preview" or image_size == "small" or image_size == "regular":
        img.thumbnail((625, 400), resample=Image.ANTIALIAS)  # 0.25 mp
    elif image_size == "medium":
        img.thumbnail((1504, 1000), resample=Image.ANTIALIAS)  # 1.5 mp
    elif image_size == "hd":
        img.thumbnail((2000, 2000), resample=Image.ANTIALIAS)  # 2.5 mp
    elif image_size != "original":
        raise ValueError("Invalid size value.")
        
    if output_path.exists():
        if output_path.is_file():
            img.save(output_path.with_suffix(".png"))
        elif output_path.is_dir():
            img.save(output_path.joinpath(file_path.stem + ".png"))
        else:
            raise ValueError("Something wrong with output path!")
    else:
        if output_path.suffix == '':
            if not output_path.exists():  # create output directory if it doesn't exist
                output_path.mkdir(parents=True, exist_ok=True)
            img.save(output_path.joinpath(file_path.stem + ".png"))
        else:
            if not output_path.parents[0].exists():  # create output directory if it doesn't exist
                output_path.parents[0].mkdir(parents=True, exist_ok=True)
            img.save(output_path.with_suffix(".png"))


def process(input_path, output_path, model_name=MODELS_NAMES[0],
            preprocessing_method_name=PREPROCESS_METHODS[0],
            postprocessing_method_name=POSTPROCESS_METHODS[0], recursive=False, crop_image=False, image_size=""):
    """
    Processes the file.
    :param input_path: The path to the image / folder with the images to be processed.
    :param output_path: The path to the save location.
    :param model_name: Model to use.
    :param postprocessing_method_name: Method for image preprocessing
    :param preprocessing_method_name: Method for image post-processing
    :param recursive: Recursive image search in folder
    :param crop: Return image cropped
    """
    if input_path is None or output_path is None:
        raise ValueError("Bad parameters! Please specify input path and output path.")

    model = networks.model_detect(model_name)  # Load model

    if not model:
        logger.warning("Warning! You specified an invalid model type. "
                       "For image processing, the model with the best processing quality will be used. "
                       "({})".format(MODELS_NAMES[0]))
        model_name = MODELS_NAMES[0]  # If the model line is wrong, select the model with better quality.
        model = networks.model_detect(model_name)  # Load model

    preprocessing_method = preprocessing.method_detect(preprocessing_method_name)
    postprocessing_method = postprocessing.method_detect(postprocessing_method_name)
    output_path = Path(output_path)

    if isinstance(input_path, str) or isinstance(input_path, Path):
        input_path = Path(input_path)
        if input_path.is_file():
            image = model.process_image(input_path, preprocessing_method, postprocessing_method)
            __save_image_file__(image, input_path, output_path, crop_image, image_size)
            gc.collect()

        elif input_path.is_dir():
            if not recursive:
                gen_ext = [input_path.glob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
            else:
                gen_ext = [input_path.rglob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
            files = []
            for gen in gen_ext:
                for f in gen:
                    files.append(f)
            files = set(files)
            for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
                image = model.process_image(file, preprocessing_method, postprocessing_method)
                __save_image_file__(image, file, output_path, crop_image, image_size)
                gc.collect()
        else:
            if input_path.exists():
                raise ValueError("Bad input path parameter! "
                                 "Please indicate the correct path to the file or folder.")
            else:
                raise FileNotFoundError("The input path does not exist!")
    elif isinstance(input_path, list):
        if len(input_path) == 1:
            input_path = Path(input_path[0])

            if input_path.is_file():
                image = model.process_image(input_path, preprocessing_method, postprocessing_method)
                __save_image_file__(image, input_path, output_path, crop_image, image_size)
                gc.collect()

            elif input_path.is_dir():
                if not recursive:
                    gen_ext = [input_path.glob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
                else:
                    gen_ext = [input_path.rglob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
                files = []
                for gen in gen_ext:
                    for f in gen:
                        files.append(f)
                files = set(files)
                for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
                    image = model.process_image(file, preprocessing_method, postprocessing_method)
                    __save_image_file__(image, file, output_path, crop_image, image_size)
                    gc.collect()
            else:
                if input_path.exists():
                    raise ValueError("Bad input path parameter! "
                                     "Please indicate the correct path to the file or folder.")
                else:
                    raise FileNotFoundError("The input path does not exist!")
        else:
            files = []
            for in_p in input_path:
                input_path_p = Path(in_p)
                if input_path_p.is_file():
                    files.append(input_path_p)
                elif input_path_p.is_dir():
                    if not recursive:
                        gen_ext = [input_path_p.glob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
                    else:
                        gen_ext = [input_path_p.rglob("*.{}".format(e)) for e in SUPPORTED_EXTENSIONS]
                    for gen in gen_ext:
                        for f in gen:
                            files.append(f)
                else:
                    if not input_path_p.exists():
                        raise FileNotFoundError("The input path does not exist! Path: ", str(input_path_p.absolute()))

            files = set(files)
            for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
                image = model.process_image(file, preprocessing_method, postprocessing_method)
                __save_image_file__(image, file, output_path, crop_image, image_size)
                gc.collect()


def cli():
    """CLI"""
    parser = argparse.ArgumentParser(description=DESCRIPTION, usage=ARGS_HELP)

    parser.add_argument('-i', required=True, nargs="+",
                        help=ARGS["-i"][1], action="store", dest="input_path")
    parser.add_argument('-o', required=True,
                        help=ARGS["-o"][1], action="store", dest="output_path")
    parser.add_argument('-m', required=False,
                        help=ARGS["-m"][1],
                        action="store", dest="model_name", default=MODELS_NAMES[0])
    parser.add_argument('-pre', required=False,
                        help=ARGS["-pre"][1],
                        action="store", dest="preprocessing_method_name", default=PREPROCESS_METHODS[0])
    parser.add_argument('-post', required=False,
                        help=ARGS["-post"][1],
                        action="store", dest="postprocessing_method_name", default=POSTPROCESS_METHODS[0])
    parser.add_argument('--recursive', required=False, default=False,
                        help=ARGS['--recursive'][1], action="store_true", dest="recursive")
    parser.add_argument('-crop', required=False,  default=False,
                        help=ARGS["-crop"][1], dest="crop_image")
    parser.add_argument('-size', required=False,  default="original",
                        help=ARGS["-size"][1], dest="image_size")
    args = parser.parse_args()

    process(args.input_path, args.output_path,
            args.model_name, args.preprocessing_method_name,
            args.postprocessing_method_name, args.recursive, args.crop_image, args.image_size)


if __name__ == "__main__":
    cli()