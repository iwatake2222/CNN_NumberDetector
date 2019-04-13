# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo to classify image."""

import argparse
import re
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image, ImageOps
import numpy

# Function to read labels from text files.
def ReadLabelFile(file_path):
  """Reads labels from text file and store it in a dict.

  Each line in the file contains id and description separted by colon or space.
  Example: '0:cat' or '0 cat'.

  Args:
    file_path: String, path to the label file.

  Returns:
    Dict of (int, string) which maps label id to description.
  """
  with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = re.split(r'[:\s]+', line.strip(), maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
  parser.add_argument(
      '--label', help='File path of label file.', required=True)
  parser.add_argument(
      '--image', help='File path of the image to be recognized.', required=True)
  args = parser.parse_args()

  # Prepare labels.
  labels = ReadLabelFile(args.label)
  # Initialize engine.
  engine = ClassificationEngine(args.model)
  
  # Read input image and convert to tensor
  img = Image.open(args.image)
  img = img.convert('L')
  img = ImageOps.invert(img)
  input_tensor_shape = engine.get_input_tensor_shape()
  if (input_tensor_shape.size != 4 or input_tensor_shape[3] != 1 or
      input_tensor_shape[0] != 1):
    raise RuntimeError(
        'Invalid input tensor shape! Expected: [1, height, width, 3]')
  _, height, width, _ = input_tensor_shape
  img = img.resize((width, height), Image.NEAREST)
  input_tensor = numpy.asarray(img).flatten()

  # Run inference.
  for result in engine.ClassifyWithInputTensor(input_tensor=input_tensor, threshold=0.1, top_k=3):
    print('---------------------------')
    print(labels[result[0]])
    print('Score : ', result[1])

  # for time measurement
  for i in range(5000):
    for result in engine.ClassifyWithInputTensor(input_tensor=input_tensor, threshold=0.1, top_k=3):
      pass

if __name__ == '__main__':
  main()
