"""
A simple demo showcasing the Python wrapping.
"""
import cv2 as cv
import numpy as np

from typing import List

# Add all the DLL directories required (todo: this is terrible, is there not a better way?)
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Program Files/NVIDIA Corporation/CUDNN/v8.3/bin")
os.add_dll_directory("C:/Program Files/NVIDIA Corporation/TensorRT-8.5.2.2/lib")
os.add_dll_directory("D:/VU/Code/tkDNN_python/build/Release")
os.add_dll_directory("D:/VU/Code/tkDNN_python/opt/x64-windows/bin")

import pythonwrapper

help(pythonwrapper)


def draw_detections(image: np.ndarray, detections: List[pythonwrapper.Detection], class_count: int) -> np.ndarray:
    output_image = image.copy()
    colors = np.random.uniform(0, 255, size=(class_count, 3))
    for detection in detections:
        top_left = int(detection.box.x), int(detection.box.y)
        bottom_right = int(detection.box.x + detection.box.width), int(detection.box.y + detection.box.height)
        color = colors[detection.class_id]
        cv.rectangle(output_image, top_left, bottom_right, color, 2)
        cv.putText(output_image, detection.class_name, (top_left[0] - 10, top_left[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output_image


def main():
    # Load input image
    input_image = cv.imread("test_image.png", cv.IMREAD_COLOR)
    assert isinstance(input_image, np.ndarray)
    assert input_image.dtype == np.uint8

    # Detect objects
    class_count = 80
    detector = pythonwrapper.ObjectDetector("../build/yolo4tiny_fp32.rt", class_count)
    detections = detector.infer([input_image])
    print(detections)

    # Create resulting image with detections and write it out
    output_image = draw_detections(input_image, detections, class_count)
    cv.imwrite("test_output_python.png", output_image)

    # Show image
    cv.imshow("result", output_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
