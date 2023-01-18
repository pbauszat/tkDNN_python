"""
A simple demo showcasing the Python wrapping.
"""
import cv2 as cv
import numpy as np
import pathlib
import timeit

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
    rng = np.random.default_rng(123456789)
    colors = rng.uniform(0, 255, size=(class_count, 3))
    for detection in detections:
        top_left = int(detection.box.x), int(detection.box.y)
        bottom_right = int(detection.box.x + detection.box.width), int(detection.box.y + detection.box.height)
        color = colors[detection.class_id]
        cv.rectangle(output_image, top_left, bottom_right, color, 2)
        cv.putText(output_image, detection.class_name, (top_left[0] - 10, top_left[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output_image


def main():
    # Load test images
    image_files = ["frame400_image.png", "frame700_image.png"]
    images = list()
    for filename in image_files:
        image = cv.imread(filename, cv.IMREAD_COLOR)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        images.append(image)

    # Detect objects (run multiple times to get cached performance)
    network_rt_file = "../build/yolo4tiny_fp32.rt"  # or use "../build/mobilenetv2ssd_fp32.rt" for MobileNet
    class_count = 80
    detector = pythonwrapper.ObjectDetector(network_rt_file, class_count, max_batch_size=2)
    detections = None
    for _ in range(10):
        start_time = timeit.default_timer()
        detections = detector.infer(images)
        end_time = timeit.default_timer()
        print("Inference time: %.1f ms" % (1000 * (end_time - start_time)))
    print(f"Detections: {detections}")

    # Create resulting image with detections and write it out
    for filename, input_image, image_detections in zip(image_files, images, detections):
        input_filename = pathlib.Path(filename)
        output_filename = input_filename.with_name(input_filename.stem + "_output" + input_filename.suffix)
        output_image = draw_detections(input_image, image_detections, class_count)
        # Save and show image
        cv.imwrite(str(output_filename), output_image)
        cv.imshow(str(output_filename), output_image)

    # Close all windows after key is pressed
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
