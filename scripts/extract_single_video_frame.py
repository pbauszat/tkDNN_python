"""
Extracts a single frame from an input video and writes out that frame as image plus a video with only a single frame.
"""
import argparse
import cv2 as cv


def main():
    # Read-in arguments
    args = argparse.ArgumentParser()
    args.add_argument("--video_file", type=str, default="../demo/yolo_test.mp4", help="Video filename.")
    args.add_argument("--frame_index", type=int, default="700", help="Video frame index.")
    args.add_argument("--output_image", type=str, default="../demo/test_image.png", help="Output image.")
    args.add_argument("--output_video", type=str, default="../demo/test_video.avi", help="Single-frame output video.")
    args = args.parse_args()

    # Load the video
    capture = cv.VideoCapture(args.video_file)
    assert capture.isOpened()
    source_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    frames_per_second = capture.get(cv.CAP_PROP_FPS)
    source_resolution = source_width, source_height

    # Jump to the target frame and read it
    assert args.frame_index < frame_count
    capture.set(cv.CAP_PROP_POS_FRAMES, args.frame_index)
    read_success, frame = capture.read()
    assert read_success

    # Write out the frame to an image file
    cv.imwrite(args.output_image, frame)

    # Write out the frame to an single-frame video
    codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    output_capture = cv.VideoWriter(args.output_video, codec, frames_per_second, source_resolution)
    output_capture.write(frame)
    output_capture.release()

    # Finally, close the input capture
    capture.release()
    print("Done.")


if __name__ == "__main__":
    main()
