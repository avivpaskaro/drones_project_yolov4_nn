from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
from datetime import datetime
import signal
import sys
def signal_handler(sig, frame):
    """
    Close video (so the video and the results.txt will be saved) when given CTAL+C.
    Relevant mainly for webcam scenario (infinite loop).
    """
    print("You pressed CTAL+C!, exiting while save log & output video (if there is) properly")
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    time.sleep(5)
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)  # CTRL+C
signal.signal(signal.SIGTERM, signal_handler)  # stop on debugger


def parser():
    """
    Arguments parser.
    """
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--export_logname", type=str, default="",
                        help="out log name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="window inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed.
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    """
    Arguments checker, returns error for false arguments.
    """
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("\nInvalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("\nInvalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("\nInvalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("\nInvalid video path {}".format(os.path.abspath(args.input))))
    if not args.export_logname:
        raise(ValueError("\nNeed to set results log-name"))
    if args.out_filename and not args.out_filename.endswith('.mp4'):
        raise(ValueError("\nOut file name need to end with '.mp4'"))


def set_saved_video(input_video, output_video, size):
    """
    creating the result video file.
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def video_capture(frame_queue, darknet_image_queue, darknet_image_time_queue):
    """
    reading frames from the caputre (webcam\video) and the time of caputre,
    and push them into queues for farther use.
    the function uses N variable for sampling each of the N'th frame.
    """
    N = 0
    while cap.isOpened():
        ret, frame = cap.read()
        t = datetime.now()
        if not ret:
            break
        if not (N % 6):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            frame_queue.put(frame_resized)
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            darknet_image_queue.put(darknet_image)
            darknet_image_time_queue.put(t)
        N += 1
    cap.release()


def inference(darknet_image_queue, darknet_image_time_queue, network_width, network_height, detections_queue, fps_queue):
    """
    inference the captures into the darknet.
    function is also in charge of the printing/writing (fps, caputre time, detections).
    """
    logname = args.export_logname
    f = open(logname, "w")
    """ OPTION: each time will open a new txt file
    logname_split = args.export_logname.rsplit(".", 1)
    index = 0
    while 1:
        logname = logname_split[0] + '_' + str(index) + '.' + logname_split[1] # name_<index>.txt
        if not os.path.isfile(logname):
            break
        index += 1
    f = open(logname, "w")
    """
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = float(1/(time.time() - prev_time))
        fps_queue.put(int(fps))
        print("FPS: {:.2f}".format(fps))
        f.write("time: {}\n".format(darknet_image_time_queue.get()))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        darknet.print_detections(detections, h/network_height, w/network_width, f)
        f.write("\n\n")
    cap.release()
    f.close()
    print("\nFinished successfully, results: {}".format(logname))


def drawing(frame_queue, detections_queue, fps_queue):
    """
    drawing bbox on the capture and writing image.
    """
    global video  # so we could release it if a signal is given
    random.seed(3)  # deterministic bbox colors
    filename = args.out_filename
    if args.out_filename: # each time will open a new out file
        filename_split = args.out_filename.rsplit(".", 1)
        index = 0
        while 1:
            filename = filename_split[0] + '_' + str(index) + '.' + filename_split[1] # save file: name_<index>.mp4
            if not os.path.isfile(filename):
                break
            index += 1
    video = set_saved_video(cap, filename, (width, height))
    while cap.isOpened():
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        if frame_resized is not None:
            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if args.out_filename is not None:
                video.write(image)
            if not args.dont_show:
                cv2.imshow('Inference', image)
            if cv2.waitKey(fps) == 27:
                break

    cap.release()
    video.release()
    cv2.destroyAllWindows()
    if args.out_filename:
        print("\nOut file: {}".format(filename))


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    darknet_image_time_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue, darknet_image_time_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, darknet_image_time_queue, width, height, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
