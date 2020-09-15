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
from statistics import mean


def signal_handler(sig, frame):
    """
    Close video (so the video and the results.txt will be saved) when given CTAL+C.
    Relevant mainly for webcam scenario (infinite loop).
    """
    print("You pressed CTAL+C!, exiting while save log & output video (if there is) properly")
    cap.release()
    video.release()
    cv2.destroyAllWindows()
	time.sleep(3)


 # CTRL+C
signal.signal(signal.SIGINT, signal_handler)
# stop on debugger
signal.signal(signal.SIGTERM, signal_handler)


def parser():
    """
    Arguments:
    input - video source
    out_filename - out video name
    export_logname - out log name
    weights - yolo weights path
    dont_show - hide window inference display
    config_file - path to config file
    data_file - path to data file
    thresh - remove detections with confidence below this value
    capture_frame_width - define the camera frame width
    capture_frame_height - define the camera frame height
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
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    parser.add_argument("--capture_frame_width", default=1280,
                        help="define the camera frame width")
    parser.add_argument("--capture_frame_height", default=720,
                        help="define the camera frame height")
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
    Arguments checker, raises error for false arguments.
    """
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("\nInvalid config path {}".format(
            os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("\nInvalid weight path {}".format(
            os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("\nInvalid data file path {}".format(
            os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("\nInvalid video path {}".format(
            os.path.abspath(args.input))))
    if not args.export_logname:
        raise(ValueError("\nNeed to set results log-name"))
    if args.out_filename and not args.out_filename.endswith('.mp4'):
        raise(ValueError("\nOut file name need to end with '.mp4'"))


def set_saved_video(input_video, output_video, size):
    """
    creating the result video obj.
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def video_capture(frame_queue, darknet_image_queue):
    """
    reading frames from the caputre (webcam\video) and the time of caputre,
    and push them into queues for farther use.
    """
	width_input = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # float
    height_input = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    print("Input resolution is: {}x{} (if 0x0, then the camera is occupied with something else)".format(int(width_input), int(height_input)))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # capture_time
        capture_time_queue.put(time.time())
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_resized)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        darknet_image_queue.put(darknet_image)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    """
    inference the captures into the darknet.
    function is also in charge of the printing/writing (fps, caputre time, detections).
    """
	# results log
    logname = args.export_logname 
	""" OPTION: """
    # each time will open a new txt file
    logname_split = args.export_logname.rsplit(".", 1)
    index = 0
    while 1:
        # name_<index>.txt
        logname = logname_split[0] + '_' + str(index) + '.' + logname_split[1]
        # file not exists
        if not os.path.isfile(logname):
            break
        # trying next index
        index += 1
	""" OPTION: END """
    f = open(logname, "w")
    enter_time_queue = [0, 0, 0]
    exit_time_queue = [1, 1, 1]
    while cap.isOpened():
        # get new image from queue
        darknet_image = darknet_image_queue.get()
        # sample entering time
        prev_time = time.time()
        enter_time_queue.pop(0)
        enter_time_queue.append(prev_time)
        # detect image (inference image in neural network)
        detections = darknet.detect_image(
            network, class_names, darknet_image, thresh=args.thresh)
        # store result in queue
        detections_queue.put(detections)
        # calculate fps of passing image
        fps = float(1 / (time.time() - prev_time))
        exit_time_queue.pop(0)
        exit_time_queue.append(time.time())
        # store fps in queue
        fps_queue.put(int(fps))
        # calculate the average fps of 3 last frame (just to follow up)
        fps_list = [1./(m - n)
                    for m, n in zip(exit_time_queue, enter_time_queue)]
        print("Average FPS over last 3 frames is: {:.2f}".format(
            mean(fps_list)))
        # store capture time to file (in ms, for ground station)	
		f.write("time: {}\n".format(str(round(capture_time_queue.get()*1000)))) 
        # store bbox to file
        height_ratio = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/height
        width_ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH)/width
        darknet.print_detections(detections, height_ratio, width_ratio, f)
        f.write("\n")
    cap.release()
    f.close()
    print("\nFinished successfully, results: {}".format(logname))


def drawing(frame_queue, detections_queue, fps_queue):
    """
    drawing bbox on the image and writing results video file or show video image.
    """
    # so we could release it if a signal is given
    global video
    # deterministic bbox colors
    random.seed(3)
    # results video file
    filename = args.out_filename
    # each time will open a new out file
    if args.out_filename:
        filename_split = args.out_filename.rsplit(".", 1)
        index = 0
        while 1:
            # save file: name_<index>.mp4
            filename = filename_split[0] + '_' + \
                str(index) + '.' + filename_split[1]
            # file not exists
            if not os.path.isfile(filename):
                break
            # trying next index
            index += 1
    # result video obj
    video = set_saved_video(
        cap, filename, (args.capture_frame_width, args.capture_frame_height))
    while cap.isOpened():
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        if frame_resized is not None:
            # draw detection bounding boxs on image
            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.capture_frame_width,
                                       args.capture_frame_height), interpolation=cv2.INTER_LINEAR)
            # writing video image
            if args.out_filename is not None:
                video.write(image)
            # show video image
            if not args.dont_show:
                cv2.imshow('Inference', image)
            # Esc key to stop GUI
            if cv2.waitKey(fps) == 27:
                break
    # Closes video file or capturing device
    cap.release()
    # Closes video write file
    video.release()
    # destroys all of the opened HighGUI windows
    cv2.destroyAllWindows()
    if args.out_filename:
        print("\nOut file: {}".format(filename))


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    capture_time_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file, args.data_file, args.weights, batch_size=1)
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_frame_height)
    Thread(target=video_capture, args=(
        frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue,
                                   detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue,
                                 detections_queue, fps_queue)).start()
