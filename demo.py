import argparse
import json

import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT, BoTSORT

from traffic_counter.detectors.co_detr.mmdet.apis import (
    inference_detector,
    init_detector,
)

from traffic_counter.trackers.smiletrack.smiletrack import SMILEtrack

from traffic_counter.detectors.yolov6.yolov6.core.inferer import Inferer
from ultralytics import RTDETR

from tracks_exporter import (
    BoTSORTTracksExporter,
    DeepOCSORTTracksExporter,
    SMILEtrackTracksExporter,
)

trackers_map = {
    "deep-oc-sort": DeepOCSORT,
    "bot-sort": BoTSORT,
    "smiletrack": SMILEtrack,
}

exporter_map = {
    "deep-oc-sort": DeepOCSORTTracksExporter,
    "bot-sort": BoTSORTTracksExporter,
    "smiletrack": SMILEtrackTracksExporter,
}


def convert_dets(dets):
    dets2 = []
    for i, class_dets in enumerate(dets):
        for j in class_dets:
            tmp = j[0:5]
            new_det = np.append([tmp], [i])
            dets2.append(new_det)
    return np.array(dets2)


def track(opt):

    vid = cv2.VideoCapture(opt.video)
    tracker_class = trackers_map.get(opt.tracker)
    exporter_class = exporter_map.get(opt.tracker)
    if tracker_class is None or exporter_class is None:
        print("Give proper tracker from ['bot-sort', 'smiletrack', 'deep-oc-sort']")
        exit(1)

    tracker = tracker_class(
        model_weights=Path("osnet_x0_25_msmt17.pt"),
        device="cuda:0",
        fp16=False,
    )
    exporter = exporter_class(tracker, opt.video)

    if opt.detector == "yolov6":
        weights = "traffic_counter/detectors/yolov6/weights/yolov6l6.pt"
        device = 0
        yaml = "traffic_counter/detectors/yolov6/data/coco.yaml"
        half = True
        img_size = [640, 640]

        inferer = Inferer(None, weights, device, yaml, img_size, half)
    elif opt.detector == "rt=detr":
        inferer = RTDETR("rtdetr-x.pt")
    elif opt.detector == "co-detr":
        checkpoint = "traffic_counter/detectors/co_detr/checkpoints/co_dino_5scale_swin_large_3x_coco.pth"
        config = "traffic_counter/detectors/co_detr/projects/configs/co_dino/co_dino_5scale_swin_large_3x_coco.py"
        model = init_detector(config, checkpoint, device="cuda:0")

    i = 0
    while True:
        i += 1
        ret, im = vid.read()

        if opt.detector == "yolov6":
            conf_thres = 0.4
            iou_thres = 0.45
            classes = None
            agnostic_nms = True
            max_det = 1000

            det = inferer.infer_on_image(
                im, conf_thres, iou_thres, classes, agnostic_nms, max_det
            )
            dets = np.array(det.cpu())
        elif opt.detector == "rt-detr":
            det = inferer(im)
            dets = np.array(det[0].boxes.data.cpu())
        elif opt.detector == "co-detr":
            det = inference_detector(model, im)
            dets = convert_dets(det)

        tracker.update(dets, im)

        exporter.update(i)
        tracker.plot_results(im, show_trajectories=True)

        # break on pressing q or space
        cv2.imshow("BoxMOT detection", im)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") or key == ord("q"):
            break

    # save_results()
    exporter.save_ottrk()
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector",
        type=str,
        default="yolov6",
        help='Detector from ["yolov6", "rt-detr", "co-detr"]',
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="inference/images",
        help='Tracker from ["bot-sort", "smiletrack", "deep-oc-sort"]',
    )
    parser.add_argument(
        "--video", type=str, default="test_video.mp4", help="video path"
    )
    opt = parser.parse_args()

    track(opt)
