import sys

import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT

from yolov6.core.inferer import Inferer
from ultralytics import RTDETR

SUPPORTED_DETECTORS = ["yolo", "rtdetr"]

detector = "yolo" if len(sys.argv) < 2 else sys.argv[1]

if detector not in SUPPORTED_DETECTORS:
    print("Invalid detector. Provide one from ", SUPPORTED_DETECTORS)
    exit(1)

tracker = DeepOCSORT(
    model_weights=Path("osnet_x0_25_msmt17.pt"),
    device="cuda:0",
    fp16=False,
)

vid = cv2.VideoCapture("test_video.mp4")

if detector == "yolo":
    weights = "weights/yolov6l6.pt"
    device = 0
    yaml = "data/coco.yaml"
    half = True
    img_size = [640, 640]

    inferer = Inferer(None, weights, device, yaml, img_size, half)
elif detector == "rtdetr":
    inferer = RTDETR("rtdetr-x.pt")

while True:
    ret, im = vid.read()

    if detector == "yolo":
        conf_thres = 0.4
        iou_thres = 0.45
        classes = None
        agnostic_nms = True
        max_det = 1000

        det = inferer.infer_on_image(
            im, conf_thres, iou_thres, classes, agnostic_nms, max_det
        )
        dets = np.array(det.cpu())
    elif detector == "rtdetr":
        det = inferer(im)
        dets = np.array(det[0].boxes.data.cpu())

    tracker.update(dets, im)
    tracker.plot_results(im, show_trajectories=True)

    # break on pressing q or space
    cv2.imshow("BoxMOT detection", im)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") or key == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
