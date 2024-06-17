import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT

from yolov6.core.inferer import Inferer


tracker = DeepOCSORT(
    model_weights=Path("osnet_x0_25_msmt17.pt"),
    device="cuda:0",
    fp16=False,
)

vid = cv2.VideoCapture("test_video.mp4")

weights = "weights/yolov6l6.pt"
device = 0
yaml = "data/coco.yaml"
half = True
img_size = [640, 640]

inferer = Inferer(None, weights, device, yaml, img_size, half)

while True:
    ret, im = vid.read()

    conf_thres = 0.4
    iou_thres = 0.45
    classes = None
    agnostic_nms = True
    max_det = 1000

    det = inferer.infer_on_image(
        im, conf_thres, iou_thres, classes, agnostic_nms, max_det
    )

    dets = np.array(det.cpu())
    tracker.update(dets, im)
    tracker.plot_results(im, show_trajectories=True)

    # break on pressing q or space
    cv2.imshow("BoxMOT detection", im)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") or key == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
