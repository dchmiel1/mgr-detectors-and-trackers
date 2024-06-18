import sys

import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT, BoTSORT

from traffic_counter.detectors.co_detr.mmdet.apis import inference_detector, init_detector, show_result_pyplot

from traffic_counter.detectors.yolov6.yolov6.core.inferer import Inferer
from ultralytics import RTDETR

SUPPORTED_DETECTORS = ["yolo", "rtdetr", "codetr"]

detector = "yolo" if len(sys.argv) < 2 else sys.argv[1]

if detector not in SUPPORTED_DETECTORS:
    print("Invalid detector. Provide one from ", SUPPORTED_DETECTORS)
    exit(1)

tracker = DeepOCSORT(
    model_weights=Path("osnet_x0_25_msmt17.pt"),
    device="cuda:0",
    fp16=False,
)

# tracker = BoTSORT(
#     model_weights=Path("osnet_x0_25_msmt17.pt"),
#     device="cuda:0",
#     fp16=False,
# )

vid = cv2.VideoCapture("test_video.mp4")

if detector == "yolo":
    weights = "traffic_counter/detectors/yolov6/weights/yolov6l6.pt"
    device = 0
    yaml = "traffic_counter/detectors/yolov6/data/coco.yaml"
    half = True
    img_size = [640, 640]

    inferer = Inferer(None, weights, device, yaml, img_size, half)
elif detector == "rtdetr":
    inferer = RTDETR("rtdetr-x.pt")
elif detector == "codetr":
    checkpoint = "traffic_counter/detectors/co_detr/checkpoints/co_dino_5scale_swin_large_3x_coco.pth"
    config = "traffic_counter/detectors/co_detr/projects/configs/co_dino/co_dino_5scale_swin_large_3x_coco.py"
    model = init_detector(config, checkpoint, device="cuda:0")

def convert_dets(dets):
    dets2 = []
    for i, class_dets in enumerate(dets):
        for j in class_dets:
            tmp = j[0:5]
            new_det = np.append([tmp], [i])
            dets2.append(new_det)
    return np.array(dets2)



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
    elif detector == "codetr":
        det = inference_detector(model, im)
        dets = convert_dets(det)
    #     dets = det[2]
    #     dets2 = []
    #     for i in dets:
    #         print(type(i))
    #         # np.insert(i, [4], [0])
    #         l = i[0:5]
    #         print(l)
    #         a = np.append([l], [2])
    #         print(a)
    #         # b = np.append([a], [i[4]])
    #         # print(b)
    #         dets2.append(a)
    # dets = np.array(dets2)

            # np.concatenate(i[0:4], [0], [i[4]])
            # i.insert(0, 4)

    tracker.update(dets, im)
    tracker.plot_results(im, show_trajectories=True)

    # break on pressing q or space
    cv2.imshow("BoxMOT detection", im)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") or key == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
