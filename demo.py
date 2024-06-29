import argparse
from pathlib import Path

import cv2
from boxmot import DeepOCSORT, BoTSORT

from traffic_counter.detectors.co_detr.co_detr_adapter import CODETRAdapter
from traffic_counter.detectors.rt_detr.rt_detr_adapter import RTDETRAdapter
from traffic_counter.detectors.yolov6.yolov6_adapter import YOLOv6Adapter
from traffic_counter.trackers.smiletrack.smiletrack import SMILEtrack
from traffic_counter.tracks_exporter import (
    BoTSORTTracksExporter,
    DeepOCSORTTracksExporter,
    SMILEtrackTracksExporter,
)

detectors_map = {
    "yolov6": YOLOv6Adapter,
    "rt-detr": RTDETRAdapter,
    "co-detr": CODETRAdapter,
}

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


def main(opt):
    vid = cv2.VideoCapture(opt.video)

    detector_class = detectors_map.get(opt.detector)
    tracker_class = trackers_map.get(opt.tracker)
    exporter_class = exporter_map.get(opt.tracker)

    if tracker_class is None or exporter_class is None:
        print("Give proper tracker from ['bot-sort', 'smiletrack', 'deep-oc-sort']")
        exit(1)

    detector = detector_class()
    tracker = tracker_class(
        model_weights=Path("osnet_x0_25_msmt17.pt"),
        device="cuda:0",
        fp16=False,
    )
    exporter = exporter_class(tracker, opt.video)

    i = 0
    while True:
        i += 1
        ret, im = vid.read()

        dets = detector.detect(im)
        tracker.update(dets, im)
        exporter.update(i)
        tracker.plot_results(im, show_trajectories=True)

        cv2.imshow("BoxMOT detection", im)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") or key == ord("q"):
            break

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

    main(opt)
