# coding: utf-8

import os
import glob
import numpy as np
import cv2
import onnxruntime

from render_utils.prior_box import PriorBox
from render_utils.nms_wrapper import nms
from render_utils.box_utils import decode
from render_utils.timer import Timer
from render_utils.config import cfg

confidence_threshold = 0.05
top_k = 5000
keep_top_k = 750
nms_threshold = 0.3
vis_thres = 0.5
resize = 1

scale_flag = True
HEIGHT, WIDTH = 720, 1080

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_onnx_path():
    path_docker = "/app/weights/skull_weights/FaceBoxesProd.onnx"
    if os.path.exists(path_docker):
        return path_docker

    path_local = os.path.abspath(os.path.join(BASE_DIR, "../../weights/skull_weights/FaceBoxesProd.onnx"))
    if os.path.exists(path_local):
        return path_local

    found_files = glob.glob('/app/**/FaceBoxesProd.onnx', recursive=True)
    if found_files:
        return found_files[0]

    return path_docker


def viz_bbox(img, dets, wfp='out.jpg'):
    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imwrite(wfp, img)
    print(f'Viz bbox to {wfp}')


class FaceBoxes_ONNX(object):
    def __init__(self, timer_flag=False):
        onnx_path = get_onnx_path()

        if not os.path.exists(onnx_path):
            print(f'[ERROR] onnx_path does not exist at: {onnx_path}')

        model_dir = os.path.dirname(onnx_path)
        model_basename = os.path.basename(onnx_path)
        current_cwd = os.getcwd()

        try:
            if model_dir and os.path.exists(model_dir):
                os.chdir(model_dir)
                self.session = onnxruntime.InferenceSession(model_basename, None)
            else:
                self.session = onnxruntime.InferenceSession(onnx_path, None)
        except Exception as e:
            print(f"[ERROR FaceBoxes] Error loading model: {e}")
            raise e
        finally:
            os.chdir(current_cwd)

        self.timer_flag = timer_flag

    def __call__(self, img_):
        img_raw = img_.copy()

        scale = 1
        if scale_flag:
            h, w = img_raw.shape[:2]
            if h > HEIGHT:
                scale = HEIGHT / h
            if w * scale > WIDTH:
                scale *= WIDTH / (w * scale)
            if scale == 1:
                img_raw_scale = img_raw
            else:
                h_s = int(scale * h)
                w_s = int(scale * w)
                img_raw_scale = cv2.resize(img_raw, dsize=(w_s, h_s))

            img = np.float32(img_raw_scale)
        else:
            img = np.float32(img_raw)

        _t = {'forward_pass': Timer(), 'misc': Timer()}
        im_height, im_width, _ = img.shape
        scale_bbox = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=np.float32)

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]

        _t['forward_pass'].tic()
        out = self.session.run(None, {'input': img})
        loc, conf = out[0], out[1]

        loc = np.squeeze(loc, axis=0)
        _t['forward_pass'].toc()
        _t['misc'].tic()

        priorbox = PriorBox(image_size=(im_height, im_width))
        # priorbox = PriorBox(image_size=(HEIGHT, WIDTH))
        priors = priorbox.forward()
        prior_data = priors.data if hasattr(priors, 'data') else priors

        boxes = decode(loc, prior_data, cfg['variance'])
        if scale_flag:
            boxes = boxes * scale_bbox / scale / resize
        else:
            boxes = boxes * scale_bbox / resize

        scores = conf[0][:, 1]

        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, nms_threshold)
        dets = dets[keep, :]

        dets = dets[:keep_top_k, :]
        _t['misc'].toc()

        if self.timer_flag:
            print('Detection: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(1, 1, _t[
                'forward_pass'].average_time, _t['misc'].average_time))

        det_bboxes = []
        for b in dets:
            if b[4] > vis_thres:
                xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                bbox = [xmin, ymin, xmax, ymax, score]
                det_bboxes.append(bbox)

        return det_bboxes


def main():
    face_boxes = FaceBoxes_ONNX(timer_key=True) if 'timer_key' in globals() else FaceBoxes_ONNX(timer_flag=True)

    fn = 'trump_hillary.jpg'
    img_fp = f'../examples/inputs/{fn}'
    img = cv2.imread(img_fp)
    if img is not None:
        print(f'input shape: {img.shape}')
        dets = face_boxes(img)
        wfn = fn.replace('.jpg', '_det.jpg')
        wfp = os.path.join('../examples/results', wfn)
        viz_bbox(img, dets, wfp)


if __name__ == '__main__':
    main()