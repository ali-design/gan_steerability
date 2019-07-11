import cv2
import numpy as np


def detect_object(g, img, coco_id, return_im=False):
    # img is a single image with dim H x W  x C 
    img = np.expand_dims(img, axis=0)
    out = g.sess.run([g.sess.graph.get_tensor_by_name('num_detections:0'),
                      g.sess.graph.get_tensor_by_name('detection_scores:0'),
                      g.sess.graph.get_tensor_by_name('detection_boxes:0'),
                      g.sess.graph.get_tensor_by_name('detection_classes:0')],
                      feed_dict={'image_tensor:0': img})

    (N, row, col, _) = img.shape
    assert N == 1
    num_detections = int(out[0][0])

    desired_class_ind = np.where(out[3][0] == coco_id)[0]
    class_scores = out[1][0][desired_class_ind]

    if len(desired_class_ind) == 0:
        # print("No class found")
        return None # empty list

    best_score_ind = np.argmax(class_scores)
    best_score_ind = desired_class_ind[best_score_ind]
    classId = int(out[3][0][best_score_ind])
    bbox = [float(v) for v in out[2][0][best_score_ind]]
    score = float(out[1][0][best_score_ind])

    if best_score_ind != 0:
        pass
        #print("best score rank: {}".format(best_score_ind))

    if score < 0.3:
        pass
        #print("low max score: {}".format(score))

    bbox_x = bbox[1] * col
    bbox_y = bbox[0] * row
    right = bbox[3] * col
    bottom = bbox[2] * row

    if not return_im:
        return (bbox_x, bbox_y, right, bottom)

    # draws the bbox on the input image
    img_out = cv2.rectangle(img[0], (int(bbox_x), int(bbox_y)), (int(right), int(bottom)),
        (255, 255, 255), thickness=2)
    if isinstance(img_out, cv2.Mat):
        img_out = cv2.UMat.get(img_out)
    return ((bbox_x, bbox_y, right, bottom), img_out)
