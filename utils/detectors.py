import numpy as np

class ObjectDetector():
    def __init__(self, session):
        import tensorflow as tf
        self.sess = sess = session

        # add detector to graph
        with tf.gfile.FastGFile('resources/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='detector')

        # output tensor
        outputs = [sess.graph.get_tensor_by_name('detector/num_detections:0'),
                   sess.graph.get_tensor_by_name('detector/detection_scores:0'),
                   sess.graph.get_tensor_by_name('detector/detection_boxes:0'),
                   sess.graph.get_tensor_by_name('detector/detection_classes:0')]
        self.outputs = outputs

    def detect(self, img, coco_id):
        # img is H x W x 3
        assert(np.ndim(img) == 3)

        img = np.expand_dims(img, axis=0)
        out = self.sess.run(self.outputs, feed_dict={'detector/image_tensor:0': img})

        (N, row, col, _) = img.shape
        num_detections = int(out[0][0])
        desired_class_ind = np.where(out[3][0] == coco_id)[0]
        class_scores = out[1][0][desired_class_ind]

        if len(desired_class_ind) == 0:
            return None # no detections of that class found

        # get the highest score corresponding to the desired class
        best_score_ind = np.argmax(class_scores)
        best_score_ind = desired_class_ind[best_score_ind]

        classId = int(out[3][0][best_score_ind])
        bbox = [float(v) for v in out[2][0][best_score_ind]]
        score = float(out[1][0][best_score_ind])

        assert(classId == coco_id)
        bbox_x = bbox[1] * col
        bbox_y = bbox[0] * row
        right = bbox[3] * col
        bottom = bbox[2] * row
        return (bbox_x, bbox_y, right, bottom)

class FaceDetector():
    def __init__(self):
        import dlib
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, img, *args):
        # img is H x W x 3
        assert(np.ndim(img) == 3)
        H, W, C = img.shape
        dets = self.detector(img, 1)
        if len(dets) == 0:
            # no face detected
            return None

        box_areas = []
        for i, d in enumerate(dets):
            box_areas.append((d.right() - d.left()) * (d.bottom() - d.top()))
        max_box = dets[np.argmax(box_areas)]
        bbox_x = max_box.left()
        bbox_y = max_box.top()
        right = max_box.right()
        bottom = max_box.bottom()

        return (bbox_x, bbox_y, right, bottom)
