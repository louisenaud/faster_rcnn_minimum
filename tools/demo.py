import cv2
import argparse
import torch
from torchvision import transforms as T

from faster_rcnn_minimum.config import cfg
from faster_rcnn_minimum.utils.cococat import get_coco_categories
from faster_rcnn_minimum.modeling.detector import build_detection_model
from faster_rcnn_minimum.utils.checkpoint import DetectronCheckpointer
from faster_rcnn_minimum.utils.logger import setup_logger
from faster_rcnn_minimum.utils.comm import get_rank
from faster_rcnn_minimum.structures.image_list import to_image_list

class Demo(object):
    """
    a class for demo inference
    """
    def __init__(self, cfg, confidence_threshold=0.7,
                 min_image_size=800, logger=None):
        """
        Args:
            cfg (config): config setting
            confidence_threshold (float): threshold to cut off the low-score objects
            min_image_size (int): shorter-side length of the resized image
            logger (object): logger
        """

        self.cfg = cfg.clone()

        # model setup
        self.model = build_detection_model(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cpu_device = torch.device("cpu")
        self.model.to(self.device)
        save_dir = cfg.OUTPUT_DIR

        # read pre-trained checkpoint
        checkpointer = DetectronCheckpointer(
            cfg, self.model, save_dir=save_dir, logger=logger)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        self.model.eval()

        self.confidence_threshold = confidence_threshold
        self.min_image_size = min_image_size
        self.transforms = self.build_transform()
        self.logger = logger
        self.coco_categories = get_coco_categories()
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

    def log_model(self):
        """
        log the model information
        """
        self.logger.info(self.model)

    def build_transform(self):
        cfg = self.cfg
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),  # convert np.ndarray to PIL Image
                T.Resize(self.min_image_size),  # resize the image
                T.ToTensor(),  # convert PIL Image to Tensor with value range of 0 to 1
                to_bgr_transform,  # change the value range to [0, 255]
                normalize_transform  # normalize the value range
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        run the demo on an image loaded by opencv
        Args:
            image (np.ndarray): an image as returned by opencv
        Returns:
            prediction (BoxList): the detected objects

        """
        # run inference
        predictions = self.compute_prediction(image)
        # select the top inference
        top_predictions = self.select_top_predictions(predictions)
        # draw the results on the input image
        result = image.copy()
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result

    def select_top_predictions(self, predictions):
        """
        filter the low-score predictions
        Args:
            predictions (list(boxlist)): prediction boxes at the output of the model
        Returns:
            predictions (list(boxlist)): filtered prediction boxes

        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_prediction(self, original_image):
        """
        run image preprocess, inference and box postprocess
        Args:
            original_image (np.ndarray): an image as returned by opencv
        Returns:
            prediction (np.ndarray): output image with visualization of the results
        """
        # image preprocess
        image = self.transforms(original_image)
        image_list = to_image_list(image)
        image_list = image_list.to(self.device)

        # run inference
        # -> faster_rcnn_minimum.modeling.detector build_detection_model
        with torch.no_grad():
            predictions = self.model(image_list)

        # results postprocess
        predictions = [o.to(self.cpu_device) for o in predictions]
        prediction = predictions[0]
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        return prediction

    def compute_colors_for_labels(self, labels):
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image
        Args:
            image (np.ndarray):
            predictions:
        Returns:

        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_class_names(self, image, predictions):
        """

        Args:
            image:
            predictions:

        Returns:

        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.coco_categories[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        '--config',
        default='configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml',
        metavar='FILE',
        help='path to config file',
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--image', default=None, type=str, help='Input image for demo')

    return parser.parse_args()

def main():
    args = parse_args()
    save_dir = ""

    # setup config
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # setup logger
    logger = setup_logger("faster_rcnn_minimum", save_dir, get_rank())
    logger.info("Configuration: ")
    logger.info(cfg)

    # demo class
    demo = Demo(cfg, logger=logger)
    demo.log_model()

    # read an image
    image = cv2.imread(args.image)
    assert image is not None

    # run inference
    predictions = demo.run_on_opencv_image(image)

    # plot the result
    import matplotlib.pyplot as plt
    # TODO: background mode
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(predictions[:, :, ::-1])
    plt.show()

if __name__ == "__main__":
    main()