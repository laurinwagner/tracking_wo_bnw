
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

    def predict_boxes(self, images, boxes):
      self.eval()
      device = list(self.parameters())[0].device
      images = images.to(device)
      boxes = boxes.to(device)

      targets = None
      original_image_sizes = [img.shape[-2:] for img in images]

      images, targets = self.transform(images, targets)

      features = self.backbone(images.tensors)
      if isinstance(features, torch.Tensor):
          features = OrderedDict([(0, features)])

      # proposals, proposal_losses = self.rpn(images, features, targets)
      from torchvision.models.detection.transform import resize_boxes

      boxes = resize_boxes(
          boxes, original_image_sizes[0], images.image_sizes[0])
      proposals = [boxes]

      box_feats = self.roi_heads.box_roi_pool(
          features, proposals, images.image_sizes)
      box_features = self.roi_heads.box_head(box_feats)
      class_logits, box_regression = self.roi_heads.box_predictor(
          box_features)

      pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
      pred_scores = F.softmax(class_logits, -1)

      pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
      pred_boxes = resize_boxes(
          pred_boxes, images.image_sizes[0], original_image_sizes[0])
      pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
      

      return pred_boxes, pred_scores

    def load_image(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])