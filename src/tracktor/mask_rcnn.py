import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.roi_heads import maskrcnn_inference, paste_masks_in_image

class Mask_RCNN(MaskRCNN):

    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(Mask_RCNN, self).__init__(backbone, num_classes)

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

      mask_features = self.roi_heads.mask_roi_pool(features, proposals, images.image_sizes)
      cropped_features = self.roi_heads.mask_head(mask_features)
      mask_logits = self.roi_heads.mask_predictor(cropped_features)

      switch_channel_masks = torch.zeros(mask_logits.size())
      switch_channel_masks[:,0,:,:] = mask_logits[:,1,:,:]

      # workaround that only works with 2 classes. otherwise try to get maskrcnn_inference running
      # or manually filter out the class with highest score here
      switch_channel_masks = torch.sigmoid(switch_channel_masks)
      pred_masks = paste_masks_in_image(switch_channel_masks,pred_boxes,original_image_sizes[0]).detach()
      

      return pred_boxes, pred_scores, pred_masks


    def load_image(self, img):
        pass