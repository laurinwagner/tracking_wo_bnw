import torch
import torch.nn.functional as F

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from .utils import py_cpu_softnms

class Mask_RCNN(MaskRCNN):

    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(Mask_RCNN, self).__init__(backbone, num_classes)

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()


    def predict_masks(self, images):
        self.eval()
        with torch.no_grad():
            prediction = self([img.to(device)])
        #get boxes
        #dets=prediction[0]['boxes'].cpu().numpy()
        #get scores
        #sc=prediction[0]['scores'].cpu().numpy()
        #perform NMS with Method specified in py_cpus_softnms to select boxes
        nms_index=py_cpu_softnms(dets,sc,Nt=0.3, thresh=thresh, method=3)
        predicted_masks=[]
        for i in nms_index:
            predicted_masks.append((prediction[0]['masks'][i, 0].cpu().numpy()>mask_thresh))


    def predict_boxes(self, images, boxes):

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

        box_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(
            box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        mask_features = self.roi_heads.mask_roi_pool(features, proposals, images.image_sizes)
        pred_masks = self.roi_heads.mask_predictor(mask_features)
        mask_sigmoid =[]
        for mask in pred_masks:
            mask_sigmoid.append(F.sigmoid(mask))
        
        # score_thresh = self.roi_heads.score_thresh
        # nms_thresh = self.roi_heads.nms_thresh

        # self.roi_heads.score_thresh = self.roi_heads.nms_thresh = 1.0
        # self.roi_heads.score_thresh = 0.0
        # self.roi_heads.nms_thresh = 1.0
        # detections, detector_losses = self.roi_heads(
        #     features, [boxes.squeeze(dim=0)], images.image_sizes, targets)

        # self.roi_heads.score_thresh = score_thresh
        # self.roi_heads.nms_thresh = nms_thresh

        # detections = self.transform.postprocess(
        #     detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]
        # return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(
            pred_boxes, images.image_sizes[0], original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores, mask_sigmoid

    def load_image(self, img):
        pass
