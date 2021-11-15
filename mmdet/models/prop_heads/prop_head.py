import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.core import delta2bbox, multiclass_nms, bbox_target, auto_fp16
from mmdet.core import force_fp32
from ..registry import HEADS

import pdb
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from ..utils import ConvModule
from ..builder import build_loss

from pycocotools import mask as cocomask
import cv2


# PyTorch bilinear interpolation is not aligned, which might
# degrade the performance, in particular for small objects
# adapted from https://github.com/Epiphqny/CondInst/issues/1
def aligned_bilinear(tensor, scale_factor):
    assert tensor.dim() == 4
    assert scale_factor >= 1
    assert int(scale_factor) == scale_factor

    if scale_factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = scale_factor * h + 1
    ow = scale_factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(scale_factor // 2, 0, scale_factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


@HEADS.register_module
class PropHead(nn.Module):
    """Propagation head, compute affinity between two feature maps and propagate labels.
    """

    def __init__(self,
                 num_convs=4,
                 conv_kernel_size=3,
                 upsample_ratio=2,
                 num_classes=41,
                 class_agnostic=False,
                 conv_in_channels=256,
                 conv_out_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(PropHead, self).__init__()
        self.convs = nn.Sequential(*[
            nn.Conv2d(4*conv_in_channels, conv_out_channels, 1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out_channels, conv_out_channels, 3, 1, 1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out_channels, conv_out_channels, 1)])

        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.num_convs = num_convs
        self.class_agnostic = class_agnostic
        self.num_classes = num_classes
        self.mask_convs = nn.ModuleList()
        for i in range(self.num_convs):
            padding = (self.conv_kernel_size - 1) // 2
            self.mask_convs.append(
                ConvModule(
                    self.conv_out_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        self.upsample_ratio = upsample_ratio
        self.upsample = nn.ConvTranspose2d(
            self.conv_out_channels,
            self.conv_out_channels,
            self.upsample_ratio,
            stride=self.upsample_ratio)
        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.loss_mask = build_loss(loss_mask)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 1)
        # nn.init.xavier_uniform_(m.weight)
        # nn.init.zeros_(m.bias)

    def batch_get_similarity_matrix(self, ref, target):
        """
        Get pixel-level similarity matrix.
        :param ref: (batchSize, feature_dim, H, W)
        :param target: (batchSize, feature_dim, H, W)
        :return: (batchSize, H*W, H*W)
        """
        (batchSize, feature_dim, H, W) = ref.shape
        ref = ref.permute(0, 2, 3, 1).reshape(batchSize, -1, feature_dim)
        target = target.reshape(batchSize, feature_dim, -1)
        T = ref.bmm(target)
        return T

    def get_spatial_weight(self, shape, sigma):
        """
        Get soft spatial weights for similarity matrix.
        :param shape: (H, W)
        :param sigma:
        :return: (H*W, H*W)
        """
        (H, W) = shape

        index_matrix = torch.arange(
            H * W, dtype=torch.long).reshape(H * W, 1).cuda()
        index_matrix = torch.cat(
            (index_matrix / W, index_matrix % W), -1)  # (H*W, 2)
        d = index_matrix - index_matrix.unsqueeze(1)  # (H*W, H*W, 2)
        d = d.float().pow(2).sum(-1)  # (H*W, H*W)
        w = (- d / sigma ** 2).exp()
        return w

    def inference_predict(self, global_similarity, weight_dense, ref_label):
        """
        Get global prediction.
        :param global_similarity: (batchSize, H*W, H*W)
        :param ref_label: onehot form (batchSize, d, H, W)
        :return: (batchSize, d, H, W)
        """
        num_masks = [ref_masks.shape[0] for ref_masks in ref_label]
        batchSize = len(ref_label)
        d = max(num_masks)
        H = ref_label[0].shape[1]
        W = ref_label[0].shape[2]
        onehot_refmasks = torch.zeros(
            (batchSize, d, H, W), dtype=torch.float).to(
            torch.cuda.current_device())
        for b_id, ref_masks in enumerate(ref_label):
            for m_id, bin_mask in enumerate(ref_masks):
                onehot_refmasks[b_id, m_id] = torch.FloatTensor(bin_mask)
        onehot_refmasks = onehot_refmasks.reshape(batchSize, d, -1)

        global_similarity = global_similarity[0].mul(weight_dense)
        # global_similarity = global_similarity.softmax(dim=0) # need to check
        global_similarity = global_similarity.view(-1, H * W)

        # get prediction
        prediction = onehot_refmasks[0].mm(global_similarity)
        prediction = prediction.reshape(batchSize, d, H, W)

        return prediction

    def extract_feats(self, x):
        x_feats = []
        for idx, x_i in enumerate(x):
            rsz_x = F.interpolate(x_i, scale_factor=float(2**(idx-1)),
                                  mode='bilinear')
            x_feats.append(rsz_x)
        del x
        x_feats = torch.cat(x_feats, 1)
        x_feats = self.convs(x_feats)
        return x_feats

    def predict(self, x_feats, r_feats, ref_bboxes, img_meta, rescale=False):
        img_h, img_w = img_meta[0]['img_shape'][:2]
        scale_factor = img_meta[0]['scale_factor']

        global_similarity = self.batch_get_similarity_matrix(r_feats, x_feats)
        global_similarity = global_similarity.softmax(dim=1)
        weight_dense = self.get_spatial_weight((r_feats.shape[-2:]), 8.0)
        pred_mdict = dict()
        for m_id, (i_id, ref_bbox) in enumerate(ref_bboxes.items()):
            bbox, label = ref_bbox['bbox'], ref_bbox['label']
            if rescale:
                bbox = bbox * scale_factor
            # bbox = self.expand_boxes(torch.tensor(bbox).to(x_feats.device), img_meta)
            # bbox = bbox.cpu().numpy()
            # scaling from imgsize to x1 size
            rsz_box = (bbox//8).astype(np.int)
            rsz_boxes = np.zeros((2, *r_feats.shape[-2:])) # fg + bg
            rsz_boxes[1, rsz_box[1]:rsz_box[3], rsz_box[0]:rsz_box[2]] = 1
            rsz_boxes[0] = 1. - rsz_boxes[1]
            prediction = self.inference_predict(global_similarity, weight_dense, [rsz_boxes])
            prediction = F.interpolate(prediction, scale_factor=8., mode='bilinear')
            # prediction = aligned_bilinear(prediction, scale_factor=8)
            prop_attn = prediction.softmax(1)[:, 1]
            rsz_attn = F.interpolate(prop_attn.unsqueeze(1), size=[*x_feats.shape[-2:]], mode='bilinear')
            rsz_attn = rsz_attn.repeat(1, x_feats.shape[1], 1, 1)
            # apply attention
            x = torch.mul(rsz_attn, x_feats.repeat(rsz_attn.shape[0], 1, 1, 1))
            for conv in self.mask_convs:
                x = conv(x)
            if self.upsample is not None:
                x = self.upsample(x)
                x = self.relu(x)
            mask_pred = self.conv_logits(x)
            mask_pred = F.interpolate(mask_pred, scale_factor=4., mode='bilinear')
            # mask_pred = aligned_bilinear(mask_pred, scale_factor=4)
            # mask_pred = mask_pred[:, :, :img_h, :img_w]
            # mask_pred = F.interpolate(mask_pred, size=[*img_meta[0]['ori_shape'][:2]],
            #                           mode='bilinear')
            pred_mdict[i_id] = mask_pred[:, label+1].sigmoid()
        return pred_mdict

    def batch_global_predict(self, global_similarity, onehot_refmasks):
        """
        Get global prediction.
        :param global_similarity: (batchSize, H*W, H*W)
        :param ref_label: onehot form (batchSize, d, H, W)
        :return: (batchSize, d, H, W)
        """
        batchSize, d, H, W = onehot_refmasks.shape
        onehot_refmasks = onehot_refmasks.reshape(batchSize, d, -1)
        prediction = onehot_refmasks.bmm(global_similarity)
        return prediction.reshape(batchSize, d, H, W)

    @auto_fp16()
    def forward(self, x, ref_x, ref_boxlist):
        # x and ref_x are backbone features of current and reference frame
        # here we compute a correlation matrix of x and ref_x
        x_feats = self.extract_feats(x)
        r_feats = self.extract_feats(ref_x)
        global_similarity = self.batch_get_similarity_matrix(r_feats, x_feats)
        global_similarity = global_similarity.softmax(dim=1)
        del ref_x, r_feats#, x_feats
        max_numbox = np.max(np.array([ref_boxes.shape[0] for ref_boxes in ref_boxlist]))
        rsz_boxes = torch.zeros(
            (len(ref_boxlist), max_numbox, *x[1].shape[2:]), 
            dtype=torch.float, device=global_similarity.device)
        for b_id, ref_boxes in enumerate(ref_boxlist):
            for m_id, ref_box in enumerate(ref_boxes):
                # scaling from imgsize to x1 size
                rsz_box = (ref_box//8).to(torch.int) 
                rsz_boxes[b_id, m_id, rsz_box[1]:rsz_box[3], rsz_box[0]:rsz_box[2]] = 1
            # rsz_boxes[b_id, 0] = 1. - torch.max(rsz_boxes[b_id], 0).values
        fg_pred = self.batch_global_predict(global_similarity, rsz_boxes)
        bg_pred = self.batch_global_predict(global_similarity, 1. - rsz_boxes)
        # prediction = F.interpolate(prediction,
        #                            scale_factor=8., mode='bilinear')
        fg_pred = F.interpolate(fg_pred, scale_factor=8., mode='bilinear')
        bg_pred = F.interpolate(bg_pred, scale_factor=8., mode='bilinear')
        # fg_pred = aligned_bilinear(fg_pred, scale_factor=8)
        # bg_pred = aligned_bilinear(bg_pred, scale_factor=8)
        fg_attnlist = []
        bg_attnlist = []
        attnfeatlist = []
        for b_id, ref_boxes in enumerate(ref_boxlist):
            # Attention output
            fg_boxattns = fg_pred[b_id, :ref_boxes.shape[0]]
            bg_boxattns = bg_pred[b_id, :ref_boxes.shape[0]]
            fg_attnlist.append(fg_boxattns)
            bg_attnlist.append(bg_boxattns)
            # Attention feature
            pred_boxattns = torch.stack([bg_boxattns, fg_boxattns], 1)
            prop_attn = pred_boxattns.softmax(1)[:, 1]
            rsz_attn = F.interpolate(prop_attn.unsqueeze(1), 
                                     size=[*x_feats[b_id].shape[-2:]], mode='bilinear')
            rsz_attn = rsz_attn.repeat(1, x_feats.shape[1], 1, 1)
            # apply attention
            attn_feat = torch.mul(rsz_attn, x_feats[b_id].repeat(rsz_attn.shape[0], 1, 1, 1))
            attnfeatlist.append(attn_feat)
        # Mask prediction
        x = torch.cat(attnfeatlist, 0)
        for conv in self.mask_convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            x = self.relu(x)
        mask_pred = self.conv_logits(x)
        mask_pred = F.interpolate(mask_pred, scale_factor=4., mode='bilinear')
        # mask_pred = aligned_bilinear(mask_pred, scale_factor=4)

        return fg_attnlist, bg_attnlist, mask_pred

    def attn_loss(self, fg_predss, bg_predss, gt_masks, gt_pids, reduce=True):
        # fg_boxattn: N x [C(n) x H x W]
        # pred_boxattns: N x 2 x H x W
        # gt_boxattns: N x H x W
        losses = dict()
        fg_boxattns = []
        bg_boxattns = []
        gt_boxattns = []
        for b_id, (fg_preds, bg_preds) in enumerate(zip(fg_predss, bg_predss)):
            HH, WW = fg_preds.shape[-2:]
            for r_id, (fg_pred, bg_pred) in enumerate(zip(fg_preds, bg_preds)):
                gt_boxattn = torch.zeros((HH, WW), dtype=torch.long, device=fg_pred.device)
                gt_id = (gt_pids[b_id] == (r_id+1)).nonzero()
                if len(gt_id) > 0:
                    gt_id = gt_id.squeeze()
                    rsz_pnts = (gt_masks[b_id][gt_id]).to(torch.int)
                    gt_boxattn[rsz_pnts[1]:rsz_pnts[3], rsz_pnts[0]:rsz_pnts[2]] = 1
                fg_boxattns.append(fg_pred)
                bg_boxattns.append(bg_pred)
                gt_boxattns.append(gt_boxattn)
        fg_boxattns = torch.stack(fg_boxattns, 0)
        bg_boxattns = torch.stack(bg_boxattns, 0)
        gt_boxattns = torch.stack(gt_boxattns, 0)
        pred_boxattns = torch.stack([bg_boxattns, fg_boxattns], 1)
        losses['loss_propattn'] = F.cross_entropy(pred_boxattns, gt_boxattns)
        argmax_pred = torch.argmax(pred_boxattns, 1)
        losses['attn_propacc'] = 100. * (gt_boxattns == argmax_pred).sum().to(torch.float) / float(argmax_pred.numel())
        return losses

    @force_fp32(apply_to=('mask_pred', ))
    def mask_loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_propmask'] = loss_mask
        return loss
    
    def expand_boxes(self, in_rois, img_meta, expand_ratio=.05):
        x_1, x_2 = torch.tensor([-1., 1.]).to(in_rois.device)*expand_ratio*(in_rois[2] - in_rois[0])
        y_1, y_2 = torch.tensor([-1., 1.]).to(in_rois.device)*expand_ratio*(in_rois[3] - in_rois[1])
        in_rois[0] = torch.clamp(in_rois[0] + x_1, 
                                 0, img_meta[0]['img_shape'][1]-1)
        in_rois[1] = torch.clamp(in_rois[1] + y_1, 
                                 0, img_meta[0]['img_shape'][0]-1)
        in_rois[2] = torch.clamp(in_rois[2] + x_2, 
                                 0, img_meta[0]['img_shape'][1]-1)
        in_rois[3] = torch.clamp(in_rois[3] + y_2, 
                                 0, img_meta[0]['img_shape'][0]-1)
        return in_rois
