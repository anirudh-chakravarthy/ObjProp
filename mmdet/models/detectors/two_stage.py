import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.core import bbox_overlaps, bbox2result_with_id
from mmdet.core import multiclass_nms
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin

import numpy as np
import pycocotools.mask as cocomask


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, #BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 track_head=None,
                 prop_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if track_head is not None:
            self.track_head = builder.build_head(track_head)
        if prop_head is not None:
            self.prop_head = builder.build_head(prop_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.prev_bboxes = None
        self.prev_roi_feats = None
        self.prev_vid = -1
        self.history_len = 24
        self.x_history = []
        self.mask_history = []
        self.bbox_history = []

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_track:
            self.track_head.init_weights()
        if self.with_prop:
            self.prop_head.apply(self.prop_head.weights_init)
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore,
                      ref_img, # images of reference frame
                      ref_bboxes, # gt bbox of reference frame
                      gt_pids, # gt ids of current frame bbox mapped to reference frame
                      gt_masks=None,
                      ref_masks=None,
                      proposals=None):
        x = self.extract_feat(img)
        ref_x = self.extract_feat(ref_img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        
        # Propagation head
        if self.with_prop:
            # Inter-frame Box Attention
            # fg_boxattnss: N x [C(n) x H x W]
            fg_boxattnss, bg_boxattnss, mask_pred = self.prop_head(
                x[:self.bbox_roi_extractor.num_inputs],
                ref_x[:self.bbox_roi_extractor.num_inputs],
                ref_bboxes)
            loss_prop = self.prop_head.attn_loss(fg_boxattnss, bg_boxattnss, gt_bboxes, gt_pids)
            losses.update(loss_prop)

            gt_masklist = []
            pred_masklist = []
            gt_labellist = []
            cnt_i = 0
            for b_id, ref_bs in enumerate(ref_bboxes):
                for r_id in range(ref_bs.shape[0]):
                    gt_id = (gt_pids[b_id] == (r_id+1)).nonzero()
                    if len(gt_id) > 0:
                        gt_id = gt_id.squeeze()
                        pred_masklist.append(mask_pred[cnt_i])
                        gt_mask = torch.zeros(*img.shape[-2:], dtype=torch.float, device=img.device)
                        gt_orig = torch.tensor(gt_masks[b_id][gt_id], dtype=torch.float, device=img.device)
                        gt_mask[:gt_orig.shape[0], :gt_orig.shape[1]] = gt_orig
                        gt_masklist.append(gt_mask)
                        gt_labellist.append(gt_labels[b_id][gt_id])
                    cnt_i += 1
            if len(pred_masklist) > 0:
                pred_masklist = torch.stack(pred_masklist, 0)
                gt_masklist = torch.stack(gt_masklist, 0)
                gt_labellist = torch.stack(gt_labellist, 0)
                loss_propmask = self.prop_head.mask_loss(pred_masklist, gt_masklist, gt_labellist)
            else:
                loss_propmask = dict()
                loss_propmask['loss_propmask'] = torch.zeros(1, device=img.device)
            losses.update(loss_propmask)

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i], 
                                                     gt_pids[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    gt_pids[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_img_n = [res.bboxes.size(0) for res in sampling_results]
            ref_rois = bbox2roi(ref_bboxes)
            ref_bbox_img_n = [x.size(0) for x in ref_bboxes]
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            ref_bbox_feats = self.bbox_roi_extractor(
                ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets, (ids, id_weights) = self.bbox_head.get_target(sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

            match_score = self.track_head(bbox_feats, ref_bbox_feats,
                                          bbox_img_n, ref_bbox_img_n)
            loss_match = self.track_head.loss(match_score,
                                              ids, id_weights)
            losses.update(loss_match)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)

        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        is_first = img_meta[0]['is_first']

        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)

        if det_bboxes.nelement()==0:
            det_obj_ids=np.array([], dtype=np.int64)
            if is_first:
                self.prev_bboxes =  None
                self.prev_roi_feats = None
                self.prev_det_labels = None
            return det_bboxes, det_labels, det_obj_ids

        res_det_bboxes = det_bboxes.clone()
        if rescale:
            res_det_bboxes[:, :4] *= scale_factor

        det_rois = bbox2roi([res_det_bboxes])
        det_roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], det_rois)

        # recompute bbox match feature
        if is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = np.arange(det_bboxes.size(0))
            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = det_roi_feats
            self.prev_det_labels = det_labels

        else:
            assert self.prev_roi_feats is not None

            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)]
            prev_bbox_img_n = [self.prev_roi_feats.size(0)]
            match_score = self.track_head(det_roi_feats, self.prev_roi_feats,
                                          bbox_img_n, prev_bbox_img_n, test_mode=True)[0]
            match_logprob = F.log_softmax(match_score, dim=1)
            label_delta = (self.prev_det_labels == det_labels.view(-1,1)).float()
            bbox_ious = bbox_overlaps(det_bboxes[:,:4], self.prev_bboxes[:,:4])

            # compute comprehensive score 
            comp_scores = self.track_head.compute_comp_scores(match_logprob, 
                det_bboxes[:,4].view(-1, 1),
                bbox_ious,
                label_delta,
                add_bbox_dummy=True)

            match_likelihood, match_ids = torch.max(comp_scores, dim =1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object, 
            # add tracking features/bboxes of new object
            match_ids = match_ids.cpu().numpy().astype(np.int32)
            det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1)
            best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)

            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    # # add detection only from mask rcnn (not from prop)
                    # if idx < num_dets:
                    # add new object
                    det_obj_ids[idx] = self.prev_roi_feats.size(0)
                    self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                    self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                    self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)

                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score 
                    obj_id = match_id - 1
                    match_score = comp_scores[idx, match_id]
                    if match_score > best_match_scores[obj_id]:
                        det_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        # update feature
                        self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                        self.prev_bboxes[obj_id] = det_bboxes[idx]

        return det_bboxes, det_labels, det_obj_ids
    
    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        assert self.with_track, "Track head must be implemented"
        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        if self.prev_vid != img_meta[0]['video_id']:
            self.x_history = []
            self.mask_history = []
            self.bbox_history = []
            self.prev_vid = img_meta[0]['video_id']

        det_bboxes, det_labels, det_obj_ids = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

        bbox_results = bbox2result_with_id(det_bboxes, det_labels, det_obj_ids)


        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels,
                rescale=rescale, det_obj_ids=det_obj_ids)

            # Propagation head
            x_feats = self.prop_head.extract_feats(
                x[:self.bbox_roi_extractor.num_inputs])

            if len(self.x_history) > 0:
                prop_mdict = dict()
                # for r_feats, r_masks in zip(self.x_history, self.mask_history):
                for r_feats, r_boxes in zip(self.x_history, self.bbox_history):
                    if type(r_boxes) == list:
                        continue
                    if img_meta[0]['video_id'] == 6:
                        pdb.set_trace()
                    else:
                        continue
                    pred_mdict = self.prop_head.predict(
                        x_feats, r_feats, r_boxes, img_meta, rescale=rescale)
                    for i_id, pred_mask in pred_mdict.items():
                        if i_id in prop_mdict:
                            prop_mdict[i_id] = torch.cat(
                                (prop_mdict[i_id], pred_mask), 0)
                        else:
                            prop_mdict[i_id] = pred_mask
                if len(prop_mdict) > 0:
                    blend_masks = dict()
                    for i_id, prop_masks in prop_mdict.items():
                        ''' compute mean propagation first and blend it to the 
                        raw prediction '''
                        prop_mask = prop_masks.mean(0).detach().cpu().numpy()
                        if i_id in segm_results:
                            continue
                        else:
                            prop_mask = (prop_mask >= .3).astype(np.float)
                            if np.sum(prop_mask) == 0:
                                continue

                            blend_masks[i_id] = prop_mask
                            avg_score = 0.
                            num_bbox = 0.
                            class_label = -1
                            for bbox_dict in self.bbox_history:
                                if i_id in bbox_dict:
                                    avg_score += bbox_dict[i_id]['bbox'][-1]
                                    num_bbox += 1.
                                    class_label = bbox_dict[i_id]['label']
                            avg_score /= num_bbox
                            y_pts, x_pts = np.where(prop_mask)
                            bbox_results[i_id] = {'bbox': np.array([x_pts.min(), y_pts.min(), x_pts.max(), y_pts.max(), avg_score], dtype='float32'), 'label': class_label}

                    for i_id, blend_mask in blend_masks.items():
                        mask_fortran = np.asfortranarray(
                            blend_mask.astype(np.uint8))
                        mask_rle = cocomask.encode(mask_fortran)
                        if type(segm_results) == list:
                            segm_results = dict()
                        segm_results[i_id] = mask_rle

            # update history
            if len(self.x_history) == self.history_len:
                del self.x_history[0]
                del self.mask_history[0]
                del self.bbox_history[0]
            self.x_history.append(x_feats)
            self.mask_history.append(segm_results)
            self.bbox_history.append(bbox_results)
            return bbox_results, segm_results
    
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
    
    def _compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta):
        bbox_iou_dummy =  torch.zeros(bbox_ious.size(0), 1, device=torch.cuda.current_device()) 
        bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
        label_dummy = torch.ones(bbox_ious.size(0), 1, device=torch.cuda.current_device())
        label_delta = torch.cat((label_dummy, label_delta),dim=1)
        
        return match_ll + self.match_coeff[0] * torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious + self.match_coeff[2] * label_delta
    
    def _get_match_score(self, x, mem, dummy_coeff=0):
        prod = []
        for i in range(len(mem)):
            t = torch.stack(mem[i], dim = 0).t()
            prod += [torch.max(torch.mm(x, t), dim = 1)[0]]
        
        prod = torch.stack(prod, dim = 0).t()
        dummy = torch.ones(prod.size(0), 1, device=torch.cuda.current_device())*dummy_coeff
        match_score = torch.cat([dummy, prod], dim=1)
        return match_score