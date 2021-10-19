import torch
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss
from detectron2.distiller_zoo import *


def rpn_loss(pred_objectness_logits, pred_anchor_deltas, prev_pred_objectness_logits, prev_pred_anchor_deltas):
    loss = logit_distillation(pred_objectness_logits[0], prev_pred_objectness_logits[0])
    loss += anchor_delta_distillation(pred_anchor_deltas[0], prev_pred_anchor_deltas[0])
    return {"loss_dist_rpn": loss}


def backbone_loss(features, prev_features):
    loss = feature_distillation(features['res4'], prev_features['res4'])
    return {"loss_dist_backbone": loss}


def roi_head_loss(pred_class_logits, pred_proposal_deltas, prev_pred_class_logits, prev_pred_proposal_deltas, dist_loss_weight=0.5):
    loss = logit_distillation(pred_class_logits, prev_pred_class_logits)
    # loss = feature_distillation(pred_class_logits, prev_pred_class_logits)
    loss += anchor_delta_distillation(pred_proposal_deltas, prev_pred_proposal_deltas)
    return {"loss_dist_roi_head": dist_loss_weight * loss}


def logit_distillation(current_logits, prev_logits, T=6.0):
    p = F.log_softmax(current_logits / T, dim=1)
    q = F.softmax(prev_logits / T, dim=1)
    kl_div = torch.sum(F.kl_div(p, q, reduction='none').clamp(min=0.0) * (T**2)) / current_logits.shape[0]
    return kl_div


def anchor_delta_distillation(current_delta, prev_delta):
    # return smooth_l1_loss(current_delta, prev_delta, beta=0.1, reduction='mean')
    return F.mse_loss(current_delta, prev_delta)


def feature_distillation(features, prev_features):
    # return smooth_l1_loss(features, prev_features, beta=0.1, reduction='mean')
    return F.mse_loss(features, prev_features)
    # criterion_kd = Attention()
    # # criterion_kd = NSTLoss()
    # # criterion_kd = DistillKL(T=4)
    # # criterion_kd = FactorTransfer()
    # loss = criterion_kd(features, prev_features)
    # loss = torch.stack(loss, dim=0).sum()
    # return loss
