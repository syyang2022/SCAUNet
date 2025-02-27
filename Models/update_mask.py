import torch
import torch.nn.functional as F
from skimage import measure


def update_gt(pred, gt_masks, size=[512, 512], thretb=0.6, threk=0.6):
    bs, c, feat_h, feat_w = pred.shape
    update_gt_masks = gt_masks.clone()
    background_length = 33
    target_length = 3
    label_image = measure.label((gt_masks[0, 0, :, :] > 0.5).cpu())

    for region in measure.regionprops(label_image, cache=False):
        cur_point_mask = pred.new_zeros(bs, c, feat_h, feat_w)
        cur_point_mask[0, 0, int(region.centroid[0]), int(region.centroid[1])] = 1
        nbr_mask = ((F.conv2d(cur_point_mask,
                                weight=torch.ones(1, 1, background_length, background_length).to(gt_masks.device),
                                stride=1, padding=background_length // 2)) > 0).float()
        targets_mask = ((F.conv2d(cur_point_mask,
                                weight=torch.ones(1, 1, target_length, target_length).to(gt_masks.device),
                                stride=1, padding=target_length // 2)) > 0).float()

        ### Extract Candidate Region
        max_limitation = size[0] * size[1] * 0.0015
        threshold_start = (pred * nbr_mask).max() * thretb
        threshold_delta = threk * ((pred * nbr_mask).max() - threshold_start) * torch.tensor(
                        len(region.coords) / max_limitation).to(gt_masks.device)
        threshold = threshold_start + threshold_delta
        thresh_mask = (pred * nbr_mask > threshold).float()

        ### Eliminate False Alarm
        label_image = measure.label((thresh_mask[0, :, :, :] > 0).cpu())
        if label_image.max() > 1:
            valid_labels = []
            for num_cur in range(1, label_image.max() + 1):
                curr_mask = torch.from_numpy(label_image == num_cur).float().unsqueeze(0).to(gt_masks.device)
                if (curr_mask * targets_mask).sum() > 0:
                    valid_labels.append(num_cur)
            if len(valid_labels) > 0:
                valid_label_mask = torch.tensor(label_image == valid_labels[0]).float().unsqueeze(0).to(gt_masks.device)
                for i in range(1, len(valid_labels)):
                    valid_label_mask += torch.tensor(label_image == valid_labels[i]).float().unsqueeze(0).to(
                        gt_masks.device)
                curr_mask = thresh_mask * valid_label_mask
            else:
                curr_mask = torch.zeros_like(thresh_mask)
            thresh_mask = thresh_mask - curr_mask


        ### Average Weighted Summation
        target_patch = (update_gt_masks * thresh_mask + pred * thresh_mask) / 2
        background_patch = update_gt_masks * (1 - thresh_mask)
        update_gt_masks = background_patch + target_patch

    ### Ensure initial GT point label
    update_gt_masks = torch.max(update_gt_masks, (gt_masks == 1).float())

    return update_gt_masks.clamp(0, 1)


