#############################
#       Import Modules      #
#############################
# Local module imports
from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map

# Standard library imports
import argparse
from copy import deepcopy
import os
import pickle
import logging
import yaml

# Third-party imports
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm

#############################
#       Global Settings     #
#############################
MODE = None
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.benchmark = True

#############################
#       Argument Parsing    #
#############################
def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # Basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'], default='deeplabv3plus')
    parser.add_argument("--resume-from", type=str, default=None, help="Path to resume model from")

    # Semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)

    # Retraining specific arguments
    parser.add_argument('--reliable-id-path', type=str)

    # Checkpoint resume argument
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')

    args = parser.parse_args()
  
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    return args

#############################
#   Loss and Utility Func   #
#############################
def compute_pixel_confidence(prediction):
    """
    Computes pixel-wise confidence using the softmax of the predictions.
    """
    probs = F.softmax(prediction, dim=1)
    return torch.max(probs, dim=1).values

def dynamic_loss_scaling(pseudo_label_reliability, base_weight=0.5, scale=2.0):
    """
    Scales the loss based on the reliability of pseudo-labels.
    """
    return base_weight + scale * pseudo_label_reliability

def weighted_cross_entropy_loss(prediction, pseudo_labels, confidence, gamma=1.0):
    """
    Computes a weighted cross-entropy loss using pixel confidence.
    """
    ce_loss = F.cross_entropy(prediction, pseudo_labels, reduction='none')
    return (confidence ** gamma * ce_loss).mean()

def decay_confidence(confidence, decay_factor=0.9, use_confidence_decay=False):
    """
    Decays the confidence values over time if enabled.
    """
    return confidence * decay_factor if use_confidence_decay else confidence

def dynamic_thresholding(confidence, base_threshold=0.6, beta=0.5, min_threshold=0.3, max_threshold=0.8):
    """
    Computes a dynamic threshold for pixel confidence.
    Uses torch.clamp to restrict the threshold within bounds.
    """
    avg_conf = confidence.mean()
    threshold = base_threshold / (1 + torch.exp(-beta * (avg_conf - 0.5)))
    return torch.clamp(threshold, min=min_threshold, max=max_threshold)

def detect_boundaries(labels):
    """
    Detects boundaries in segmentation masks using Sobel filters.
    Returns a binary boundary mask.
    """
    num_classes = labels.max().item() + 1
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=labels.device, dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=labels.device, dtype=torch.float32)
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)

    edges_x = F.conv2d(one_hot_labels, sobel_x, padding=1, groups=num_classes)
    edges_y = F.conv2d(one_hot_labels, sobel_y, padding=1, groups=num_classes)
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    boundary_mask = edges.sum(dim=1) > 0

    return boundary_mask.float()

def boundary_loss(pred, pseudo_labels, confidence, boundary_mask, gamma=1.0):
    """
    Computes a boundary-aware loss by combining weighted cross-entropy loss and an additional loss on boundary regions.
    """
    base_loss = weighted_cross_entropy_loss(pred, pseudo_labels, confidence, gamma)
    boundary_mask = boundary_mask.unsqueeze(1).expand_as(pred)
    masked_pred = pred * boundary_mask
    boundary_ce_loss = F.cross_entropy(masked_pred, pseudo_labels, reduction='none')
    boundary_ce_loss = (boundary_ce_loss * boundary_mask[:, 0]).mean()
    return base_loss + 0.5 * boundary_ce_loss

#############################
#  Training and Validation  #
#############################
def train_with_confidence_weighted_learning(model, teacher_model, train_loader, optimizer, criterion, 
                                            gamma=1.0, decay_factor=0.9, base_threshold=0.6, beta=0.5, 
                                            device='cuda', use_confidence_decay=False):
    """
    Training loop with confidence-weighted learning.
    The teacher model generates pseudo-labels, which are refined using boundary detection,
    dynamic thresholding, and confidence decay.
    """
    model.train()
    teacher_model.eval()
    best_model, checkpoints = None, []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Generate pseudo-labels and compute confidence scores using the teacher model
        with torch.no_grad():
            teacher_logits = teacher_model(images)
            pseudo_labels = torch.argmax(teacher_logits, dim=1)
            pixel_confidence = compute_pixel_confidence(teacher_logits)

        # Detect boundaries and apply dynamic thresholding
        boundary_mask = detect_boundaries(pseudo_labels)
        threshold = dynamic_thresholding(pixel_confidence, base_threshold, beta)
        mask = pixel_confidence > threshold
        pseudo_labels *= mask.long()

        logits = model(images)
        loss = boundary_loss(logits, pseudo_labels, pixel_confidence, boundary_mask, gamma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optionally decay the confidence values
        pixel_confidence = decay_confidence(pixel_confidence, decay_factor, use_confidence_decay)

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    return best_model, checkpoints

def train(model, trainloader, valloader, criterion, optimizer, args, device, accumulation_steps=4):
    """
    General training loop with gradient accumulation and learning rate scheduling.
    Validates and saves the best model based on mIoU.
    """
    iters, best_miou = 0, 0.0
    total_iters = len(trainloader) * args.epochs
    best_model = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        print(f"\n==> Epoch {epoch + 1}/{args.epochs}")

        for i, (img, mask) in enumerate(tqdm(trainloader)):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(trainloader):.4f}")

        current_miou = validate_and_checkpoint(model, valloader, criterion, optimizer, epoch, best_miou, args, device)
        if current_miou > best_miou:
            best_miou = current_miou
            best_model = deepcopy(model)

    return best_model if best_model else model

def validate_and_checkpoint(model, valloader, criterion, optimizer, epoch, best_miou, args, device):
    """
    Validates the model on the validation set, computes the mean IoU,
    and saves a checkpoint if a new best is achieved.
    """
    model.eval()
    metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(valloader, desc="Validating"):
            if len(batch) == 2:
                img, mask = batch
            elif len(batch) >= 3:
                img, mask, *_ = batch

            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            loss = criterion(logits, mask)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            metric.add_batch(preds.cpu().numpy(), mask.cpu().numpy())

    mIOU = metric.evaluate()[-1] * 100
    print(f"Validation mIoU: {mIOU:.2f}%")

    if mIOU > best_miou:
        save_checkpoint(epoch, model, optimizer, mIOU, os.path.join(args.save_path, "best_checkpoint.pth"))
        print(f"New best mIoU: {mIOU:.2f}%")
        return mIOU

    return best_miou

#############################
#   Checkpoint Functions    #
#############################
def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a checkpoint and updates the model and optimizer.
    Adjusts keys if DataParallel is used.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    
    print(f"Loading checkpoint from '{checkpoint_path}'")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        print("Checkpoint loaded with weights_only=True")
    except (TypeError, pickle.UnpicklingError):
        print("weights_only=True failed. Re-loading with weights_only=False.")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_state_dict = model.state_dict()
    updated_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not next(iter(model_state_dict)).startswith('module.'):
            updated_state_dict[k[len('module.'):]] = v
        elif not k.startswith('module.') and next(iter(model_state_dict)).startswith('module.'):
            updated_state_dict['module.' + k] = v
        else:
            updated_state_dict[k] = v

    model.load_state_dict(updated_state_dict, strict=False)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_miou = checkpoint.get('best_miou', 0.0)
    print(f"Resumed training from epoch {epoch}, best mIoU: {best_miou:.2f}")
    return epoch + 1, best_miou

def save_checkpoint(epoch, model, optimizer, mIOU, path):
    """
    Saves a checkpoint containing the model state, optimizer state, and best mIoU.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mIOU': mIOU
    }
    torch.save(state, path)
    print(f"Checkpoint saved at epoch {epoch}, mIOU: {mIOU:.2f}")

#############################
#   Retraining Pipeline     #
#############################
def select_reliable(models, dataloader, args):
    """
    Selects reliable image IDs based on pairwise IoU between model predictions.
    Splits the IDs into reliable and unreliable files.
    """
    os.makedirs(args.reliable_id_path, exist_ok=True)
    for model in models:
        model.eval()
    id_to_reliability = []

    with torch.no_grad():
        for img, mask, img_id in tqdm(dataloader, desc="Selecting reliable IDs"):
            img = img.cuda()
            preds = [torch.argmax(model(img), dim=1).cpu().numpy() for model in models]
            mious = []
            for i in range(len(preds) - 1):
                metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
                metric.add_batch(preds[i], preds[-1])
                mious.append(metric.evaluate()[-1])
            reliability = sum(mious) / len(mious) if mious else 0.0
            id_to_reliability.append((img_id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    half = len(id_to_reliability) // 2
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:half]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[half:]:
            f.write(elem[0] + '\n')

def label(model, dataloader, args):
    """
    Generates pseudo-labels for images using the trained model and saves them.
    """
    model.eval()
    tbar = tqdm(dataloader, desc="Generating Pseudo-Labels")
    metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    cmap = color_map(args.dataset)
    os.makedirs(args.pseudo_mask_path, exist_ok=True)

    with torch.no_grad():
        for img, mask, img_id in tbar:
            img = img.cuda()
            pred = model(img)
            pred_labels = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred_labels.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1] * 100

            pred_image = Image.fromarray(pred_labels.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred_image.putpalette(cmap)
            save_path = os.path.join(args.pseudo_mask_path, os.path.basename(img_id[0].split(' ')[1]))
            pred_image.save(save_path)

            tbar.set_description(f'Generating pseudo-labels - mIoU: {mIOU:.2f}%')

def retrain_on_pseudo_labeled_data(args, unlabeled_path, best_model, device, valloader, criterion):
    """
    Retrains the model on a mix of labeled and pseudo-labeled data.
    """
    MODE = 'semi_train'
    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, unlabeled_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args, device)
    best_model = train(model, trainloader, valloader, CrossEntropyLoss(ignore_index=255), optimizer, args, device)
    return best_model

def pseudo_label_and_retrain(args, best_model, device, valloader, criterion):
   
    # Stage A: Select Reliable IDs
    print('\n================> A: Select Reliable IDs')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    select_reliable([best_model], dataloader, args)

    # Stage B: Pseudo Label Reliable Images
    print('\n================> B: Pseudo Label Reliable Images')
    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    label(best_model, dataloader, args)

    # Stage C: Pseudo Label Reliable Images
    print('\n================> C: Pseudo Label Reliable Images')
    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    label(best_model, dataloader, args)

#############################
#  Model Initialization     #
#############################
def init_basic_elems(args, device):
    """
    Initializes the segmentation model and optimizer.
    Loads a pretrained backbone if applicable and converts BatchNorm layers for multi-GPU training.
    """
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, 21 if args.dataset == 'pascal' else 19)
    
    # Convert BatchNorm to SyncBatchNorm for better multi-GPU training
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Load pretrained weights for the backbone if available
    if args.model == 'deeplabv3plus' and args.backbone in ['resnet50', 'resnet101']:
        backbone_state_dict = torch.load(f'pretrained/{args.backbone}.pth')
        model.backbone.load_state_dict(backbone_state_dict, strict=False)

    optimizer = SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': args.lr * 10.0}
    ], lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    return model, optimizer

#############################
#        Main Routine       #
#############################
if __name__ == '__main__':
    args = parse_args()

    # Set default parameters based on the dataset
    dataset_defaults = {
        'pascal': {'epochs': 80, 'lr': 0.001, 'crop_size': 321},
        'cityscapes': {'epochs': 240, 'lr': 0.004, 'crop_size': 721}
    }
    defaults = dataset_defaults.get(args.dataset, {})
    args.epochs = args.epochs or defaults.get('epochs')
    args.lr = args.lr or (defaults.get('lr') / 16 * args.batch_size)
    args.crop_size = args.crop_size or defaults.get('crop_size')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Arguments: {args}")

    # Create the save directory if it does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    criterion = CrossEntropyLoss(ignore_index=255).to(device)

    # Initialize datasets and dataloaders
    trainset = SemiDataset(args.dataset, args.data_root, 'train', args.crop_size, args.labeled_id_path)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=min(8, os.cpu_count()), pin_memory=True, drop_last=True
    )
    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(
        valset, batch_size=4 if args.dataset == 'cityscapes' else 1, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )

    # Initialize the model and optimizer
    model, optimizer = init_basic_elems(args, device)
    print(f"\nModel parameters: {count_params(model):.1f}M")

    start_epoch, best_miou = 0, 0.0
    if args.resume:
        start_epoch, best_miou = load_checkpoint(args.resume, model, optimizer)
        print(f"Resuming from epoch {start_epoch}, best mIoU: {best_miou:.2f}")

    # Training loop using confidence-weighted learning
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_with_confidence_weighted_learning(
            model, deepcopy(model), trainloader, optimizer, criterion,
            gamma=args.gamma, decay_factor=args.decay_factor, base_threshold=args.base_threshold,
            beta=args.beta, device=device, use_confidence_decay=args.use_confidence_decay
        )
        print(f"Completed epoch {epoch + 1}")
        best_miou = validate_and_checkpoint(model, valloader, criterion, optimizer, epoch, best_miou, args, device)

    print("Training completed. Evaluating final performance...")
    best_miou = validate_and_checkpoint(model, valloader, criterion, optimizer, args.epochs, best_miou, args, device)

