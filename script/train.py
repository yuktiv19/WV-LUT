import logging
import math
import os
import sys
import time
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, "../")
sys.path.insert(0, "./")

from common.option import TrainOptions
from common.architecture import WVLUT
from common.utils import (
    logger_info,
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    net_loss,
    SSIMLoss,
    color_constancy_loss,
    fft_loss,
)
from script.data import LowLightDataProvider, LowLightBenchmark

torch.backends.cudnn.benchmark = True


# ==================== Helper Functions ====================

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].copy_(self.decay * self.shadow[k] + (1.0 - self.decay) * v)

    def apply_shadow(self):
        self.backup = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore(self):
        if self.backup:
            self.model.load_state_dict(self.backup)
            self.backup = {}


def save_checkpoint(model_G: torch.nn.Module, 
                   opt_G: torch.optim.Optimizer, 
                   opt: TrainOptions, 
                   iteration: int, 
                   best: bool = False,
                   suffix: str = "") -> None:
    """Save model and optimizer checkpoints.
    
    Args:
        model_G: Generator model
        opt_G: Generator optimizer
        opt: Training options
        iteration: Current iteration number
        best: Whether this is the best model so far
        suffix: Optional suffix for checkpoint filename
    """
    if best:
        tag = "_best"
    elif suffix:
        tag = f"_{suffix}"
    else:
        tag = ""
    
    # Save model
    torch.save(
        model_G, 
        os.path.join(opt.expDir, f"Model_{iteration:06d}{tag}.pth")
    )
    
    # Save optimizer state
    torch.save(
        opt_G, 
        os.path.join(opt.expDir, f"Opt_{iteration:06d}{tag}.pth")
    )
    
    logger.info(f"Checkpoint saved {iteration}{tag}")

def validate_model(model_G: torch.nn.Module, 
                  valid: LowLightBenchmark, 
                  opt: TrainOptions, 
                  iteration: int) -> float:
    """Validate model on validation datasets.
    
    Args:
        model_G: Generator model
        valid: Validation dataset
        opt: Training options
        iteration: Current iteration number
        
    Returns:
        Average PSNR across validation datasets
    """
    datasets = opt.valDataset
    
    with torch.no_grad():
        model_G.eval()
        
        for dataset_name in datasets:
            psnrs, ssims, lpipss = [], [], []
            files = valid.files[dataset_name]
            
            # Create result directory
            result_path = os.path.join(opt.valoutDir, dataset_name)
            os.makedirs(result_path, exist_ok=True)
            
            for file_name in files:
                key = f"{dataset_name}_{file_name[:-4]}"
                
                # Get ground truth and input images
                lb = valid.images[key]
                input_im = valid.images[key + "_low"]
                
                # Preprocess input
                input_im = input_im.astype(np.float32) / 255.0
                im = torch.Tensor(
                    np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)
                ).cuda()
                
                # Forward pass
                pred = model_G(im) * 255.0
                pred = np.transpose(np.squeeze(pred.data.cpu().numpy(), 0), [1, 2, 0])
                pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)
                
                # Calculate metrics
                psnrs.append(calculate_psnr(lb, pred))
                ssims.append(calculate_ssim(lb, pred))
                lpipss.append(calculate_lpips(lb, pred))
                
                # Save images for first 1000 iterations
                if iteration < 1000:
                    input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(np.uint8)
                    Image.fromarray(input_img).save(
                        os.path.join(result_path, f"{key.split('_')[-1]}_input.png")
                    )
                    Image.fromarray(lb.astype(np.uint8)).save(
                        os.path.join(result_path, f"{key.split('_')[-1]}_gt.png")
                    )
                
                # Save prediction
                Image.fromarray(pred).save(
                    os.path.join(result_path, f"{key.split('_')[-1]}_net.png")
                )
            
            # Log metrics
            avg_psnr = np.mean(np.asarray(psnrs))
            avg_ssim = np.mean(np.asarray(ssims))
            avg_lpips = np.mean(np.asarray(lpipss))
            
            logger.info(
                f"Iter {iteration} | Dataset {dataset_name} | "
                f"AVG Val PSNR: {avg_psnr:02f} | "
                f"AVG Val SSIM: {avg_ssim:02f} | "
                f"AVG Val LPIPS: {avg_lpips:02f}"
            )
            
            writer.add_scalar(f"PSNR_valid/{dataset_name}", avg_psnr, iteration)
            writer.flush()
            
        return avg_psnr

# ==================== Main Training Loop ====================

if __name__ == "__main__":
    # Initialize options and logging
    opt_inst = TrainOptions()
    opt = opt_inst.parse()
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=opt.expDir)
    
    # Setup logging
    logger_name = "train"
    logger_info(logger_name, os.path.join(opt.expDir, f"{logger_name}.log"))
    logger = logging.getLogger(logger_name)
    logger.info(opt_inst.print_options(opt))
    
    # Initialize model
    model_G = WVLUT(
        nf=opt.nf, 
        model_type=opt.model_type, 
        gam_type=getattr(opt, 'gam_type', 'pll_4')
    ).cuda()
    if opt.gpuNum > 1:
        model_G = torch.nn.DataParallel(model_G, device_ids=list(range(opt.gpuNum)))
    
    # Initialize EMA and AMP scaler if enabled
    ema = EMA(model_G) if getattr(opt, 'use_ema', False) else None
    scaler = torch.cuda.amp.GradScaler() if getattr(opt, 'use_amp', False) else None
    
    # Setup optimizer
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(
        params_G,
        lr=opt.lr0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt.weightDecay,
        amsgrad=False,
    )
    
    # Setup learning rate scheduler
    if opt.lr1 < 0:
        lf = lambda x: (((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0) * 0.8 + 0.2
    else:
        lr_b = opt.lr1 / opt.lr0
        lr_a = 1 - lr_b
        lf = lambda x: (((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0) * lr_a + lr_b
    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)
    
    # Initialize datasets
    train_iter = LowLightDataProvider(
        batch_size=opt.batchSize,
        num_workers=opt.workerNum,
        data_path=opt.trainDir,
        patch_size=opt.cropSize
    )
    valid = LowLightBenchmark(opt.valDir, datasets=opt.valDataset)
    
    # Load checkpoint if specified
    highest_psnr = 0.0
    if opt.startIter > 0:
        model_path = os.path.join(opt.expDir, f"Model_{opt.startIter:06d}.pth")
        opt_path = os.path.join(opt.expDir, f"Opt_{opt.startIter:06d}.pth")
        
        model_G.load_state_dict(torch.load(model_path).state_dict(), strict=True)
        opt_G.load_state_dict(torch.load(opt_path).state_dict())
        highest_psnr = validate_model(model_G, valid, opt, opt.startIter)
    
    # Initialize loss criteria
    ssim_criterion = SSIMLoss(window_size=11, channel=3).cuda()
    
    # Training loop
    l_accum = [0.0, 0.0, 0.0]
    dT = rT = 0.0
    accum_samples = 0
    
    for i in range(opt.startIter + 1, opt.totalIter + 1):
        model_G.train()
        
        # Prepare data
        st = time.time()
        im, lb = train_iter.next()
        im, lb = im.cuda(), lb.cuda()
        dT += time.time() - st
        
        # Training step
        st = time.time()
        opt_G.zero_grad()
        
        if getattr(opt, 'use_amp', False):
            with torch.cuda.amp.autocast():
                pred = model_G(im)
                loss_pixel = F.smooth_l1_loss(pred, lb)
                loss_perceptual = 0.04 * net_loss(pred, lb)
                loss_ssim = 0.1 * ssim_criterion(pred, lb)
                loss_freq = 0.01 * fft_loss(pred, lb)
                loss_color = 0.05 * color_constancy_loss(pred)
                loss_G = loss_pixel + loss_perceptual + loss_ssim + loss_freq + loss_color
            
            scaler.scale(loss_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(params_G, max_norm=1.0)
            scaler.step(opt_G)
            scaler.update()
        else:
            pred = model_G(im)
            
            # Multi-objective loss function
            loss_pixel = F.smooth_l1_loss(pred, lb)
            loss_perceptual = 0.04 * net_loss(pred, lb)
            loss_ssim = 0.1 * ssim_criterion(pred, lb)
            loss_freq = 0.01 * fft_loss(pred, lb)
            loss_color = 0.05 * color_constancy_loss(pred)
            loss_G = loss_pixel + loss_perceptual + loss_ssim + loss_freq + loss_color
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, max_norm=1.0)
            opt_G.step()
        
        scheduler.step()
        if ema is not None:
            ema.update()
        
        rT += time.time() - st
        
        # Update monitoring metrics
        accum_samples += opt.batchSize
        l_accum[0] += loss_G.item()
        
        # Display training progress
        if i % opt.displayStep == 0:
            logger.info(
                f"{opt.expDir} | Iter:{i:6d}, Sample:{accum_samples:6d}, "
                f"Loss:{l_accum[0]/opt.displayStep:.2e}, "
                f"dT:{dT/opt.displayStep:.4f}, rT:{rT/opt.displayStep:.4f}"
            )
            l_accum = [0.0, 0.0, 0.0]
            dT = rT = 0.0
        
        # Save checkpoint
        if i % opt.saveStep == 0:
            target_model = model_G.module if opt.gpuNum > 1 else model_G
            save_checkpoint(target_model, opt_G, opt, i)
            if ema is not None:
                ema.apply_shadow()
                save_checkpoint(target_model, opt_G, opt, i, suffix="ema")
                ema.restore()
        
        # Validation
        if i % opt.valStep == 0:
            logger.info(f"lr = {scheduler.get_last_lr()[0]:.8f}")
            target_model = model_G.module if opt.gpuNum > 1 else model_G
            
            psnr = validate_model(target_model, valid, opt, i)
            if psnr > highest_psnr:
                logger.info(f"highest psnr: {psnr:.4f}")
                highest_psnr = psnr
                save_checkpoint(target_model, opt_G, opt, i, best=True)

            if ema is not None:
                ema.apply_shadow()
                psnr_ema = validate_model(target_model, valid, opt, i)
                logger.info(f"EMA psnr: {psnr_ema:.4f}")
                if psnr_ema > highest_psnr:
                    logger.info(f"highest EMA psnr: {psnr_ema:.4f}")
                    highest_psnr = psnr_ema
                    save_checkpoint(target_model, opt_G, opt, i, best=True, suffix="best_ema")
                ema.restore()
        
        # Clear memory
        del im, lb, pred, loss_G
        torch.cuda.empty_cache()
    
    logger.info("Training Complete")

