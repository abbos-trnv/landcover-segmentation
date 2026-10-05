"""Continue a legacy best checkpoint; subsequent interruptions restore training state.

Each invocation writes a new child run. Source files are never overwritten.
Only load trusted checkpoints produced by this project (weights_only=False).
"""
import argparse
import os
import shutil
import subprocess

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import LandCoverTileDataset
from .models import create_model
from .run_utils import create_run_dir, load_config, save_config, save_json, set_seed
from .train import train_one_epoch, evaluate
from .transforms import training_transform, validation_transform


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--additional-epochs', type=int, default=15,
                        help='Legacy recovery only; a full checkpoint retains its existing target.')
    args = parser.parse_args()
    if args.additional_epochs < 1:
        raise ValueError('additional-epochs must be positive')
    source = args.source_run
    config = load_config(source / 'config.yaml')
    set_seed(config['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU before continuing training.')
    device = torch.device('cuda')
    full = (source / 'last_checkpoint.pth').exists()
    checkpoint = torch.load(source / ('last_checkpoint.pth' if full else 'best_checkpoint.pth'),
                            map_location='cpu', weights_only=False)
    epoch = int(checkpoint['epoch'])
    source_history = pd.read_csv(source / 'history.csv')
    history = checkpoint['history'] if full else source_history.loc[source_history.epoch <= epoch].to_dict('records')
    if not history or int(history[-1]['epoch']) != epoch:
        raise ValueError('Checkpoint epoch missing from history; inspect source run.')
    model = create_model(config['architecture'], config['encoder_name'], config['encoder_weights'], config['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # LR in history is recorded AFTER scheduler.step: it is the next epoch's LR.
    lr = float(history[-1]['learning_rate'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    if full:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best = checkpoint['best_score']
        best_payload = checkpoint['best_checkpoint']
        stale = checkpoint['no_improvement']
        target = checkpoint['target_epoch']
        recovery = checkpoint['recovery']
    else:
        # Scheduler is deterministic given validation history; replay its state.
        for row in history:
            scheduler.step(float(row['val_foreground_miou']))
        if abs(optimizer.param_groups[0]['lr'] - lr) > 1e-12:
            raise ValueError('Reconstructed scheduler LR differs from saved history.')
        best = float(history[-1]['val_foreground_miou'])
        if best < max(float(row['val_foreground_miou']) for row in history) - 1e-8:
            raise ValueError('Source best checkpoint does not match validation history.')
        best_payload, stale = checkpoint, 0
        target = epoch + args.additional_epochs
        recovery = {'original_source': str(source), 'restored_epoch': epoch,
                    'original_last_logged_epoch': int(source_history.epoch.max()),
                    'optimizer_reset': True, 'scheduler_replayed': True,
                    'additional_epoch_limit': args.additional_epochs}
    if epoch >= target or stale >= config['early_stopping_patience']:
        raise RuntimeError('Training already reached its stopping condition; evaluate the saved best checkpoint.')

    index = pd.read_csv(source / 'tile_index.csv')
    loaders, datasets = {}, {}
    for split in ('train', 'val'):
        transform = training_transform() if split == 'train' else validation_transform()
        datasets[split] = LandCoverTileDataset(index.query('split == @split').reset_index(drop=True),
            args.data_root / 'images', args.data_root / 'masks', config['tile_size'], transform)
        loaders[split] = DataLoader(datasets[split], batch_size=config['batch_size'],
            shuffle=split == 'train', num_workers=config['num_workers'], pin_memory=True,
            persistent_workers=False)
    run = create_run_dir(source.parent, config['experiment_name'] + '_continued')
    save_config(config, run)
    shutil.copy2(source / 'tile_index.csv', run / 'tile_index.csv')
    shutil.copy2(source / 'history.csv', run / 'source_history.csv')
    save_json(recovery, run / 'recovery.json')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save_json({'code_revision': revision, 'device': str(device), 'gpu': torch.cuda.get_device_name(0),
               'torch': torch.__version__, 'parent_run': str(source)}, run / 'environment.json')
    count = sum(p.numel() for p in model.parameters())
    save_json({'parameters': count, 'parameters_m': count / 1e6}, run / 'model.json')

    def persist(current_epoch):
        # One atomic bundle is authoritative; CSV and best file can be regenerated.
        atomic_save({'epoch': current_epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_checkpoint': best_payload, 'best_score': best, 'no_improvement': stale,
            'target_epoch': target, 'history': history, 'config': config, 'recovery': recovery}, run / 'last_checkpoint.pth')
        atomic_save(best_payload, run / 'best_checkpoint.pth')
        pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
        save_json({'best_val_foreground_miou': best, 'epochs_completed': len(history),
                   'last_epoch': current_epoch, 'target_epoch': target, 'recovery': recovery}, run / 'summary.json')

    persist(epoch)
    print(f'Continuation run: {run}\nRestored epoch {epoch}; next {epoch + 1}; target {target}; LR {lr:g}', flush=True)
    print('Legacy optimizer was reset. Future restarts restore optimizer/scheduler. Test is not run.', flush=True)
    for current in range(epoch + 1, target + 1):
        # Epoch seeds and fresh workers make boundary recovery independent of previous RNG consumption.
        set_seed(config['seed'] + current)
        for dataset in datasets.values():
            transform = getattr(dataset, 'transform', None)
            if hasattr(transform, 'set_random_seed'):
                transform.set_random_seed(config['seed'] + current)
        train = train_one_epoch(model, loaders['train'], optimizer, device, config['num_classes'], config['ignore_index'])
        val, _ = evaluate(model, loaders['val'], device, config['num_classes'], config['ignore_index'])
        scheduler.step(val['foreground_miou'])
        row = {'epoch': current, 'learning_rate': optimizer.param_groups[0]['lr']}
        row.update({f'train_{k}': v for k, v in train.items()})
        row.update({f'val_{k}': v for k, v in val.items()})
        history.append(row)
        if val['foreground_miou'] > best:
            best, stale = val['foreground_miou'], 0
            best_payload = {'epoch': current, 'config': config,
                'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        else:
            stale += 1
        persist(current)
        print(f'Epoch {current:02d} | train loss={train["loss"]:.4f} | val fg-mIoU={val["foreground_miou"]:.4f} | best={best:.4f}', flush=True)
        if stale >= config['early_stopping_patience']:
            print('Early stopping.', flush=True)
            break
    print(f'Training completed. Evaluate best checkpoint in: {run}', flush=True)


if __name__ == '__main__':
    main()
