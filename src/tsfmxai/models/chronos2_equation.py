"""Chronos-2 wrapper with per-channel skeleton and innovation-scale heads."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class Chronos2EquationModel(nn.Module):
    """Add equation/noise heads using Chronos-2 context patch states."""
    def __init__(self, chronos_model: nn.Module, *, skeleton_dimension: int, freeze_forecaster: bool = False):
        super().__init__()
        if skeleton_dimension < 1: raise ValueError('skeleton_dimension must be positive')
        self.chronos=chronos_model
        dimension=int(chronos_model.model_dim)
        self.skeleton_head=nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,skeleton_dimension))
        self.noise_head=nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,1))
        if freeze_forecaster:
            for parameter in self.chronos.parameters(): parameter.requires_grad_(False)

    def forward(self,batch):
        required={'context','future_target','group_ids','num_output_patches'}
        missing=required-set(batch)
        if missing: raise ValueError(f'missing Chronos fields: {sorted(missing)}')
        target=batch['future_target']
        covariates=batch.get('future_covariates')
        if covariates is None: covariates=torch.full_like(target,float('nan'))
        if covariates.shape!=target.shape: raise ValueError('covariate/target shape mismatch')
        encoder = getattr(self.chronos, 'encoder', None)
        chronos_config = getattr(self.chronos, 'chronos_config', None)
        if encoder is None or chronos_config is None:
            raise TypeError('chronos_model must expose the pinned Chronos2Model encoder/config')
        hidden_states = []

        def capture_encoder_output(_module, _inputs, output):
            hidden = getattr(output, 'last_hidden_state', None)
            if hidden is None:
                hidden = output[0]
            hidden_states.append(hidden)

        handle = encoder.register_forward_hook(capture_encoder_output)
        try:
            forecast=self.chronos(context=batch['context'],group_ids=batch['group_ids'],
                future_covariates=covariates,num_output_patches=int(batch['num_output_patches']),future_target=target)
        finally:
            handle.remove()
        if len(hidden_states) != 1:
            raise RuntimeError(f'expected one Chronos encoder pass, observed {len(hidden_states)}')
        hidden = hidden_states[0]
        num_future_patches = int(batch['num_output_patches'])
        num_reg_tokens = int(bool(chronos_config.use_reg_token))
        n_context = hidden.shape[1] - num_future_patches - num_reg_tokens
        if n_context < 1:
            raise RuntimeError('Chronos encoder output has no context patches')
        features=hidden[:,:n_context].mean(dim=1)
        return {'forecast_loss':forecast.loss,'quantile_preds':forecast.quantile_preds,
            'channel_features':features,'skeleton_embedding':self.skeleton_head(features),
            'predicted_log_noise_std':self.noise_head(features).squeeze(-1)}

def equation_training_loss(output,batch,*,embedding_weight=1.0,noise_weight=1.0,normalize_teacher=True):
    """Combine native Chronos quantile loss with skeleton-embedding and log-std losses."""
    forecast=output.get('forecast_loss'); teacher=batch.get('teacher_embedding'); noise=batch.get('noise_std')
    if forecast is None: raise ValueError('Chronos forecast_loss is required')
    if teacher is None or noise is None: raise ValueError('teacher_embedding and noise_std are required')
    if teacher.shape!=output['skeleton_embedding'].shape: raise ValueError('teacher shape mismatch')
    if noise.shape!=output['predicted_log_noise_std'].shape: raise ValueError('noise target shape mismatch')
    pred=output['skeleton_embedding']; teacher=teacher.detach()
    if normalize_teacher:
        teacher=F.normalize(teacher,dim=-1); pred=F.normalize(pred,dim=-1)
    skeleton_loss=F.mse_loss(pred,teacher)
    log_std_target=torch.log(noise.clamp_min(torch.finfo(noise.dtype).eps))
    noise_loss=F.smooth_l1_loss(output['predicted_log_noise_std'],log_std_target)
    total=forecast+embedding_weight*skeleton_loss+noise_weight*noise_loss
    return total,{'forecast':forecast.detach(),'skeleton_embedding':skeleton_loss.detach(),
                  'innovation_log_std':noise_loss.detach()}
