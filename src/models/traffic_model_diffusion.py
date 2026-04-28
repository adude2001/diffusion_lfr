import torch
from torch import nn

from models.diffusion_models import FullyConditionedUnet
from models.diffusion import GaussianDiffusionConditional
from models.traffic_model import TrafficModel
from utils.transforms import transform2frame


class DiffusionTrafficModel(TrafficModel):
    def __init__(self, npast, nfuture, map_obs_size_pix, nclasses,
                 map_feat_size=64,
                 past_feat_size=64,
                 future_feat_size=64,
                 latent_size=32,
                 output_bicycle=True,
                 traj_encoder='mlp',
                 conv_channel_in=4,
                 conv_kernel_list=[7, 5, 5, 3, 3, 3],
                 conv_stride_list=[2, 2, 2, 2, 2, 2],
                 conv_filter_list=[16, 32, 64, 64, 128, 128],
                 diffusion_dim=128,
                 diffusion_timesteps=100,
                 diffusion_sampling_timesteps=20,
                 diffusion_objective='pred_v',
                 diffusion_cond_drop_prob=0.1,
                 diffusion_guidance_scale=1.0):
        super().__init__(
            npast=npast,
            nfuture=nfuture,
            map_obs_size_pix=map_obs_size_pix,
            nclasses=nclasses,
            map_feat_size=map_feat_size,
            past_feat_size=past_feat_size,
            future_feat_size=future_feat_size,
            latent_size=latent_size,
            output_bicycle=output_bicycle,
            traj_encoder=traj_encoder,
            conv_channel_in=conv_channel_in,
            conv_kernel_list=conv_kernel_list,
            conv_stride_list=conv_stride_list,
            conv_filter_list=conv_filter_list,
        )
        self.model_family = 'diffusion'

        self.channels = 4
        self.cond_dim = self.past_feat_size + self.map_feat_out_size + self.NC + self.att_feat_size + self.z_size

        self.denoiser = FullyConditionedUnet(
            dim=diffusion_dim,
            channels=self.channels,
            cond_dim=self.cond_dim,
            cond_drop_prob=diffusion_cond_drop_prob,
            dim_mults=(1, 2, 4),
        )
        self.diffusion = GaussianDiffusionConditional(
            model=self.denoiser,
            seq_length=nfuture,
            classifier_free_guidance=True,
            timesteps=diffusion_timesteps,
            sampling_timesteps=diffusion_sampling_timesteps,
            objective=diffusion_objective,
            cfg_drop_prob=diffusion_cond_drop_prob,
            cfg_guidance_scale=diffusion_guidance_scale,
            auto_normalize=False,
        )

    def set_bicycle_params(self, bicycle_params):
        self.bicycle_params = bicycle_params

    def _encode_condition_inputs(self, scene_graph, map_idx, map_env, include_posterior=False, use_posterior_z=False):
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env)
        past_feat = self.encode_past(scene_graph)
        prior_mu, prior_var = self.prior(scene_graph, map_feat, past_feat)

        posterior_out = None
        z_cond = self.rsample(prior_mu, prior_var)
        if include_posterior and 'future' in scene_graph:
            future_feat = self.encode_future(scene_graph)
            post_mu, post_var = self.encoder(scene_graph, map_feat, past_feat, future_feat)
            posterior_out = (post_mu, post_var)
            if use_posterior_z:
                z_cond = self.rsample(post_mu, post_var)

        cond = torch.cat([past_feat, map_feat, scene_graph.sem, scene_graph.lw, z_cond], dim=-1)
        return cond, map_feat, past_feat, (prior_mu, prior_var), posterior_out, z_cond

    def _build_inits(self, scene_graph):
        return scene_graph.past[:, -1, :4]

    def _future_target(self, scene_graph):
        if 'future_gt' in scene_graph:
            future = scene_graph.future_gt
        else:
            future = scene_graph.future
        future = future[:, :, :4].contiguous()
        inits = self._build_inits(scene_graph)
        return transform2frame(inits, future)

    def _sample_future(self, cond, inits):
        samp = self.diffusion.sample(cond=cond, inits=inits, cond_scale=1.0, rescaled_phi=0.0)
        future_local = samp.transpose(1, 2).contiguous()
        future_local = torch.nan_to_num(future_local, nan=0.0, posinf=1e3, neginf=-1e3)
        return transform2frame(inits, future_local, inverse=True)

    def forward(self, scene_graph, map_idx, map_env,
                use_post_mean=False,
                future_sample=False):
        cond, map_feat, past_feat, prior_out, posterior_out, _ = self._encode_condition_inputs(
            scene_graph,
            map_idx,
            map_env,
            include_posterior=('future' in scene_graph),
            use_posterior_z=('future' in scene_graph)
        )
        inits = self._build_inits(scene_graph)
        target = self._future_target(scene_graph)
        target_diff = target.transpose(1, 2).contiguous()
        future_mask = scene_graph.future_vis if 'future_vis' in scene_graph else None

        diffusion_loss = self.diffusion(target_diff, cond, inits, mask=future_mask)

        if posterior_out is not None and (use_post_mean or (not self.training)):
            z_eval = posterior_out[0]
        elif posterior_out is not None:
            z_eval = self.rsample(posterior_out[0], posterior_out[1])
        else:
            z_eval = self.rsample(prior_out[0], prior_out[1])

        cond_eval = torch.cat([past_feat, map_feat, scene_graph.sem, scene_graph.lw, z_eval], dim=-1)
        future_pred = self._sample_future(cond_eval, inits)

        out = {
            'model_family': 'diffusion',
            'future_pred': future_pred,
            'diffusion_loss': diffusion_loss,
            'prior_out': prior_out,
            'posterior_out': posterior_out,
        }
        if future_sample:
            z_prior = self.rsample(prior_out[0], prior_out[1])
            cond_prior = torch.cat([past_feat, map_feat, scene_graph.sem, scene_graph.lw, z_prior], dim=-1)
            out['future_samp'] = self._sample_future(cond_prior, inits)
        return out

    def reconstruct(self, scene_graph, map_idx, map_env):
        cond, _, _, prior_out, posterior_out, _ = self._encode_condition_inputs(
            scene_graph,
            map_idx,
            map_env,
            include_posterior=('future' in scene_graph),
            use_posterior_z=('future' in scene_graph)
        )
        inits = self._build_inits(scene_graph)
        future_pred = self._sample_future(cond, inits)
        return {
            'model_family': 'diffusion',
            'future_pred': future_pred,
            'posterior_out': posterior_out,
            'prior_out': prior_out,
        }

    def sample_batched(self, scene_graph, map_idx, map_env, num_samples,
                        include_mean=False,
                        nfuture=None):
        cond, _, _, prior_out, _, _ = self._encode_condition_inputs(
            scene_graph,
            map_idx,
            map_env,
            include_posterior=False,
            use_posterior_z=False
        )
        inits = self._build_inits(scene_graph)
        fut_list = []
        for _ in range(num_samples):
            fut_list.append(self._sample_future(cond, inits))
        future_pred = torch.stack(fut_list, dim=1)
        return {
            'model_family': 'diffusion',
            'future_pred': future_pred,
            'z_samp': None,
            'z_logprob': None,
            'z_mdist': None,
            'prior_out': prior_out,
            'posterior_out': None,
        }

    def embed(self, scene_graph, map_idx, map_env):
        cond, map_feat, past_feat, prior_out, posterior_out, z_cond = self._encode_condition_inputs(
            scene_graph,
            map_idx,
            map_env,
            include_posterior=('future' in scene_graph),
            use_posterior_z=('future' in scene_graph)
        )
        inits = self._build_inits(scene_graph)
        NA = cond.size(0)
        opt_init = torch.randn((NA, self.channels, self.FT), device=cond.device)
        return {
            'model_family': 'diffusion',
            'cond': cond,
            'inits': inits,
            'map_feat': map_feat,
            'past_feat': past_feat,
            'opt_init': opt_init,
            'z_cond': z_cond,
            'prior_out': prior_out,
            'posterior_out': posterior_out,
        }

    def decode_embedding(self, z, embed_out, scene_graph, map_idx, map_env,
                        ext_future=None,
                        nfuture=None):
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(-1, self.channels, -1)
        t = torch.zeros((z.size(0),), dtype=torch.long, device=z.device)
        pred = self.diffusion.model_predictions(z, t, embed_out['cond'], embed_out['inits']).pred_x_start
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)
        future_local = pred.transpose(1, 2).contiguous()
        future_pred = transform2frame(embed_out['inits'], future_local, inverse=True)
        if ext_future is not None:
            ego_inds = scene_graph.ptr[:-1]
            future_pred = future_pred.clone()
            future_pred[ego_inds] = ext_future
        return {'future_pred': future_pred, 'model_family': 'diffusion'}
