from models.traffic_model import TrafficModel
from models.traffic_model_diffusion import DiffusionTrafficModel


def build_traffic_model(cfg, map_env, num_categories):
    common_kwargs = dict(
        map_feat_size=cfg.map_feat_size,
        past_feat_size=cfg.past_feat_size,
        future_feat_size=cfg.future_feat_size,
        latent_size=cfg.latent_size,
        output_bicycle=cfg.model_output_bicycle,
        conv_channel_in=map_env.num_layers,
        conv_kernel_list=cfg.conv_kernel_list,
        conv_stride_list=cfg.conv_stride_list,
        conv_filter_list=cfg.conv_filter_list,
    )

    if getattr(cfg, 'model_family', 'cvae') == 'diffusion':
        return DiffusionTrafficModel(
            cfg.past_len,
            cfg.future_len,
            cfg.map_obs_size_pix,
            num_categories,
            diffusion_timesteps=getattr(cfg, 'diff_timesteps', 100),
            diffusion_sampling_timesteps=getattr(cfg, 'diff_sampling_timesteps', 20),
            diffusion_objective=getattr(cfg, 'diff_objective', 'pred_v'),
            diffusion_cond_drop_prob=getattr(cfg, 'cfg_drop_prob', 0.1),
            diffusion_guidance_scale=getattr(cfg, 'cfg_guidance_scale', 1.0),
            **common_kwargs,
        )

    return TrafficModel(
        cfg.past_len,
        cfg.future_len,
        cfg.map_obs_size_pix,
        num_categories,
        **common_kwargs,
    )
