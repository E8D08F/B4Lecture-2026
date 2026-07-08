# -*- coding: utf-8 -*-
"""
Ex8 B4講義 — Diffusion Model（拡散モデル）
このファイルは DiffusionModel の実装を行うためのコードです。
以下のメソッドの #TODO と記載されている箇所を実装してください:
    - DiffusionModel.q_sample
    - DiffusionModel.p_sample
    - DiffusionModel.training_step
"""

from typing import Any, Dict, Literal

import torch
import torch.nn as nn
from torch import Tensor, randint, randn_like, sqrt
from tqdm import tqdm


def expand(x: Tensor) -> Tensor:
    """Make (B,) coefficients broadcast over (B, C, H, W) images."""
    return x[:, *[None] * 3]


class DiffusionModel(nn.Module):
    """Denoising Diffusion Probabilistic Model (DDPM).

    Args:
        model (nn.Module):                      ノイズ推定モデル（例: UNet2DModel）
        criterion (nn.Module):                  損失関数
        num_timesteps (int):                    拡散過程のタイムステップ数 T
        noise_schedule (Literal["linear"]):     ノイズスケジュールの種類
        noise_schedule_kwargs (Dict[str, Any]): ノイズスケジュールの引数（start, end）

    Functions:
        __init__:                              初期化・α, β, ᾱ バッファを登録（実装済み）
        q_sample(x0, t, noise):               順拡散過程 x_t ~ q(x_t|x_0)
        p_sample(x, t):                       逆拡散過程 x_{t-1} ~ p_θ(x_{t-1}|x_t)
        training_step(images):                1バッチの損失を計算
        generate(num_timesteps, shape):       サンプル生成（実装済み）
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        num_timesteps: int,
        noise_schedule: Literal["linear"],
        noise_schedule_kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the diffusion model."""
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_timesteps = num_timesteps

        if noise_schedule == "linear":
            beta = torch.linspace(
                noise_schedule_kwargs["start"],
                noise_schedule_kwargs["end"],
                num_timesteps,
            )
            alpha = 1.0 - beta
            alpha_prod = alpha.cumprod(dim=0)
            self.register_buffer("beta", beta)
            self.register_buffer("alpha", alpha)
            self.register_buffer("alpha_prod", alpha_prod)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ノイズ推定モデルを呼び出す。（実装済み・変更不要）

        Args:
            x (B, C, H, W): ノイズ付き画像 x_t
            t (B,):          タイムステップ

        Returns:
            noise_pred (B, C, H, W): 推定ノイズ ε_θ(x_t, t)
        """
        return self.model(x, t).sample

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """順拡散過程（Forward process）: x_0 に t ステップ分のノイズを加える。

        Args:
            x0 (B, C, H, W):    クリーン画像 x_0
            t (B,):             タイムステップ
            noise (B, C, H, W): ノイズ ε

        Returns:
            x_t (B, C, H, W): ノイズ付き画像

        参照: "Denoising Diffusion Probabilistic Models" Eq. (4)
        """

        # q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
        alpha_ = expand(self.get_buffer("alpha_prod")[t])
        mean = sqrt(alpha_) * x0
        return mean + sqrt(1 - alpha_) * noise

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """逆拡散過程（Reverse process）: x_t から x_{t-1} を推定する。

        Args:
            x (B, C, H, W): ノイズ付き画像 x_t
            t (B,):          タイムステップ（全要素が同じ値）

        Returns:
            x_prev (B, C, H, W): 1ステップ前の画像 x_{t-1}

        参照: "Denoising Diffusion Probabilistic Models" Algorithm 2, Eq. (11)
        """
        beta = expand(self.get_buffer("beta")[t])
        alpha = expand(self.get_buffer("alpha")[t])
        alpha_ = expand(self.get_buffer("alpha_prod")[t])
        eps = self.forward(x, t)
        mean = (x - beta / sqrt(1 - alpha_) * eps) / sqrt(alpha)

        # Final step: return the denoised estimate
        if t[0].item() == 0:
            return mean
        z = randn_like(x)
        return mean + sqrt(beta) * z

    def training_step(self, images: torch.Tensor) -> torch.Tensor:
        """1バッチの損失を計算する。

        Args:
            images (B, C, H, W): クリーン画像

        Returns:
            loss (scalar): 訓練損失

        参照: "Denoising Diffusion Probabilistic Models" Algorithm 1
        """
        x0 = images
        batch_size = x0.size(0)
        t = randint(0, self.num_timesteps, (batch_size,), device=x0.device)
        noise = randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_pred = self.forward(xt, t)
        loss = self.criterion(noise_pred, noise)
        return loss

    def generate(self, num_timesteps: int, shape: tuple) -> torch.Tensor:
        """サンプルを生成する。（実装済み・変更不要）"""
        device = self.get_buffer("alpha").device
        x = torch.randn(shape, device=device)
        for step in tqdm(range(num_timesteps - 1, -1, -1)):
            t = torch.full((x.size(0),), step, dtype=torch.long, device=device)
            x = self.p_sample(x, t)
        return x
