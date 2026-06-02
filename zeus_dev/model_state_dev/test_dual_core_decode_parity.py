"""dual-core index-K vs 单核 paged:多步 decode logits 对拍。
对拍点 = logits(topk 之前),避免 topk 边界翻转误判。
Run: $PY -m pytest test_dual_core_decode_parity.py -v
"""
import os
import sys

# dev_dsa_attn 内部用裸 `import _common`. pytest 从 model_state_dev/ 收集本文件时
# glm5next_modules/ 不在 sys.path 上, 手动补一下让 `$PY -m pytest test_*.py` 直接可跑.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "glm5next_modules"))

import torch
import dev_glm5next_dsa_decode_test as dsa
from glm5next_modules.dev_dsa_attn import Glm5NextDsaAttn


def test_dual_core_vs_single_core_logits_parity():
    B, S_hist, n_steps = 2, 1000, 3
    page_size_dc, P = 512, 4
    page_size_ref = S_hist + n_steps + 1          # 1004: degenerate single big page

    # 同 which+seed => 权重相同,无需复制
    attn_ref = Glm5NextDsaAttn(which="16b", seed=42)
    attn_dc = Glm5NextDsaAttn(which="16b", seed=42)
    cfg = attn_ref.cfg

    hist_ref = dsa.init_history(cfg, B, S_hist, seed=1)
    hist_dc = dsa.init_history(cfg, B, S_hist, seed=1)
    st_ref = attn_ref.init_paged_state(hist_ref, page_size=page_size_ref)
    st_dc = attn_dc.init_paged_state(hist_dc, page_size=page_size_dc,
                                     num_physical_pages=P, dual_core=True)

    for step in range(n_steps):
        torch.manual_seed(100 + step)
        hidden = torch.randn(B, cfg.H, dtype=torch.bfloat16)
        lg_ref = attn_ref.forward_zeus_logits(hidden.clone().to("zeus"), st_ref)
        lg_dc = attn_dc.forward_zeus_logits(hidden.clone().to("zeus"), st_dc)
        seq_len = S_hist + step + 1               # valid logical positions [0, seq_len)
        ref = lg_ref[:, :seq_len].cpu().float()
        dc = lg_dc[:, :seq_len].cpu().float()
        # 只比两边都有效的位置(padding sentinel 不参与)
        valid = (ref > -1e29) & (dc > -1e29)
        assert valid.any(), f"step {step}: no valid overlap"
        torch.testing.assert_close(dc[valid], ref[valid], rtol=2e-3, atol=2e-3)
