"""dual-core forward_zeus 端到端跑通(不验数值,只验不抛异常)。
Run: $PY -m pytest test_dual_core_forward_smoke.py -v
"""
import os
import sys

# dev_dsa_attn 内部用裸 `import _common` (脚本以 `python glm5next_modules/dev_*.py`
# 跑时 glm5next_modules/ 自动进 sys.path[0]). pytest 从 model_state_dev/ 收集本文件
# 时该目录不在 path 上, 故手动补一下, 让 `$PY -m pytest test_*.py` 直接可跑.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "glm5next_modules"))

import torch
import dev_glm5next_dsa_decode_test as dsa
from glm5next_modules.dev_dsa_attn import Glm5NextDsaAttn


def test_dual_core_forward_runs():
    attn = Glm5NextDsaAttn(which="16b", seed=42)
    cfg = attn.cfg
    # 几何须给 decode 留位: history 摊进 pages_per_seq=ceil(S/PS) 页, 后续
    # decode 写在 pos S, S+1, ... 必须仍落在已分配页内. S=600/PS=512 →
    # pages_per_seq=2 (真实跨双核), 写在 pos 600+ → page 1, 界内.
    B, seqlen, page_size, P = 2, 600, 512, 4
    st = attn.init_paged_state(dsa.init_history(cfg, B, seqlen, seed=1),
                               page_size=page_size, num_physical_pages=P)
    for step in range(2):                       # multi-step: seq_lens grows
        torch.manual_seed(100 + step)
        hidden = torch.randn(B, cfg.H, dtype=torch.bfloat16).to("zeus")
        out = attn.forward_zeus(hidden, st)
        assert out.shape == (B, cfg.H)
        # logits-only path also runs
        lg = attn.forward_zeus_logits(hidden, st)
        assert lg.shape[0] == B
