"""init_paged_state 的 dual-core small-page 分支几何自检。
Run: $PY -m pytest glm5next_modules/test_dual_core_init.py -v
"""
import os
import sys

# 本文件在 glm5next_modules/ 内, pytest "prepend" 模式只把本目录放进 sys.path,
# 而 dev_glm5next_dsa_decode_test 在上一级 model_state_dev/. 手动补一下父目录,
# 让 `$PY -m pytest glm5next_modules/test_dual_core_init.py` 直接可跑.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import dev_glm5next_dsa_decode_test as dsa
from dev_dsa_attn import Glm5NextDsaAttn


def test_dual_core_init_geometry():
    attn = Glm5NextDsaAttn(which="16b", seed=42)
    cfg = attn.cfg
    B, seqlen, page_size, P = 2, 1024, 512, 4
    history = dsa.init_history(cfg, B, seqlen, seed=1)
    st = attn.init_paged_state(history, page_size=page_size,
                               num_physical_pages=P)
    assert P % 2 == 0
    assert st["page_size"] == page_size
    assert st["num_physical_pages"] == P
    assert st["body_cache_c0"].shape == (P // 2, page_size, cfg.Di)
    assert st["body_cache_c1"].shape == (P // 2, page_size, cfg.Di)
    assert st["scale_cache"].shape == (P * page_size,)
    pages_per_seq = -(-seqlen // page_size)        # ceil
    assert pages_per_seq >= 2, "config must exercise multi-page-per-seq"
    assert st["block_table_host"].shape[1] >= pages_per_seq
    # 同源自检:每个有效 (b, lp_logical) 的 pp 唯一且在 [0,P)
    bt = st["block_table_host"]
    seen = set()
    for b in range(B):
        for lp in range(pages_per_seq):
            pp = int(bt[b, lp])
            assert 0 <= pp < P
            assert pp not in seen, "physical page reused — allocation not unique"
            seen.add(pp)
    # gather buffers 仍在(下游 #9 需要)
    for k in ("k_local_c0", "k_local_c1", "k_local_t_c0", "k_local_t_c1"):
        assert k in st


def test_dual_core_init_rejects_underprovisioned_odd_pages():
    """奇数 pages_per_seq 下 per-core 容量不足必须报错,而不是静默页冲突。"""
    attn = Glm5NextDsaAttn(which="16b", seed=42)
    cfg = attn.cfg
    B, seqlen, page_size, P = 2, 1100, 512, 6   # pages_per_seq=3; core0 需 4 页 > P/2=3
    history = dsa.init_history(cfg, B, seqlen, seed=1)
    with pytest.raises(AssertionError):
        attn.init_paged_state(history, page_size=page_size,
                              num_physical_pages=P)


def test_dual_core_init_odd_pages_no_collision():
    """奇数 pages_per_seq 且容量充足时,分配无物理页冲突。"""
    attn = Glm5NextDsaAttn(which="16b", seed=42)
    cfg = attn.cfg
    B, seqlen, page_size, P = 2, 1100, 512, 8   # pages_per_seq=3; core0 需 4<=P/2=4
    history = dsa.init_history(cfg, B, seqlen, seed=1)
    st = attn.init_paged_state(history, page_size=page_size,
                               num_physical_pages=P)
    bt = st["block_table_host"]
    pages_per_seq = -(-seqlen // page_size)
    seen = set()
    for b in range(B):
        for lp in range(pages_per_seq):
            pp = int(bt[b, lp])
            assert 0 <= pp < P
            assert pp not in seen, "physical page reused"
            seen.add(pp)
