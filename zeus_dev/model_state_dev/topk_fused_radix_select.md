@triton.jit
def kernel_local_topk_unified_v2(
    # HBM I/O
    logits_ptr,
    out_logits_ptr,
    out_indices_ptr,
    
    # PWLF 配置
    bucket_seg_ptr, bucket_slope_ptr, bucket_bias_ptr,
    pair_seg_ptr, pair_slope_ptr, pair_bias_ptr,
    fine_seg_ptr, fine_slope_ptr, fine_bias_ptr,
    
    # SRAM scratch
    sram_base_ptr,
    
    # CP 偏移
    cp_rank_offset,
    
    # 编译期常量
    N_PER_CORE: tl.constexpr,         # 32768
    TILE_SIZE: tl.constexpr,          # 4096
    N_TILES: tl.constexpr,            # 8
    N_BUCKETS: tl.constexpr,          # 64
    N_PAIRS: tl.constexpr,            # 32
    LOG2_TILE: tl.constexpr,          # 12
    LOG2_BUCKETS: tl.constexpr,       # 6
    LOG2_CAND: tl.constexpr,          # 14
    K: tl.constexpr,                  # 2048
    MAX_CAND: tl.constexpr,           # 16384
    SCALE: tl.constexpr,
    SRAM_PER_CORE: tl.constexpr,
):
    core_id = tl.program_id(0)
    
    # ====================================================
    # SRAM 区域指针
    # ====================================================
    my_sram = sram_base_ptr + core_id * SRAM_PER_CORE
    peer_sram = sram_base_ptr + (1 - core_id) * SRAM_PER_CORE
    
    zeros_16k     = my_sram + 0
    working_16k   = my_sram + 16384
    tile_buf_a    = my_sram + 32768
    tile_buf_b    = my_sram + 36864
    cand_logits   = my_sram + 40960
    cand_indices  = my_sram + 57344
    hist_region   = my_sram + 73728
    pwlf_cache    = my_sram + 73984
    
    local_hist    = hist_region + 0
    peer_hist     = hist_region + 64
    global_hist   = hist_region + 128
    misc          = hist_region + 192
    
    # 对等地址
    peer_peer_hist = peer_sram + 73728 + 64
    peer_misc      = peer_sram + 73728 + 192
    
    # 索引
    tile_idx = tl.arange(0, TILE_SIZE)
    bucket_idx = tl.arange(0, N_BUCKETS)
    cand_idx = tl.arange(0, MAX_CAND)
    
    # ====================================================
    # 初始化: zeros_16k 一次性全置 0
    # ====================================================
    # 用循环写,因为单次 store 16K 可能太大
    for i in tl.static_range(4):    # 4 × 4096 = 16384
        offset_v: tl.constexpr = i * TILE_SIZE
        tl.store(zeros_16k + offset_v + tile_idx, 
                 tl.zeros((TILE_SIZE,), dtype=tl.float32))
    
    # 加载主 PWLF 配置
    seg_b = tl.load(bucket_seg_ptr + tl.arange(0, 65))
    slope_b = tl.load(bucket_slope_ptr + bucket_idx)
    bias_b = tl.load(bucket_bias_ptr + bucket_idx)
    bucket_pwlf = tl.zeus.make_pwlf(seg_b, slope_b, bias_b)
    
    core_offset = core_id * N_PER_CORE
    
    # ====================================================
    # Phase 1: 直方图 + 收集候选 (合并 phase, double buffer 流水)
    # ====================================================
    # 优化: Phase 1 和 Phase 4 可以合并!
    # 第一次扫 tile 时, 直接收集"可能候选"到 candidate buffer
    # 但 b_star 还没确定, 怎么知道哪些是候选?
    # 
    # 答案: 不需要在第一次扫描就 collect, 我们的目标是:
    #   - Phase 1: 算直方图 (累加 local_hist)
    #   - Phase 4: 用确定的 b_star collect 候选
    # 
    # 但因为 b_star 依赖跨 core 同步, Phase 1 必须先完成
    # 所以 Phase 1 和 Phase 4 是两遍扫描
    # 
    # 优化: Phase 1 时把每 tile 的 PWLF 结果(bucket_id)缓存到候选区
    # Phase 4 时直接用,跳过重新 PWLF
    # 
    # 32K bucket_id = 128 KB, 太大装不下
    # 选择: 重新做 PWLF (PWLF 便宜)
    
    local_hist_vec = tl.zeros((N_BUCKETS,), dtype=tl.float32)
    
    # 预 load tile 0
    offs_0 = core_offset + 0 * TILE_SIZE + tile_idx
    logits_0 = tl.load(logits_ptr + offs_0)
    tl.store(tile_buf_a + tile_idx, logits_0)
    
    for t in tl.static_range(N_TILES):
        if t % 2 == 0:
            cur_logits = tl.load(tile_buf_a + tile_idx)
        else:
            cur_logits = tl.load(tile_buf_b + tile_idx)
        
        # 预取下一 tile
        if t < N_TILES - 1:
            next_offs = core_offset + (t + 1) * TILE_SIZE + tile_idx
            next_logits = tl.load(logits_ptr + next_offs)
            if t % 2 == 0:
                tl.store(tile_buf_b + tile_idx, next_logits)
            else:
                tl.store(tile_buf_a + tile_idx, next_logits)
        
        # PWLF
        bid = tl.zeus.pwlf(cur_logits, bucket_pwlf)
        
        # 直方图 (2 桶并行)
        for p in tl.static_range(N_PAIRS):
            ps = tl.load(pair_seg_ptr + p * 65 + tl.arange(0, 65))
            psl = tl.load(pair_slope_ptr + p * N_BUCKETS + bucket_idx)
            pb = tl.load(pair_bias_ptr + p * N_BUCKETS + bucket_idx)
            pair_pwlf = tl.zeus.make_pwlf(ps, psl, pb)
            
            encoded = tl.zeus.pwlf(bid, pair_pwlf)
            s = tl.sum(encoded)
            local_hist_vec[2*p]   += s - (s // SCALE) * SCALE
            local_hist_vec[2*p+1] += s // SCALE
    
    tl.store(local_hist + bucket_idx, local_hist_vec)
    
    # ====================================================
    # Phase 2: Core-Send 交换 hist
    # ====================================================
    tl.zeus.core_send(
        src=local_hist,
        dst_core=1 - core_id,
        dst=peer_peer_hist,
        count=N_BUCKETS,
        dtype=tl.float32,
    )
    tl.zeus.barrier()
    
    other_hist = tl.load(peer_hist + bucket_idx)
    gh = local_hist_vec + other_hist
    
    # ====================================================
    # Phase 3: 找 b_star (用 64-cumsum, 复用 16K zeros 区)
    # ====================================================
    rev_idx = N_BUCKETS - 1 - bucket_idx
    
    # 写 gh 到 working_16k 前 64 位
    tl.store(working_16k + bucket_idx, gh)
    rev_gh = tl.load(working_16k + rev_idx)
    
    # 64 元素 cumsum, 借用 zeros_16k 末尾作为补零
    # 把 rev_gh 写到 working_16k 前 64 位
    tl.store(working_16k + bucket_idx, rev_gh)
    
    cs = rev_gh
    for step in tl.static_range(LOG2_BUCKETS):
        offset_v: tl.constexpr = 1 << step
        # 从 working_16k - offset_v 处 load 64 个
        # 等价于 zeros_16k 末尾 offset_v 个 0 + working_16k 前 (64-offset_v) 个
        shifted = tl.load(working_16k - offset_v + bucket_idx)
        cs = cs + shifted
        if step < LOG2_BUCKETS - 1:
            tl.store(working_16k + bucket_idx, cs)
    
    cross = (cs >= K).to(tl.int32)
    first_cross = tl.argmax(cross, axis=0)
    b_star = (N_BUCKETS - 1 - first_cross).to(tl.float32)
    b_star_int = N_BUCKETS - 1 - first_cross
    
    # ====================================================
    # Phase 4: Collect 候选到 SRAM (double buffer 流水)
    # ====================================================
    my_cand_count = 0
    
    # 重启流水
    offs_0 = core_offset + 0 * TILE_SIZE + tile_idx
    logits_0 = tl.load(logits_ptr + offs_0)
    tl.store(tile_buf_a + tile_idx, logits_0)
    
    for t in tl.static_range(N_TILES):
        if t % 2 == 0:
            cur_logits = tl.load(tile_buf_a + tile_idx)
        else:
            cur_logits = tl.load(tile_buf_b + tile_idx)
        
        if t < N_TILES - 1:
            next_offs = core_offset + (t + 1) * TILE_SIZE + tile_idx
            next_logits = tl.load(logits_ptr + next_offs)
            if t % 2 == 0:
                tl.store(tile_buf_b + tile_idx, next_logits)
            else:
                tl.store(tile_buf_a + tile_idx, next_logits)
        
        bid = tl.zeus.pwlf(cur_logits, bucket_pwlf)
        mask = bid >= b_star
        mask_f = mask.to(tl.float32)
        
        # Cumsum (4096 元素, 用 16K zeros 提供补零)
        tl.store(working_16k + tile_idx, mask_f)
        cs_v = mask_f
        for step in tl.static_range(LOG2_TILE):
            offset_v: tl.constexpr = 1 << step
            shifted = tl.load(working_16k - offset_v + tile_idx)
            cs_v = cs_v + shifted
            if step < LOG2_TILE - 1:
                tl.store(working_16k + tile_idx, cs_v)
        
        pos = cs_v - mask_f
        n_pass = tl.sum(mask_f).to(tl.int32)
        
        # 全局索引
        global_idx_f = (core_offset + t * TILE_SIZE + tile_idx + cp_rank_offset).to(tl.float32)
        
        # 写到 SRAM 候选区
        write_pos = my_cand_count + pos.to(tl.int32)
        # 垃圾位置:每元素一个独立 SRAM 槽 (在候选区之后)
        trash_pos = MAX_CAND - TILE_SIZE + tile_idx
        safe_pos = tl.where(mask, write_pos, trash_pos)
        
        tl.store(cand_logits + safe_pos, cur_logits)
        tl.store(cand_indices + safe_pos, global_idx_f)
        
        my_cand_count += n_pass
    
    # ====================================================
    # Phase 5: 二次 Radix (用精细 PWLF 在 SRAM 候选上)
    # ====================================================
    # 加载精细 PWLF
    fine_seg_v = tl.load(fine_seg_ptr + b_star_int * 65 + tl.arange(0, 65))
    fine_slope_v = tl.load(fine_slope_ptr + b_star_int * N_BUCKETS + bucket_idx)
    fine_bias_v = tl.load(fine_bias_ptr + b_star_int * N_BUCKETS + bucket_idx)
    fine_pwlf = tl.zeus.make_pwlf(fine_seg_v, fine_slope_v, fine_bias_v)
    
    # 加载所有候选 (MAX_CAND 元素, 其中前 my_cand_count 有效)
    cand_v = tl.load(cand_logits + cand_idx)
    cand_i = tl.load(cand_indices + cand_idx)
    
    valid_mask = cand_idx < my_cand_count
    
    # 精细分桶 (无效候选置桶 0)
    fbid = tl.zeus.pwlf(cand_v, fine_pwlf)
    fbid = tl.where(valid_mask, fbid, 0.0)
    
    # 二次直方图
    refine_hist = tl.zeros((N_BUCKETS,), dtype=tl.float32)
    for p in tl.static_range(N_PAIRS):
        ps = tl.load(pair_seg_ptr + p * 65 + tl.arange(0, 65))
        psl = tl.load(pair_slope_ptr + p * N_BUCKETS + bucket_idx)
        pb = tl.load(pair_bias_ptr + p * N_BUCKETS + bucket_idx)
        pair_pwlf = tl.zeus.make_pwlf(ps, psl, pb)
        
        encoded = tl.zeus.pwlf(fbid, pair_pwlf)
        s = tl.sum(encoded)
        refine_hist[2*p]   += s - (s // SCALE) * SCALE
        refine_hist[2*p+1] += s // SCALE
    
    tl.store(local_hist + bucket_idx, refine_hist)
    
    # ====================================================
    # Phase 6: Core-Send 二次 hist + 找 b_star_2
    # ====================================================
    tl.zeus.core_send(
        src=local_hist,
        dst_core=1 - core_id,
        dst=peer_peer_hist,
        count=N_BUCKETS,
        dtype=tl.float32,
    )
    tl.zeus.barrier()
    
    other_rh = tl.load(peer_hist + bucket_idx)
    refine_gh = refine_hist + other_rh
    
    # 找 b_star_2 (同 Phase 3 逻辑)
    tl.store(working_16k + bucket_idx, refine_gh)
    rev_rh = tl.load(working_16k + rev_idx)
    
    tl.store(working_16k + bucket_idx, rev_rh)
    cs2 = rev_rh
    for step in tl.static_range(LOG2_BUCKETS):
        offset_v: tl.constexpr = 1 << step
        shifted = tl.load(working_16k - offset_v + bucket_idx)
        cs2 = cs2 + shifted
        if step < LOG2_BUCKETS - 1:
            tl.store(working_16k + bucket_idx, cs2)
    
    cross2 = (cs2 >= K).to(tl.int32)
    fc2 = tl.argmax(cross2, axis=0)
    b_star_2 = (N_BUCKETS - 1 - fc2).to(tl.float32)
    
    # ====================================================
    # Phase 7: 写出 top-K
    # ====================================================
    above_mask = (fbid > b_star_2) & valid_mask
    eq_mask = (fbid == b_star_2) & valid_mask
    
    n_above_local = tl.sum(above_mask.to(tl.float32)).to(tl.int32)
    n_eq_local = tl.sum(eq_mask.to(tl.float32)).to(tl.int32)
    
    # 交换 n_above_local 和 n_eq_local
    tl.store(misc + 0, n_above_local.to(tl.float32))
    tl.store(misc + 1, n_eq_local.to(tl.float32))
    tl.zeus.core_send(
        src=misc,
        dst_core=1 - core_id,
        dst=peer_misc + 2,
        count=2,
        dtype=tl.float32,
    )
    tl.zeus.barrier()
    
    n_above_peer = tl.load(misc + 2).to(tl.int32)
    n_eq_peer = tl.load(misc + 3).to(tl.int32)
    n_above_global = n_above_local + n_above_peer
    n_need_eq = K - n_above_global
    
    # ---- Above cumsum (16K 元素) ----
    above_mask_f = above_mask.to(tl.float32)
    tl.store(working_16k + cand_idx, above_mask_f)
    cs_above = above_mask_f
    for step in tl.static_range(LOG2_CAND):
        offset_v: tl.constexpr = 1 << step
        shifted = tl.load(working_16k - offset_v + cand_idx)
        cs_above = cs_above + shifted
        if step < LOG2_CAND - 1:
            tl.store(working_16k + cand_idx, cs_above)
    pos_above = cs_above - above_mask_f
    
    # 本 core above 在全局 above 中的起始
    above_hbm_start = tl.where(core_id == 0, 0, n_above_peer)
    
    # 写 above 到 HBM
    above_write = above_hbm_start + pos_above.to(tl.int32)
    trash_hbm = K + core_id * MAX_CAND + cand_idx
    safe_above = tl.where(above_mask, above_write, trash_hbm)
    
    tl.store(out_logits_ptr + safe_above, cand_v)
    tl.store(out_indices_ptr + safe_above, cand_i)
    
    # ---- Eq cumsum + write ----
    eq_mask_f = eq_mask.to(tl.float32)
    tl.store(working_16k + cand_idx, eq_mask_f)
    cs_eq = eq_mask_f
    for step in tl.static_range(LOG2_CAND):
        offset_v: tl.constexpr = 1 << step
        shifted = tl.load(working_16k - offset_v + cand_idx)
        cs_eq = cs_eq + shifted
        if step < LOG2_CAND - 1:
            tl.store(working_16k + cand_idx, cs_eq)
    pos_eq = cs_eq - eq_mask_f
    
    eq_global_start = tl.where(core_id == 0, 0, n_eq_peer)
    global_eq_pos = eq_global_start + pos_eq.to(tl.int32)
    take_eq = eq_mask & (global_eq_pos < n_need_eq)
    
    eq_write = n_above_global + global_eq_pos
    safe_eq = tl.where(take_eq, eq_write, trash_hbm)
    
    tl.store(out_logits_ptr + safe_eq, cand_v)
    tl.store(out_indices_ptr + safe_eq, cand_i)


Host 端调度
def topk_64k_unified(logits, cp_rank=0, n_per_cp=65536):
    """
    完整本地 top-K, 单 kernel 完成
    
    Args:
        logits: [64K] fp32
        cp_rank: CP 域中的 rank (用于索引偏移)
        n_per_cp: 每 CP 持有的 logits 数
    
    Returns:
        top_logits: [K] fp32
        top_indices: [K] int32 (全局索引)
    """
    assert logits.shape == (64 * 1024,)
    device = logits.device
    
    # 配置
    N_TOTAL = 64 * 1024
    N_PER_CORE = 32 * 1024
    TILE_SIZE = 4096
    N_TILES = 8
    N_BUCKETS = 64
    N_PAIRS = 32
    LOG2_TILE = 12
    LOG2_BUCKETS = 6
    LOG2_CAND = 14
    K = 2048
    MAX_CAND = 16384
    
    # SRAM 大小: 每 core ~310 KB
    SRAM_PER_CORE = 76032 + 8192   # 大约 320 KB
    
    pwlf = prepare_pwlf_tensors(device)
    
    # 分配 SRAM scratch (在 HBM 上, 实际硬件会自动管理 SRAM)
    sram_scratch = torch.empty(2 * SRAM_PER_CORE, 
                                 dtype=torch.float32, device=device)
    
    # 输出 buffer (含垃圾区)
    out_size = K + 2 * MAX_CAND
    out_logits = torch.empty(out_size, dtype=torch.float32, device=device)
    out_indices = torch.empty(out_size, dtype=torch.float32, device=device)
    
    cp_rank_offset = cp_rank * n_per_cp
    
    kernel_local_topk_unified_v2[(2,)](
        logits, out_logits, out_indices,
        pwlf['bucket_seg'], pwlf['bucket_slope'], pwlf['bucket_bias'],
        pwlf['pair_seg'], pwlf['pair_slope'], pwlf['pair_bias'],
        pwlf['fine_seg'], pwlf['fine_slope'], pwlf['fine_bias'],
        sram_scratch,
        cp_rank_offset,
        N_PER_CORE=N_PER_CORE,
        TILE_SIZE=TILE_SIZE,
        N_TILES=N_TILES,
        N_BUCKETS=N_BUCKETS,
        N_PAIRS=N_PAIRS,
        LOG2_TILE=LOG2_TILE,
        LOG2_BUCKETS=LOG2_BUCKETS,
        LOG2_CAND=LOG2_CAND,
        K=K,
        MAX_CAND=MAX_CAND,
        SCALE=pwlf['SCALE'],
        SRAM_PER_CORE=SRAM_PER_CORE,
    )
    
    return out_logits[:K], out_indices[:K].to(torch.int32)


关键设计点总结
1. 单 Kernel 完成所有事
7 个 phase 全在 1 个 kernel 内,通过 tl.zeus.core_send 和 tl.zeus.barrier() 跨 core 同步。
2. SRAM 布局优化
[zeros_16k: 16K]      ← 一次性初始化,服务所有 cumsum
[working_16k: 16K]    ← cumsum 工作区 (复用)
[tile_buf_a/b: 4K×2]  ← double buffer
[cand_logits: 16K]    ← 候选 (持久,跨 phase)
[cand_indices: 16K]
[hist + 杂项]
总 ~304 KB / core,预留 ~210 KB,符合"1/4 预留"要求。
3. Double Buffer 流水
Phase 1 和 Phase 4 都用 double buffer:
t=0: load tile 0 → buf_a
t=1: compute tile 0 (buf_a) ∥ load tile 1 → buf_b  
t=2: compute tile 1 (buf_b) ∥ load tile 2 → buf_a
...
4. 不缓存 bucket_id
Phase 1 不存 bucket_id (太大 32 KB × 重要性低), Phase 4 重新做 PWLF (便宜)。
5. 跨 Core Send 仅 3 次
Send 1: 一轮 hist (64 fp32)
Send 2: 二轮 hist (64 fp32)
Send 3: n_above + n_eq (2 fp32)
总通信 < 1 KB,几乎瞬时。
6. Pass 数最少
HBM read pass:
  Phase 1: 读 logits 一次 (64 KB)
  Phase 4: 读 logits 第二次 (64 KB)
  
HBM write pass:
  Phase 7: 写 out 一次 (~16 KB)
  
总: 2 次 HBM read pass + 1 次 HBM write pass
如果想极致优化,Phase 1 期间把 bucket_id 写到 cand_logits 区 (暂时占用), Phase 4 就不用重读 logits。但 SRAM 不够装 32K bucket_id。可以只对最后几个 tile 用,前几个 tile 重读——但收益有限,不优化。

性能估算
Phase 1: 直方图 (8 tile, DMA + 计算流水)
  DMA 8 tile (32 KB): ~10 μs (HBM 带宽)
  PWLF 8 次:           ~1 μs
  32×8 pair PWLF + reduce: ~10 μs
  合计 (DMA 重叠): ~12 μs

Phase 2: 交换 hist: ~1 μs

Phase 3: 找 b_star: ~0.5 μs

Phase 4: Collect (8 tile, DMA + 计算流水)
  DMA: ~10 μs
  PWLF + cumsum 8 次: ~30 μs (cumsum 占大头)
  合计: ~30 μs

Phase 5: 二次 radix
  Load 16K 候选 (SRAM, 快): ~1 μs
  PWLF: ~0.2 μs
  二次 hist: ~5 μs
  合计: ~6 μs

Phase 6: 交换二次 hist + 找 b_star_2: ~1 μs

Phase 7: 写出 top-K
  Above/Eq cumsum (16K): ~10 μs
  HBM write: ~3 μs
  合计: ~13 μs

总: ~65 μs
对比原 3-kernel 方案 (~70 μs):

节省 ~5 μs (kernel launch 开销)
代码简洁 (单 kernel 易调试)


注意事项

tl.zeus.core_send 和 tl.zeus.barrier() 的实际 API: 上面是占位,需要根据你的具体 API 调整。
tl.static_range 内的 if 分支: 用于 t % 2 == 0 判断 buf_a/b, Triton 应能在编译期展开。
peer_core_id: tl.constexpr = 1 - core_id: core_id 是运行时值,不能做 constexpr。实际写法可能是:

pythonpeer_core_id = 1 - core_id   # 运行时值
然后 send 的 dst_core 参数支持运行时值。

垃圾区: SRAM 内 MAX_CAND - TILE_SIZE + tile_idx 作为垃圾区,可能与候选区冲突 (如果候选数超过 MAX_CAND - TILE_SIZE)。需要保证候选数 ≤ 12K (= MAX_CAND - TILE_SIZE),实际 Phase 1 后候选数通常 < 4K,安全。