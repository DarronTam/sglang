import logging
from typing import TYPE_CHECKING

from sglang.srt.utils.common import is_zeus

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)
_is_zeus = is_zeus()


def set_default_server_args(args: "ServerArgs") -> None:
    """Apply Zeus-specific defaults and validation to ServerArgs.

    Mirrors ``hardware_backend/npu/utils.set_default_server_args``. Caller
    (``server_args._handle_zeus_backends``) must gate on ``is_zeus()``.
    """

    from sglang.srt.configs.model_config import is_deepseek_nsa
    from sglang.srt.connector import ConnectorType
    from sglang.srt.utils.common import parse_connector_type

    # NSA models (DeepSeek 3.2 / GLM-Next / GlmMoeDsa) use MLA + sparse
    # attention; route them through the dedicated zeus_mla backend which
    # understands latent KV + topk_indices from the indexer. Best-effort:
    # if hf_config cannot be loaded here (e.g. INSTANCE connector path,
    # where _handle_model_specific_adjustments short-circuits later), fall
    # back to the standard ``zeus`` backend.
    is_nsa = False
    try:
        if parse_connector_type(args.model_path) != ConnectorType.INSTANCE:
            is_nsa = is_deepseek_nsa(args.get_model_config().hf_config)
    except Exception:
        is_nsa = False

    if args.attention_backend is None:
        args.attention_backend = "zeus_mla" if is_nsa else "zeus"
    # Zeus graph capture/replay is supported — don't force-disable
    if args.cuda_graph_max_bs is None:
        args.cuda_graph_max_bs = 32
    # Zeus requires page-aligned KV cache. The NSA indexer expects
    # page_size == 64 (see nsa_indexer.forward_indexer); standard zeus path
    # expects a multiple of 128.
    if args.page_size is None:
        args.page_size = 64 if is_nsa else 128
    if args.linear_attn_backend == "triton":
        args.linear_attn_backend = "zeus"

    if is_nsa:
        assert args.page_size == 64, (
            f"Zeus NSA models require page_size == 64, got {args.page_size}"
        )
    else:
        assert args.page_size % 128 == 0, (
            f"Zeus requires page_size to be a multiple of 128, got {args.page_size}"
        )
