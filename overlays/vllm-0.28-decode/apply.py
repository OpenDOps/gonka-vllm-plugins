#!/usr/bin/env python3
"""Apply the decode-PoC seams onto an installed vLLM 0.28 tree.

The nonce math lives in gonka_poc. What 0.28 lacks is the engine plumbing
that release/v0.30-decode-int added in Python: PoCParams on the request,
scheduler step sizes, and the runner bridge hooks. This script inserts those
seams into the stock 0.28 sources. It does not copy 0.30 files over 0.28.

Chat preemption while a PoC row waits is omitted. That path calls
Scheduler._request_blocks_can_be_freed, which exists only on 0.30. A decode
container does not mix chat into the same process.

Nothing here is CUDA. Do not nvcc/cucc this overlay. MetaX torch keeps the
torch.cuda name; the MACA kernels are already in the S0 image.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

OVERLAY = Path(__file__).resolve().parent
FILES = OVERLAY / "files"


def _once(text: str, old: str, new: str, label: str) -> str:
    found = text.count(old)
    if found != 1:
        raise SystemExit(f"{label}: expected 1 occurrence, found {found}")
    return text.replace(old, new, 1)


def _write(path: Path, text: str) -> None:
    path.write_text(text)


def apply(root: Path) -> None:
    root = root.resolve()
    engine = root / "v1" / "engine" / "__init__.py"
    if not engine.is_file():
        raise SystemExit(f"not a vllm package: {root}")

    dest_poc = root / "poc"
    if dest_poc.exists():
        shutil.rmtree(dest_poc)
    shutil.copytree(FILES / "vllm" / "poc", dest_poc)

    # --- forward context: second cudagraph key bit ---
    p = root / "forward_context.py"
    text = p.read_text()
    text = _once(
        text,
        '''    to be properly captured.
    """


def _compute_sp_num_tokens(''',
        '''    to be properly captured.
    """
    poc: bool = False
    """
    Whether this batch carries gonka decode-PoC rows. The PoC transforms are
    captured into their own cudagraphs; other batches run the bare model.
    """


def _compute_sp_num_tokens(''',
        "forward_context.BatchDescriptor.poc",
    )
    _write(p, text)

    # --- dispatcher ---
    p = root / "v1" / "cudagraph_dispatcher.py"
    text = p.read_text()
    text = _once(
        text,
        "def __init__(self, vllm_config: VllmConfig):",
        "def __init__(self, vllm_config: VllmConfig, poc_graphs: bool = False):",
        "dispatcher ctor",
    )
    text = _once(
        text,
        """        # Default cudagraph_mode to NONE until initialize_cudagraph_keys is called
        self.cudagraph_mode = CUDAGraphMode.NONE
""",
        """        # Default cudagraph_mode to NONE until initialize_cudagraph_keys is called
        self.cudagraph_mode = CUDAGraphMode.NONE
        # gonka PoC: a second key set for batches with decode-PoC rows.
        # On MACA this stays unused when the server is started --enforce-eager.
        self.poc_graphs = poc_graphs
""",
        "dispatcher poc_graphs",
    )
    text = _once(
        text,
        """                )

        self.keys_initialized = True

    def dispatch(
""",
        """                )

        if self.poc_graphs:
            for keys in self.cudagraph_keys.values():
                keys |= {replace(k, poc=True) for k in keys}

        self.keys_initialized = True

    def dispatch(
""",
        "dispatcher key copy",
    )
    text = _once(
        text,
        """        num_active_loras: int = 0,
        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
""",
        """        num_active_loras: int = 0,
        poc: bool = False,
        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
""",
        "dispatcher dispatch sig",
    )
    text = _once(
        text,
        """        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, effective_num_active_loras
        )

        if CUDAGraphMode.FULL in allowed_modes:
""",
        """        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, effective_num_active_loras
        )
        if poc and self.poc_graphs:
            batch_desc = replace(batch_desc, poc=True)

        if CUDAGraphMode.FULL in allowed_modes:
""",
        "dispatcher dispatch poc",
    )
    _write(p, text)

    # --- engine structs. New fields go last so array_like stays aligned. ---
    p = root / "v1" / "engine" / "__init__.py"
    text = p.read_text()
    text = _once(
        text,
        "from vllm.pooling_params import PoolingParams\n",
        "from vllm.poc.poc_params import PoCParams\n"
        "from vllm.pooling_params import PoolingParams\n",
        "engine import PoCParams",
    )
    text = _once(
        text,
        """    session_id: str | None = None

    @property
    def params(self) -> SamplingParams | PoolingParams:
""",
        """    session_id: str | None = None

    # PoC: consensus parameters; None for normal requests.
    poc_params: PoCParams | None = None

    @property
    def params(self) -> SamplingParams | PoolingParams:
""",
        "EngineCoreRequest.poc_params",
    )
    text = _once(
        text,
        """    new_sampling_mask: SamplingMaskLists | None = None

    @property
    def finished(self) -> bool:
""",
        """    new_sampling_mask: SamplingMaskLists | None = None

    # Decode-PoC artifact. None on chat and prefill rows.
    poc_output: dict[str, Any] | None = None

    @property
    def finished(self) -> bool:
""",
        "EngineCoreOutput.poc_output",
    )
    _write(p, text)

    # --- Request ---
    p = root / "v1" / "request.py"
    text = p.read_text()
    text = _once(
        text,
        """if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash
""",
        """if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.poc.poc_params import PoCParams
    from vllm.v1.core.kv_cache_utils import BlockHash
""",
        "request TYPE_CHECKING",
    )
    text = _once(
        text,
        """        abort_immediately: bool = False,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
""",
        """        abort_immediately: bool = False,
        poc_params: "PoCParams | None" = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.poc_params = poc_params
""",
        "Request ctor",
    )
    text = _once(
        text,
        """                self.kv_cache_report_mode = "incremental"
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")
""",
        """                self.kv_cache_report_mode = "incremental"
        elif poc_params is not None:
            assert sampling_params is None and pooling_params is None
            # Finish is the artifact, not sampled tokens.
            self.max_tokens = 1
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")
""",
        "Request max_tokens",
    )
    text = _once(
        text,
        """            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
""",
        """            pooling_params=request.pooling_params,
            poc_params=getattr(request, "poc_params", None),
            arrival_time=request.arrival_time,
""",
        "Request.from_engine_core_request",
    )
    text = _once(
        text,
        """    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
""",
        """    def get_skip_reading_prefix_cache(self) -> bool:
        if self.poc_params is not None:
            # A prefix hit would skip the atomic prefill and shift the chain.
            return True
        if (
            self.sampling_params is not None
""",
        "Request prefix cache",
    )
    _write(p, text)

    # --- NewRequestData ---
    p = root / "v1" / "core" / "sched" / "output.py"
    text = p.read_text()
    text = _once(
        text,
        """    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
""",
        """    from vllm.poc.poc_params import PoCParams
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
""",
        "NewRequestData import",
    )
    text = _once(
        text,
        """    prompt_embeds: "torch.Tensor | None" = None
    prompt_is_token_ids: list[bool] | None = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None
""",
        """    prompt_embeds: "torch.Tensor | None" = None
    prompt_is_token_ids: list[bool] | None = None
    poc_params: "PoCParams | None" = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None
""",
        "NewRequestData field",
    )
    text = _once(
        text,
        """            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
        )
""",
        """            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            poc_params=request.poc_params,
            prefill_token_ids=prefill_token_ids,
        )
""",
        "NewRequestData.from_request",
    )
    _write(p, text)

    # --- scheduler ---
    p = root / "v1" / "core" / "sched" / "scheduler.py"
    text = p.read_text()
    text = _once(
        text,
        "from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs\n",
        "from gonka_poc.mixed.admission import poc_step_tokens\n"
        "from vllm.v1.engine import (\n"
        "    EngineCoreEventType,\n"
        "    EngineCoreOutput,\n"
        "    EngineCoreOutputs,\n"
        "    FinishReason,\n"
        ")\n",
        "scheduler imports",
    )
    text = _once(
        text,
        """        ) and any(not r.is_prefill_chunk for r in self.running)

        # First, schedule the RUNNING requests.
""",
        """        ) and any(not r.is_prefill_chunk for r in self.running)

        # While PoC rows are live, chat gets no tokens. Chat keeps its KV.
        # Preempting chat to free a slot needs Scheduler._request_blocks_can_be_freed,
        # which 0.28 does not have; a decode container does not mix chat.
        poc_gate = self._poc_rows_waiting() or any(
            r.poc_params is not None for r in self.running
        )

        # First, schedule the RUNNING requests.
""",
        "scheduler poc_gate",
    )
    text = _once(
        text,
        """            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
""",
        """            if (
                request.poc_params is None
                and request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
""",
        "scheduler skip poc on async stop",
    )
    text = _once(
        text,
        """            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
""",
        """            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            if poc_gate and request.poc_params is None:
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
""",
        "scheduler running chat skip",
    )
    text = _once(
        text,
        """            num_new_tokens = self._reserve_prefill_lookahead(
                request, request.num_computed_tokens, num_new_tokens
            )

            if num_new_tokens == 0:
""",
        """            num_new_tokens = self._reserve_prefill_lookahead(
                request, request.num_computed_tokens, num_new_tokens
            )

            # PoC row: atomic prefill, then one token per decode step.
            num_new_tokens = poc_step_tokens(request, num_new_tokens, token_budget)
            if num_new_tokens == 0 and request.poc_params is not None:
                req_index += 1
                continue

            if num_new_tokens == 0:
""",
        "scheduler running poc_step_tokens",
    )
    text = _once(
        text,
        """                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
""",
        """                request = request_queue.peek_request()
                request_id = request.request_id

                if poc_gate and request.poc_params is None:
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # try to promote blocked statuses while traversing skipped queue.
""",
        "scheduler waiting chat skip",
    )
    text = _once(
        text,
        """                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break

                # During async KV load, no forward pass is run yet.
""",
        """                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break

                num_new_tokens = poc_step_tokens(request, num_new_tokens, token_budget)
                if num_new_tokens == 0 and request.poc_params is not None:
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # During async KV load, no forward pass is run yet.
""",
        "scheduler waiting poc_step_tokens",
    )
    text = _once(
        text,
        """            if output_is_stale and request.drop_stale_output:
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
""",
        """            if output_is_stale and request.drop_stale_output:
                continue

            if request.poc_params is not None:
                # PoC finish = artifact presence (emit-once).
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= 1
                poc_outputs = getattr(model_runner_output, "poc_outputs", None)
                poc_obj = poc_outputs.get(req_id) if poc_outputs else None
                if poc_obj is None:
                    continue
                poc_payload = {
                    "nonce": poc_obj.nonce,
                    "vector_b64": poc_obj.vector_b64,
                    "k_points_steps": getattr(poc_obj, "k_points_steps", []),
                    "n_sphere_mismatches": getattr(poc_obj, "n_sphere_mismatches", -1),
                    "sph_indices_steps": getattr(poc_obj, "sph_indices_steps", []),
                    "sph_values_steps": getattr(poc_obj, "sph_values_steps", []),
                    "n_nan_steps": getattr(poc_obj, "n_nan_steps", 0),
                    "mismatch_margin_max": getattr(poc_obj, "mismatch_margin_max", 0.0),
                }
                status_before_stop = request.status
                request.status = RequestStatus.FINISHED_STOPPED
                self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=[],
                        finish_reason=FinishReason.STOP,
                        poc_output=poc_payload,
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
""",
        "scheduler poc finish",
    )
    text = _once(
        text,
        """        return new_block_ids_to_zero or None

    def _preempt_request(
""",
        """        return new_block_ids_to_zero or None

    def _poc_rows_waiting(self) -> bool:
        return any(r.poc_params is not None for r in self.waiting) or any(
            r.poc_params is not None for r in self.skipped_waiting
        )

    def _preempt_request(
""",
        "scheduler _poc_rows_waiting",
    )
    _write(p, text)

    # --- async llm ---
    p = root / "v1" / "engine" / "async_llm.py"
    text = p.read_text()
    text = _once(
        text,
        """        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> RequestOutputCollector:
        \"\"\"Add new request to the AsyncLLM.\"\"\"

        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)
""",
        """        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        poc_params=None,
    ) -> RequestOutputCollector:
        \"\"\"Add new request to the AsyncLLM.\"\"\"

        if self.errored:
            raise EngineDeadError()

        if poc_params is not None:
            # Dummy tokens: seeded embeds are injected in-model.
            queue = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)
            request = EngineCoreRequest(
                request_id=request_id,
                prompt_token_ids=[0] * poc_params.seq_len,
                mm_features=None,
                sampling_params=None,
                pooling_params=None,
                arrival_time=arrival_time if arrival_time is not None else time.time(),
                lora_request=lora_request,
                cache_salt=request_id,
                data_parallel_rank=data_parallel_rank,
                priority=priority,
                poc_params=poc_params,
            )
            self.input_processor.assign_request_id(request)
            self._run_output_handler()
            await self._add_request(request, None, None, 0, queue)
            return queue

        is_pooling = isinstance(params, PoolingParams)
""",
        "async_llm add_request",
    )
    text = _once(
        text,
        """        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
""",
        """        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        poc_params=None,
    ) -> AsyncGenerator[RequestOutput, None]:
""",
        "async_llm generate sig",
    )
    text = _once(
        text,
        """        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                session_id=session_id,
                prompt_text=prompt_text,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=reasoning_parser_kwargs,
            )
""",
        """        q: RequestOutputCollector | None = None
        if poc_params is not None and prompt is not None:
            raise ValueError("poc_params cannot be used with a prompt")
        try:
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                session_id=session_id,
                poc_params=poc_params,
                prompt_text=prompt_text,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=reasoning_parser_kwargs,
            )
""",
        "async_llm generate call",
    )
    _write(p, text)

    # --- output processor ---
    p = root / "v1" / "engine" / "output_processor.py"
    text = p.read_text()
    text = _once(
        text,
        """            temperature = sampling_params.temperature
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind
""",
        """            temperature = sampling_params.temperature
        elif getattr(request, "poc_params", None) is not None:
            from vllm.sampling_params import RequestOutputKind

            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            output_kind = RequestOutputKind.FINAL_ONLY
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind
""",
        "output_processor from_new_request",
    )
    text = _once(
        text,
        """            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            finish_reason = engine_core_output.finish_reason
""",
        """            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            poc_output = getattr(engine_core_output, "poc_output", None)
            if poc_output is not None:
                poc_request_output = RequestOutput(
                    request_id=req_id,
                    prompt=None,
                    prompt_token_ids=[],
                    prompt_logprobs=None,
                    outputs=[],
                    finished=True,
                    poc_output=poc_output,
                )
                if req_state.queue is not None:
                    req_state.queue.put(poc_request_output)
                else:
                    request_outputs.append(poc_request_output)
                self._finish_request(req_state)
                continue
            finish_reason = engine_core_output.finish_reason
""",
        "output_processor poc_output",
    )
    _write(p, text)

    # --- public RequestOutput ---
    p = root / "outputs.py"
    text = p.read_text()
    text = _once(
        text,
        """        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        # Forward compatibility, code that uses args added in new release can
""",
        """        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        poc_output: dict[str, Any] | None = None,
        # Forward compatibility, code that uses args added in new release can
""",
        "RequestOutput sig",
    )
    text = _once(
        text,
        """        self.kv_transfer_params = kv_transfer_params
        self.ec_transfer_params = ec_transfer_params
""",
        """        self.kv_transfer_params = kv_transfer_params
        self.ec_transfer_params = ec_transfer_params
        self.poc_output = poc_output
""",
        "RequestOutput assign",
    )
    _write(p, text)

    # --- ModelRunnerOutput + PoCOutput ---
    p = root / "v1" / "outputs.py"
    text = p.read_text()
    text = _once(
        text,
        """# ModelRunnerOutput is serialized and sent to the scheduler process.
""",
        '''@dataclass
class PoCOutput:
    nonce: int
    vector_b64: str
    k_points_steps: list[int] = field(default_factory=list)
    n_sphere_mismatches: int = -1
    n_nan_steps: int = 0
    mismatch_margin_max: float = 0.0
    sph_indices_steps: list[list[int]] = field(default_factory=list)
    sph_values_steps: list[str] = field(default_factory=list)


# ModelRunnerOutput is serialized and sent to the scheduler process.
''',
        "PoCOutput",
    )
    text = _once(
        text,
        """    # ``None`` when ``return_sampling_mask`` is off.
    sampling_masks: SamplingMaskLists | None = None
""",
        """    # ``None`` when ``return_sampling_mask`` is off.
    sampling_masks: SamplingMaskLists | None = None

    # PoC artifacts per request id (None when PoC is idle).
    poc_outputs: dict[str, PoCOutput] | None = None
""",
        "ModelRunnerOutput.poc_outputs",
    )
    _write(p, text)

    # --- CachedRequestState + input batch ---
    p = root / "v1" / "worker" / "gpu_input_batch.py"
    text = p.read_text()
    text = _once(
        text,
        """    prompt_embeds: torch.Tensor | None = None
    # To accumulate prompt logprobs tensor chunks across prefill steps.
""",
        """    prompt_embeds: torch.Tensor | None = None
    poc_params: object | None = None
    # To accumulate prompt logprobs tensor chunks across prefill steps.
""",
        "CachedRequestState.poc_params",
    )
    text = _once(
        text,
        """            self.logits_processing_needs_token_ids[req_index] = (
                pooling_params.requires_token_ids
            )
        else:
            raise NotImplementedError("Unrecognized request type")
""",
        """            self.logits_processing_needs_token_ids[req_index] = (
                pooling_params.requires_token_ids
            )
        elif request.poc_params is not None:
            pass
        else:
            raise NotImplementedError("Unrecognized request type")
""",
        "input batch poc row",
    )
    _write(p, text)

    # --- gpu model runner ---
    p = root / "v1" / "worker" / "gpu_model_runner.py"
    text = p.read_text()
    text = _once(
        text,
        "        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)\n",
        "        self.cudagraph_dispatcher = CudagraphDispatcher(\n"
        "            self.vllm_config,\n"
        "            poc_graphs=self.compilation_config.mode == CompilationMode.NONE,\n"
        "        )\n",
        "runner dispatcher",
    )
    text = _once(
        text,
        """                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
""",
        """                output_token_ids=[],
                lora_request=new_req_data.lora_request,
                poc_params=new_req_data.poc_params,
            )
""",
        "CachedRequestState ctor",
    )
    text = _once(
        text,
        """        num_encoder_reqs: int = 0,
    ) -> tuple[
""",
        """        num_encoder_reqs: int = 0,
        poc: bool = False,
    ) -> tuple[
""",
        "determine_batch sig",
    )
    text = _once(
        text,
        """            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
""",
        """            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                poc=poc,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
""",
        "determine_batch dispatch",
    )
    text = _once(
        text,
        """        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
""",
        """        if spec_decode_metadata is None:
            bridge = getattr(self, "_poc_bridge", None)
            if bridge is not None and bridge.mixed_active():
                return bridge.sample_chat_rows(logits, sampling_metadata)
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
""",
        "runner _sample",
    )
    text = _once(
        text,
        """        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
""",
        """        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if getattr(self, "_poc_bridge", None) is None:
            from gonka_poc.mixed.bridge import PoCRunnerBridge

            self._poc_bridge = PoCRunnerBridge(self)

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
""",
        "runner bridge create",
    )
    text = _once(
        text,
        """            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                allow_microbatching=self._allow_microbatching(
                    num_reqs, num_scheduled_tokens_np
                ),
            )
""",
        """            poc_step = self._poc_bridge.pre_step(scheduler_output)
            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                allow_microbatching=self._allow_microbatching(
                    num_reqs, num_scheduled_tokens_np
                ),
                poc=poc_step,
            )
""",
        "runner pre_step",
    )
    text = _once(
        text,
        """            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        # Encoder-decoder models can only compile the pure decode steps where no
""",
        """            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        if getattr(self, "_poc_bridge", None) is not None:
            self._poc_bridge.pre_forward(
                scheduler_output,
                positions,
                scheduler_output.total_num_scheduled_tokens,
            )

        # Encoder-decoder models can only compile the pure decode steps where no
""",
        "runner pre_forward",
    )
    text = _once(
        text,
        """                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
""",
        """                sample_hidden_states = hidden_states[logits_indices]
                logits = (
                    self._poc_bridge.compute_logits(sample_hidden_states)
                    if getattr(self, "_poc_bridge", None) is not None
                    and self._poc_bridge.mixed_active()
                    else self.model.compute_logits(sample_hidden_states)
                )
            else:
""",
        "runner compute_logits common",
    )
    text = _once(
        text,
        """                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
""",
        """                    logits = (
                        self._poc_bridge.compute_logits(sample_hidden_states)
                        if getattr(self, "_poc_bridge", None) is not None
                        and self._poc_bridge.mixed_active()
                        else self.model.compute_logits(sample_hidden_states)
                    )

                model_output_broadcast_data: dict[str, Any] = {}
""",
        "runner compute_logits broadcast",
    )
    text = _once(
        text,
        """                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                routed_experts=None,
            )
""",
        """                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                routed_experts=None,
                poc_outputs=self._poc_bridge.extract(hidden_states)
                if getattr(self, "_poc_bridge", None) is not None
                else None,
            )
""",
        "runner poc_outputs",
    )
    text = _once(
        text,
        """                if self.lora_config:
                    self.model = self.load_lora_model(
                        self.model, self.vllm_config, self.device
                    )
                if hasattr(self, "drafter"):
""",
        """                if self.lora_config:
                    self.model = self.load_lora_model(
                        self.model, self.vllm_config, self.device
                    )
                # Attach PoC transforms before compilation or capture.
                if getattr(self, "_poc_bridge", None) is None:
                    from gonka_poc.mixed.bridge import PoCRunnerBridge

                    self._poc_bridge = PoCRunnerBridge(self)
                self._poc_bridge.load(self.model)
                if hasattr(self, "drafter"):
""",
        "runner load_model",
    )
    text = _once(
        text,
        """        profile_seq_lens: int | None = None,
        randomize_inputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
""",
        """        profile_seq_lens: int | None = None,
        randomize_inputs: bool = False,
        poc: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
""",
        "dummy_run sig",
    )
    text = _once(
        text,
        """                force_num_active_loras=num_active_loras,
            )
        )

        if cudagraph_runtime_mode is None:
""",
        """                force_num_active_loras=num_active_loras,
                poc=poc,
            )
        )

        if cudagraph_runtime_mode is None:
""",
        "dummy_run poc",
    )
    text = _once(
        text,
        """        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
        for _ in range(num_warmups):
""",
        """        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
        if getattr(self, "_poc_bridge", None) is not None:
            self._poc_bridge.set_active(desc.poc)
        for _ in range(num_warmups):
""",
        "warmup set_active",
    )
    text = _once(
        text,
        """                is_graph_capturing=True,
                profile_seq_lens=profile_seq_lens,
            )
""",
        """                is_graph_capturing=True,
                profile_seq_lens=profile_seq_lens,
                poc=desc.poc,
            )
""",
        "capture dummy poc",
    )
    _write(p, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-root",
        type=Path,
        help="Path to the installed vllm package (the directory that contains __init__.py).",
    )
    args = parser.parse_args()
    if args.vllm_root is None:
        import vllm

        root = Path(vllm.__file__).resolve().parent
    else:
        root = args.vllm_root
    apply(root)
    print(f"decode PoC seams applied to {root}")


if __name__ == "__main__":
    main()
