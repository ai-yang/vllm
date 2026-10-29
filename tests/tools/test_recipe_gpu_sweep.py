# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise recipe planning and generated CLI dry runs without model execution."""

import ast
import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from transformers import PreTrainedConfig, Qwen3MoeConfig

from vllm.benchmarks.sweep.param_sweep import ParameterSweep
from vllm.benchmarks.sweep.server import ServerProcess


@pytest.fixture(autouse=True)
def recipe_import_path(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "tools" / "recipes")
    )


@pytest.fixture
def generation():
    return importlib.import_module("sweep.sweep_generation")


@pytest.fixture
def workload(generation):
    return generation.WorkloadHints(input_tokens=128, output_tokens=32, concurrency=8)


@pytest.fixture
def hardware_factory():
    hw = importlib.import_module("hardware_detection")

    def make(count=4, *, device_type="cuda", numa_count=2):
        return hw.HardwareInfo(
            architecture="x86_64",
            socket_count=2,
            numa_nodes=tuple(
                hw.NumaNodeInfo(i, (i,), 1, 512 << 30, 256 << 30)
                for i in range(numa_count)
            ),
            allowed_logical_cpu_count=numa_count,
            physical_core_count=numa_count,
            device_type=device_type,
            gpu_devices=tuple(
                hw.GpuDeviceInfo(
                    visible_index=i,
                    physical_index=count - i - 1,
                    uuid=f"GPU-test-{i}",
                    name="NVIDIA GeForce RTX 3090",
                    compute_capability=(8, 6),
                    total_memory_bytes=24 << 30,
                    free_memory_bytes=23 << 30,
                    pci_bus_id=f"0000:{i:02x}:00.0",
                    numa_node_id=i % numa_count,
                )
                for i in range(count)
            ),
            gpu_links=tuple(
                hw.GpuLinkInfo(i, j, 1, False, 40)
                for i in range(count)
                for j in range(count)
                if i != j
            ),
        )

    return make


def _layouts(candidates):
    return {
        (item["tensor_parallel_size"], item["data_parallel_size"])
        for item in candidates
    }


@pytest.mark.parametrize("count", [1, 3, 4, 6, 8])
def test_gpu_layouts_cover_single_partial_and_full_budget(
    generation, workload, hardware_factory, count
):
    candidates = generation.build_parallel_layout_params(
        {}, workload, hardware_factory(count)
    )
    expected = {(tp, dp) for tp in (1, 2, 4, 8) for dp in range(1, count // tp + 1)}
    assert _layouts(candidates) == expected
    assert (1, 1) in expected
    for item in ParameterSweep.from_records(candidates):
        tp, dp = item["tensor_parallel_size"], item["data_parallel_size"]
        assert item.name == f"tp{tp}_dp{dp}_gpu{tp * dp}of{count}"


def test_gpu_sweep_rejects_no_visible_gpu(generation, workload, hardware_factory):
    with pytest.raises(ValueError, match="visible GPU"):
        generation.build_parallel_layout_params({}, workload, hardware_factory(0))


@pytest.mark.parametrize(
    "count,expected",
    [
        (2, ["tp2_dp1", "tp1_dp2"]),
        (4, ["tp4_dp1", "tp2_dp2", "tp1_dp4"]),
        (6, ["tp4_dp1_numa4of6", "tp2_dp3", "tp1_dp6"]),
        (8, ["tp8_dp1", "tp4_dp2", "tp2_dp4", "tp1_dp8"]),
    ],
)
def test_cpu_candidates_and_names_are_unchanged(
    generation, workload, hardware_factory, count, expected
):
    constraints = importlib.import_module("model_constraints")
    candidates = generation.build_parallel_layout_params(
        {},
        workload,
        hardware_factory(0, device_type="cpu", numa_count=count),
        head_constraints=(constraints.HeadConstraint("text", 1, 1),),
    )
    assert [item["_benchmark_name"] for item in candidates] == expected
    with pytest.raises(ValueError, match="two NUMA"):
        generation.build_parallel_layout_params(
            {}, workload, hardware_factory(0, device_type="cpu", numa_count=1)
        )


def test_gpu_heads_allow_kv_replication_but_reject_bad_partitions(
    generation, workload, hardware_factory
):
    constraints = importlib.import_module("model_constraints")
    replicated = generation.build_parallel_layout_params(
        {},
        workload,
        hardware_factory(),
        head_constraints=(constraints.HeadConstraint("text", 8, 1),),
    )
    assert (4, 1) in _layouts(replicated)
    for heads, kv in ((6, 2), (8, 3)):
        candidates, skipped = generation.build_parallel_layout_plan(
            {},
            workload,
            hardware_factory(),
            head_constraints=(constraints.HeadConstraint("text", heads, kv),),
        )
        assert (4, 1) not in _layouts(candidates)
        assert any(
            item["tensor_parallel_size"] == 4
            and item["policy"] == "attention_head_partition"
            and "text" in item["reason"]
            for item in skipped
        )


def test_gpu_moe_retains_dp_without_cpu_collectives(
    generation, workload, hardware_factory
):
    metadata = generation.ModelParallelMetadata(True, 7, 6)
    gpu, skipped = generation.build_parallel_layout_plan(
        {},
        workload,
        hardware_factory(),
        model_metadata=metadata,
        cpu_moe_dp_supported=False,
    )
    assert _layouts(gpu) == {(1, 1), (1, 2), (1, 3), (2, 1)}
    assert {item["policy"] for item in skipped} == {"moe_effective_parallel_size"}
    ep = generation.build_parallel_layout_params(
        {"enable-expert-parallel": True},
        workload,
        hardware_factory(),
        model_metadata=metadata,
        cpu_moe_dp_supported=False,
    )
    assert {(4, 1), (2, 2), (1, 4)} <= _layouts(ep)
    cpu, skipped = generation.build_parallel_layout_plan(
        {},
        workload,
        hardware_factory(0, device_type="cpu", numa_count=6),
        model_metadata=generation.ModelParallelMetadata(True, 8, 16),
        cpu_moe_dp_supported=False,
    )
    assert _layouts(cpu) == {(4, 1), (2, 1), (1, 1)}
    assert {item["policy"] for item in skipped} == {"cpu_moe_dp_collectives"}


@pytest.mark.parametrize(
    "key,value",
    [
        ("pipeline-parallel-size", 2),
        ("prefill-context-parallel-size", 2),
        ("decode-context-parallel-size", 2),
        ("nnodes", 2),
        ("node-rank", 1),
        ("data-parallel-rank", 0),
        ("data-parallel-start-rank", 0),
        ("data-parallel-size-local", 1),
        ("distributed-executor-backend", "external_launcher"),
        ("distributed-executor-backend", "ray"),
        ("data-parallel-backend", "ray"),
        ("data-parallel-external-lb", True),
        ("data-parallel-hybrid-lb", True),
        ("data-parallel-multi-port-external-lb", True),
        ("headless", True),
    ],
)
def test_gpu_layout_scope_rejects_unsupported_deployment(
    generation, workload, hardware_factory, key, value
):
    config = {key: value}
    with pytest.raises(ValueError, match=key):
        generation.build_parallel_layout_params(config, workload, hardware_factory())
    assert config == {key: value}


@pytest.fixture
def hf_config_service(monkeypatch):
    """Only the external HF lookup is replaced; metadata extraction stays real."""

    def load(model, **kwargs):
        return PreTrainedConfig(
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=1,
            **(kwargs.get("hf_overrides_kw") or {}),
        )

    monkeypatch.setattr("vllm.transformers_utils.config.get_config", load)


def _write_package(
    generation, tmp_path, workload, hardware, config=None, *, full=False
):
    config = {"model": "test/model", **(config or {})}
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config))
    env_path = tmp_path / "env.sh"
    env_path.write_text("")
    directory = tmp_path / "sweep"
    writer = (
        generation.write_full_sweep_files
        if full
        else generation.write_parallel_layout_sweep_files
    )
    writer(
        str(directory),
        config_path=str(config_path),
        env_path=str(env_path),
        config=config,
        workload=workload,
        hardware=hardware,
    )
    return directory


@pytest.mark.parametrize(
    "overrides,expected",
    [
        (
            {"num_attention_heads": 6, "num_key_value_heads": 1},
            {(1, 1), (1, 2), (1, 4), (2, 1), (2, 2)},
        ),
        (
            {"moe_intermediate_size": 6},
            {(1, 1), (1, 2), (1, 3), (2, 1)},
        ),
    ],
)
def test_local_model_overrides_filter_layouts(
    generation, workload, hardware_factory, tmp_path, overrides, expected
):
    model_dir = tmp_path / "model"
    Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=128,
        head_dim=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
    ).save_pretrained(model_dir)
    directory = _write_package(
        generation,
        tmp_path,
        workload,
        hardware_factory(),
        {"model": str(model_dir), "hf-overrides": json.dumps(overrides)},
    )
    params = ParameterSweep.read_json(directory / "parallel_layout_serve_params.json")
    assert _layouts(params) == expected
    assert json.loads((directory / "parallel_layout_precheck.json").read_text()) == {
        "heads": "checked",
        "moe": "checked",
    }
    assert all(
        set(item)
        == {
            "_benchmark_name",
            "tensor_parallel_size",
            "data_parallel_size",
        }
        for item in params
    )


def test_unknown_metadata_is_unprechecked_not_filtered(
    generation, workload, hardware_factory, tmp_path, monkeypatch
):
    def unavailable(*args, **kwargs):
        raise OSError("HF unavailable")

    monkeypatch.setattr("vllm.transformers_utils.config.get_config", unavailable)
    directory = _write_package(generation, tmp_path, workload, hardware_factory())
    precheck = json.loads((directory / "parallel_layout_precheck.json").read_text())
    assert precheck["heads"] == precheck["moe"] == "not_prechecked"
    assert (
        len(ParameterSweep.read_json(directory / "parallel_layout_serve_params.json"))
        == 7
    )


def test_eplb_single_gpu_failure_saves_skip_reason(
    generation, workload, hardware_factory, tmp_path, hf_config_service
):
    with pytest.raises(ValueError, match="No parallel-layout candidates"):
        _write_package(
            generation,
            tmp_path,
            workload,
            hardware_factory(1),
            {"enable-expert-parallel": True, "enable-eplb": True},
        )
    skipped = json.loads(
        (tmp_path / "sweep" / "parallel_layout_skips.json").read_text()
    )
    assert len(skipped) == 1
    assert skipped[0]["policy"] == "eplb_parallel_size"
    assert skipped[0]["tensor_parallel_size"] == skipped[0]["data_parallel_size"] == 1
    assert skipped[0]["reason"]


@pytest.mark.parametrize("profile", ["rtx_3090", "rtx_3090_4x"])
def test_gpu_runtime_policy_preserves_layout_and_uses_least_free_device(
    hardware_factory, profile
):
    runtime = importlib.import_module("sweep.runtime_tuning")
    hardware = hardware_factory()
    devices = list(hardware.gpu_devices)
    devices[-1] = replace(devices[-1], free_memory_bytes=13 << 30)
    hardware = replace(hardware, gpu_devices=tuple(devices))
    for recipe_fraction, expected in ((0.9, 0.52), (0.4, 0.4)):
        config = {
            "tensor-parallel-size": 4,
            "data-parallel-size": 1,
            "gpu-memory-utilization": recipe_fraction,
        }
        result = runtime.finetune_runtime_config(
            config,
            hardware=hardware,
            policies=runtime.get_runtime_tuning_policies(profile),
        )
        effective = config | result.overrides
        assert effective["tensor-parallel-size"] == 4
        assert effective["data-parallel-size"] == 1
        assert effective["gpu-memory-utilization"] == expected
        assert config["gpu-memory-utilization"] == recipe_fraction
        assert expected <= 13 / 24 - 0.02


def test_gpu_runtime_policy_rejects_overbudget_and_non_sm86(hardware_factory):
    runtime = importlib.import_module("sweep.runtime_tuning")
    policies = runtime.get_runtime_tuning_policies("rtx_3090")
    hardware = hardware_factory(1)
    with pytest.raises(ValueError, match="only 1 GPUs"):
        runtime.finetune_runtime_config(
            {"tensor-parallel-size": 2}, hardware=hardware, policies=policies
        )
    hardware = replace(
        hardware,
        gpu_devices=(replace(hardware.gpu_devices[0], compute_capability=(8, 0)),),
    )
    with pytest.raises(ValueError, match="SM86"):
        runtime.finetune_runtime_config({}, hardware=hardware, policies=policies)


@pytest.mark.parametrize(
    "full,runner,stage_config,host,client_host",
    [
        (
            False,
            "run_parallel_layout_sweep.sh",
            "parallel-layout-benchmark-config.yml",
            "0.0.0.0",
            "127.0.0.1",
        ),
        (
            False,
            "run_concurrency_sweep.sh",
            "concurrency-benchmark-config.yml",
            "0.0.0.0",
            "127.0.0.1",
        ),
        (
            True,
            "run_parallel_layout_sweep.sh",
            "parallel-layout-benchmark-config.yml",
            "127.0.0.1",
            "127.0.0.1",
        ),
        (True, "run_concurrency_sweep.sh", "parallel-layout-config.yml", "::", "[::1]"),
        (True, "run_sweep.sh", "parallel-layout-sweep-config.yml", "::1", "[::1]"),
    ],
)
def test_generated_runners_use_actual_stage_endpoint_in_real_cli_dry_run(
    generation,
    workload,
    hardware_factory,
    tmp_path,
    hf_config_service,
    full,
    runner,
    stage_config,
    host,
    client_host,
):
    directory = _write_package(
        generation,
        tmp_path,
        workload,
        hardware_factory(1),
        {"port": 8000},
        full=full,
    )
    if not full and runner == "run_concurrency_sweep.sh":
        generation.write_concurrency_sweep_files(
            str(directory),
            config_path=str(tmp_path / "config.yml"),
            env_path=str(tmp_path / "env.sh"),
            config={"model": "test/model", "port": 8000},
            workload=workload,
        )
    # A selected layout can differ from the original recipe: resolve at run time.
    (directory / stage_config).write_text(
        yaml.safe_dump(
            {
                "model": "test/model",
                "host": host,
                "port": 8100,
            }
        )
    )
    if runner == "run_sweep.sh":
        (directory / "serve_params.json").write_text('[{"max_num_seqs": 8}]')
    environment = dict(os.environ)
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    )
    environment["VLLM_RECIPE_SWEEP_RETRIES"] = "0"
    result = subprocess.run(
        ["bash", str(directory / runner), "--dry-run", "--num-runs", "1"],
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    servers = []
    benchmarks = []
    for line in result.stdout.splitlines():
        if line.startswith("Server command: "):
            servers.append(ast.literal_eval(line.removeprefix("Server command: ")))
        elif line.startswith("Benchmark command: "):
            benchmarks.append(
                ast.literal_eval(line.removeprefix("Benchmark command: "))
            )
    assert servers and benchmarks
    expected_url = f"http://{client_host}:8100"
    for command in servers:
        assert command[command.index("--host") + 1] == host
        assert command[command.index("--port") + 1] == "8100"
        server = ServerProcess(command, [], show_stdout=False)
        assert server._get_vllm_server_address() == expected_url
    for command in benchmarks:
        assert command[command.index("--base-url") + 1] == expected_url


@pytest.mark.parametrize(
    ("before_config", "after_config", "expected"),
    [
        ([], [], "http://127.0.0.1:18086"),
        (["--host", "::", "-p", "18087"], [], "http://[::1]:18087"),
        (
            [],
            ["--host=127.0.0.2", "--port=18088"],
            "http://127.0.0.2:18088",
        ),
        ([], ["--port", "18087", "-p18089"], "http://127.0.0.1:18089"),
    ],
)
def test_native_sweep_address_uses_config_and_cli_precedence(
    tmp_path, before_config, after_config, expected
):
    config = tmp_path / "serve.yaml"
    config.write_text(
        yaml.safe_dump({"model": "test/model", "host": "0.0.0.0", "port": 18086})
    )
    server = ServerProcess(
        [
            "vllm",
            "serve",
            *before_config,
            "--config",
            str(config),
            *after_config,
        ],
        [],
        show_stdout=False,
    )
    assert server._get_vllm_server_address() == expected


def test_native_sweep_does_not_load_other_servers_config(tmp_path):
    config = tmp_path / "infinity.json"
    config.write_text('{"port": 18086}')
    server = ServerProcess(
        ["infinity_emb", "--config", str(config), "--port", "18087"],
        [],
        show_stdout=False,
    )
    assert server._get_vllm_server_address() == "http://localhost:18087"
