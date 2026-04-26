from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    for lib in ("httpx", "httpcore", "websockets", "uvicorn.access"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def _load_settings(config: str | None) -> "Settings":
    from ez_comfy.config.schema import load_settings
    return load_settings(config)


# ---------------------------------------------------------------------------
# check sub-command
# ---------------------------------------------------------------------------

async def _cmd_check(args: argparse.Namespace) -> int:
    from ez_comfy.comfyui.autodetect import detect_comfyui
    from ez_comfy.comfyui.client import ComfyUIClient
    from ez_comfy.hardware.comfyui_inventory import scan_inventory
    from ez_comfy.hardware.probe import probe_hardware

    settings = _load_settings(args.config)

    if settings.comfyui.auto_detect:
        detected = detect_comfyui(
            explicit_url=settings.comfyui.base_url if settings.comfyui.base_url != "http://127.0.0.1:8188" else None,
            explicit_model_path=settings.comfyui.model_base_path or None,
        )
        if detected.base_url:
            settings.comfyui.base_url = detected.base_url
        if detected.model_base_path and not settings.comfyui.model_base_path:
            settings.comfyui.model_base_path = detected.model_base_path
        print(f"Detected ComfyUI at {settings.comfyui.base_url} (source: {detected.source})")
    else:
        print(f"Checking ComfyUI at {settings.comfyui.base_url} …")

    client = ComfyUIClient(base_url=settings.comfyui.base_url)
    ok = await client.health_check()
    if not ok:
        print("ERROR: ComfyUI is not reachable. Is it running?")
        await client.close()
        return 1

    hw = probe_hardware()
    print(f"GPU     : {hw.gpu_name}  ({hw.gpu_vram_gb:.1f} GB VRAM)")
    print(f"RAM     : {hw.system_ram_gb:.1f} GB")
    print(f"Platform: {hw.platform}")

    print("\nScanning inventory …")
    inv = await scan_inventory(client)
    print(f"Checkpoints : {len(inv.checkpoints)}")
    for c in inv.checkpoints[:10]:
        print(f"  • {c.filename}  [{c.family}]  {c.size_bytes / 1e9:.1f}GB")
    if len(inv.checkpoints) > 10:
        print(f"  … and {len(inv.checkpoints) - 10} more")
    print(f"LoRAs       : {len(inv.loras)}")
    print(f"VAEs        : {len(inv.vaes)}")
    print(f"Upscalers   : {len(inv.upscale_models)}")
    caps = sorted(inv.discovered_class_types)
    print(f"Capabilities: {caps[:10]}")
    await client.close()
    return 0


# ---------------------------------------------------------------------------
# plan sub-command
# ---------------------------------------------------------------------------

async def _cmd_plan(args: argparse.Namespace) -> int:
    from ez_comfy.comfyui.client import ComfyUIClient
    from ez_comfy.hardware.comfyui_inventory import scan_inventory
    from ez_comfy.hardware.probe import probe_hardware
    from ez_comfy.planner.planner import GenerationRequest, plan_generation

    settings = _load_settings(args.config)
    client = ComfyUIClient(base_url=settings.comfyui.base_url)
    hw = probe_hardware()
    inv = await scan_inventory(client)
    await client.close()

    request = GenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative or "",
        intent_override=args.intent,
        checkpoint_override=args.checkpoint,
        recipe_override=args.recipe,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
    )
    plan = plan_generation(
        request=request,
        hardware=hw,
        inventory=inv,
        prefer_speed=settings.preferences.prefer_speed,
        auto_negative=settings.preferences.auto_negative_prompt,
    )
    summary = plan.summary()
    print(json.dumps(summary, indent=2))
    if plan.warnings:
        print("\nWarnings:")
        for w in plan.warnings:
            print(f"  [WARN] {w}")
    return 0


# ---------------------------------------------------------------------------
# recommend sub-command
# ---------------------------------------------------------------------------

async def _cmd_recommend(args: argparse.Namespace) -> int:
    from ez_comfy.comfyui.client import ComfyUIClient
    from ez_comfy.hardware.comfyui_inventory import scan_inventory
    from ez_comfy.hardware.probe import probe_hardware
    from ez_comfy.models.catalog import recommend_models
    from ez_comfy.planner.intent import detect_intent

    settings = _load_settings(args.config)
    client = ComfyUIClient(base_url=settings.comfyui.base_url)
    hw = probe_hardware()
    inv = await scan_inventory(client)
    await client.close()

    intent = args.intent or detect_intent(args.prompt, False, False).value
    recs = recommend_models(
        prompt=args.prompt,
        intent=intent,
        hardware=hw,
        inventory=inv,
        prefer_speed=settings.preferences.prefer_speed,
    )
    print(f"Top recommendations for intent={intent!r}:\n")
    for i, r in enumerate(recs[:8], 1):
        installed_tag = "installed" if r.installed else "not installed"
        print(f"  {i}. [{installed_tag}] {r.entry.name}  (score={r.score})")
        print(f"     {'; '.join(r.match_reasons)}")
        print(f"     VRAM >= {r.entry.vram_min_gb}GB  |  {r.entry.filename}")
        if not r.installed and r.entry.download_command:
            print(f"     Get: {r.entry.download_command}")
        print()
    return 0


# ---------------------------------------------------------------------------
# generate sub-command
# ---------------------------------------------------------------------------

async def _cmd_generate(args: argparse.Namespace) -> int:
    from ez_comfy.comfyui.client import ComfyUIClient, ProgressEvent
    from ez_comfy.engine import GenerationEngine
    from ez_comfy.hardware.comfyui_inventory import scan_inventory
    from ez_comfy.hardware.probe import probe_hardware
    from ez_comfy.planner.planner import GenerationRequest

    settings = _load_settings(args.config)
    client = ComfyUIClient(base_url=settings.comfyui.base_url)
    hw = probe_hardware()
    inv = await scan_inventory(client)

    ref_bytes: bytes | None = None
    if args.reference:
        with open(args.reference, "rb") as f:
            ref_bytes = f.read()

    request = GenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative or "",
        reference_image=ref_bytes,
        intent_override=args.intent,
        checkpoint_override=args.checkpoint,
        recipe_override=args.recipe,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        denoise_strength=args.denoise,
    )

    engine = GenerationEngine(
        client=client,
        settings=settings,
        hardware=hw,
        inventory=inv,
    )

    def on_progress(event: ProgressEvent) -> None:
        if event.event_type == "progress" and event.step is not None:
            pct = int(event.step / (event.total_steps or 1) * 100)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)
        elif event.event_type == "executing" and event.node_id:
            print(f"\n  Executing node {event.node_id} …", end="", flush=True)

    print(f"Generating: {args.prompt[:60]}")
    try:
        result = await engine.generate(request, on_progress=on_progress, timeout=args.timeout)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        await client.close()
        return 1

    print(f"\nDone in {result.duration_seconds:.1f}s  |  recipe={result.plan.recipe.id}  |  checkpoint={result.plan.checkpoint}")
    if result.plan.warnings:
        for w in result.plan.warnings:
            print(f"  [WARN] {w}")
    print(f"\nOutputs ({len(result.outputs)}):")
    base = settings.comfyui.base_url.rstrip("/")
    for out in result.outputs:
        fn = out.get("filename", "")
        sf = out.get("subfolder", "")
        tp = out.get("type", "output")
        print(f"  {fn}  ->  {base}/view?filename={fn}&subfolder={sf}&type={tp}")

    await client.close()
    return 0


# ---------------------------------------------------------------------------
# serve sub-command
# ---------------------------------------------------------------------------

def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from ez_comfy.api.server import create_app

    settings = _load_settings(args.config)
    host = args.host or "127.0.0.1"
    port = args.port or 7860

    app = create_app(settings)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info" if not args.verbose else "debug",
    )
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ez-comfy",
        description="EZ Comfy — hardware-aware ComfyUI orchestrator",
    )
    parser.add_argument("--config", "-c", help="Path to settings YAML")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # check
    sub.add_parser("check", help="Check ComfyUI connection and show inventory")

    # plan
    p_plan = sub.add_parser("plan", help="Plan a generation (no GPU)")
    p_plan.add_argument("prompt")
    p_plan.add_argument("--negative", "-n")
    p_plan.add_argument("--intent")
    p_plan.add_argument("--checkpoint")
    p_plan.add_argument("--recipe")
    p_plan.add_argument("--width", type=int)
    p_plan.add_argument("--height", type=int)
    p_plan.add_argument("--steps", type=int)
    p_plan.add_argument("--seed", type=int, default=-1)

    # recommend
    p_rec = sub.add_parser("recommend", help="Get model recommendations")
    p_rec.add_argument("prompt")
    p_rec.add_argument("--intent")

    # generate
    p_gen = sub.add_parser("generate", help="Generate an image/audio/video")
    p_gen.add_argument("prompt")
    p_gen.add_argument("--negative", "-n")
    p_gen.add_argument("--intent")
    p_gen.add_argument("--checkpoint")
    p_gen.add_argument("--recipe")
    p_gen.add_argument("--reference", help="Reference image path")
    p_gen.add_argument("--width", type=int)
    p_gen.add_argument("--height", type=int)
    p_gen.add_argument("--steps", type=int)
    p_gen.add_argument("--seed", type=int, default=-1)
    p_gen.add_argument("--denoise", type=float, default=0.7)
    p_gen.add_argument("--timeout", type=float, default=300.0)

    # serve
    p_srv = sub.add_parser("serve", help="Start the web server")
    p_srv.add_argument("--host", default=None)
    p_srv.add_argument("--port", type=int, default=None)

    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "serve":
        sys.exit(_cmd_serve(args))
    elif args.command == "check":
        sys.exit(asyncio.run(_cmd_check(args)))
    elif args.command == "plan":
        sys.exit(asyncio.run(_cmd_plan(args)))
    elif args.command == "recommend":
        sys.exit(asyncio.run(_cmd_recommend(args)))
    elif args.command == "generate":
        sys.exit(asyncio.run(_cmd_generate(args)))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
