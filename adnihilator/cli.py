"""CLI for AdNihilator."""

import os

# Workaround for OpenMP duplicate library issue on macOS
# This must be set before importing any libraries that use OpenMP
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from . import __version__
from .ad_keywords import find_ad_candidates
from .ad_llm import create_llm_client
from .audio import AudioError, get_duration, validate_audio_file
from .config import Config, load_config
from .models import DetectionResult
from .splice import SpliceError, splice_audio
from .transcript import generate_marked_transcript, generate_summary
from .transcribe import TranscriptionError, download_model as dl_model, transcribe_audio
from .two_pass import two_pass_detect

app = typer.Typer(
    name="adnihilator",
    help="Detect advertisements in podcast audio files.",
    add_completion=False,
)


def format_duration(seconds: float) -> str:
    """Format duration in HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@app.command()
def detect(
    input_mp3: Annotated[Path, typer.Argument(help="Path to the input audio file")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output JSON file path")],
    transcript_out: Annotated[
        Optional[Path],
        typer.Option("--transcript", "-t", help="Output marked transcript file path"),
    ] = None,
    whisper_model: Annotated[
        str, typer.Option("--whisper-model", "-m", help="Whisper model to use")
    ] = "small",
    device: Annotated[
        str, typer.Option("--device", "-d", help="Device to use (cpu or cuda)")
    ] = "cpu",
    llm_provider: Annotated[
        str,
        typer.Option("--llm-provider", "-l", help="LLM provider (none or openai)"),
    ] = "none",
    two_pass: Annotated[
        bool,
        typer.Option("--two-pass", help="Use two-pass mode: fast segment transcription + targeted word timestamps"),
    ] = False,
    config_path: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """Detect advertisements in a podcast audio file."""
    # Validate input file
    try:
        validate_audio_file(str(input_mp3))
    except AudioError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Load configuration
    try:
        config = load_config(str(config_path) if config_path else None)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Override LLM provider if specified on command line
    if llm_provider != "none":
        config.llm.provider = llm_provider

    # Get audio duration
    typer.echo(f"Processing: {input_mp3}")
    try:
        duration = get_duration(str(input_mp3))
    except AudioError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Duration: {format_duration(duration)}")

    # Two-pass mode requires LLM
    if two_pass and config.llm.provider == "none":
        typer.echo("Warning: Two-pass mode works best with LLM enabled. Consider using --llm-provider openai")

    if two_pass:
        # Use two-pass optimization
        typer.echo("Using two-pass mode (segment-level + targeted word timestamps)...")
        try:
            llm_client = create_llm_client(config)
            segments, ad_spans = two_pass_detect(
                audio_path=str(input_mp3),
                llm_client=llm_client,
                config=config,
                duration=duration,
                model_name=whisper_model,
                device=device,
            )
            # Reconstruct candidates from spans for compatibility
            candidates = find_ad_candidates(segments, duration)
        except (TranscriptionError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
    else:
        # Original single-pass mode
        # Transcribe
        typer.echo(f"Transcribing with Whisper ({whisper_model})...")
        try:
            segments = transcribe_audio(str(input_mp3), whisper_model, device)
        except TranscriptionError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Segments: {len(segments)}")

        # Find ad candidates
        typer.echo("Detecting ad candidates...")
        candidates = find_ad_candidates(segments, duration)
        typer.echo(f"Heuristic ad candidates: {len(candidates)}")

        # Refine with LLM
        typer.echo(f"Refining with LLM ({config.llm.provider})...")
        try:
            llm_client = create_llm_client(config)
            ad_spans = llm_client.refine_candidates(segments, candidates, config)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

    typer.echo(f"Final ad spans: {len(ad_spans)}")

    # Build result
    result = DetectionResult(
        audio_path=str(input_mp3.absolute()),
        duration=duration,
        segments=segments,
        candidates=candidates,
        ad_spans=ad_spans,
        model_info={
            "whisper_model": whisper_model,
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model if config.llm.provider != "none" else None,
            "two_pass": two_pass,
        },
    )

    # Write JSON output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(result.model_dump_json(indent=2))

    # Write transcript output if requested
    if transcript_out:
        transcript_out.parent.mkdir(parents=True, exist_ok=True)
        transcript_content = generate_marked_transcript(result)
        with open(transcript_out, "w") as f:
            f.write(transcript_content)
        typer.echo(f"Transcript written to {transcript_out}")

    # Print summary
    typer.echo()
    typer.echo(generate_summary(result))
    typer.echo()
    typer.echo(f"Output written to {out}")


@app.command("download-model")
def download_model(
    model_name: Annotated[
        str, typer.Argument(help="Whisper model name to download")
    ] = "small",
    device: Annotated[
        str, typer.Option("--device", "-d", help="Device to use (cpu or cuda)")
    ] = "cpu",
) -> None:
    """Download a Whisper model for offline use."""
    typer.echo(f"Downloading Whisper model: {model_name}")
    typer.echo(f"Device: {device}")
    typer.echo()

    try:
        dl_model(model_name, device)
        typer.echo()
        typer.echo(f"Model '{model_name}' is ready to use.")
    except TranscriptionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def transcript(
    json_file: Annotated[Path, typer.Argument(help="Path to detection result JSON file")],
    out: Annotated[
        Optional[Path],
        typer.Option("--out", "-o", help="Output file (default: stdout)"),
    ] = None,
) -> None:
    """Generate a marked transcript from a detection result JSON file."""
    if not json_file.exists():
        typer.echo(f"Error: File not found: {json_file}", err=True)
        raise typer.Exit(1)

    try:
        with open(json_file) as f:
            data = json.load(f)
        result = DetectionResult.model_validate(data)
    except Exception as e:
        typer.echo(f"Error parsing JSON: {e}", err=True)
        raise typer.Exit(1)

    transcript_content = generate_marked_transcript(result)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(transcript_content)
        typer.echo(f"Transcript written to {out}")
    else:
        typer.echo(transcript_content)


@app.command()
def refine(
    json_file: Annotated[Path, typer.Argument(help="Path to existing detection result JSON")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output JSON file path")],
    transcript_out: Annotated[
        Optional[Path],
        typer.Option("--transcript", "-t", help="Output marked transcript file path"),
    ] = None,
    llm_provider: Annotated[
        str,
        typer.Option("--llm-provider", "-l", help="LLM provider (none or openai)"),
    ] = "openai",
    config_path: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to config file")
    ] = None,
) -> None:
    """Re-run ad detection on an existing transcript (skips transcription)."""
    if not json_file.exists():
        typer.echo(f"Error: File not found: {json_file}", err=True)
        raise typer.Exit(1)

    # Load existing result
    try:
        with open(json_file) as f:
            data = json.load(f)
        existing = DetectionResult.model_validate(data)
    except Exception as e:
        typer.echo(f"Error parsing JSON: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded transcript: {len(existing.segments)} segments")
    typer.echo(f"Duration: {format_duration(existing.duration)}")

    # Load configuration
    try:
        config = load_config(str(config_path) if config_path else None)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Override LLM provider
    config.llm.provider = llm_provider

    # Re-run heuristic detection
    typer.echo("Detecting ad candidates...")
    candidates = find_ad_candidates(existing.segments, existing.duration)
    typer.echo(f"Heuristic ad candidates: {len(candidates)}")

    # Refine with LLM
    typer.echo(f"Refining with LLM ({config.llm.provider})...")
    try:
        llm_client = create_llm_client(config)
        ad_spans = llm_client.refine_candidates(existing.segments, candidates, config)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Final ad spans: {len(ad_spans)}")

    # Build result
    result = DetectionResult(
        audio_path=existing.audio_path,
        duration=existing.duration,
        segments=existing.segments,
        candidates=candidates,
        ad_spans=ad_spans,
        model_info={
            "whisper_model": existing.model_info.get("whisper_model", "unknown"),
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model if config.llm.provider != "none" else None,
        },
    )

    # Write JSON output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(result.model_dump_json(indent=2))

    # Write transcript output if requested
    if transcript_out:
        transcript_out.parent.mkdir(parents=True, exist_ok=True)
        transcript_content = generate_marked_transcript(result)
        with open(transcript_out, "w") as f:
            f.write(transcript_content)
        typer.echo(f"Transcript written to {transcript_out}")

    # Print summary
    typer.echo()
    typer.echo(generate_summary(result))
    typer.echo()
    typer.echo(f"Output written to {out}")


@app.command()
def splice(
    json_file: Annotated[Path, typer.Argument(help="Path to detection result JSON file")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output audio file path")],
    min_confidence: Annotated[
        float,
        typer.Option("--min-confidence", "-c", help="Minimum confidence for ad removal"),
    ] = 0.5,
) -> None:
    """Remove detected ads from an audio file.

    Takes a detection result JSON and the original audio file,
    then creates a new audio file with ad segments removed.
    """
    if not json_file.exists():
        typer.echo(f"Error: File not found: {json_file}", err=True)
        raise typer.Exit(1)

    # Load detection result
    try:
        with open(json_file) as f:
            data = json.load(f)
        result = DetectionResult.model_validate(data)
    except Exception as e:
        typer.echo(f"Error parsing JSON: {e}", err=True)
        raise typer.Exit(1)

    # Validate original audio file exists
    audio_path = Path(result.audio_path)
    if not audio_path.exists():
        typer.echo(f"Error: Original audio file not found: {audio_path}", err=True)
        typer.echo("Make sure the audio file hasn't been moved since detection.", err=True)
        raise typer.Exit(1)

    # Filter ads by confidence
    qualifying_ads = [s for s in result.ad_spans if s.confidence >= min_confidence]
    typer.echo(f"Original duration: {format_duration(result.duration)}")
    typer.echo(f"Ad spans to remove: {len(qualifying_ads)} (confidence >= {min_confidence})")

    if not qualifying_ads:
        typer.echo("No ads meet the confidence threshold. Nothing to remove.")
        raise typer.Exit(0)

    # Show what will be removed
    total_ad_time = sum(s.end - s.start for s in qualifying_ads)
    typer.echo(f"Total ad time to remove: {format_duration(total_ad_time)}")
    typer.echo()

    # Perform splice
    typer.echo("Splicing audio...")
    try:
        stats = splice_audio(
            input_path=str(audio_path),
            output_path=str(out),
            ad_spans=result.ad_spans,
            duration=result.duration,
            min_confidence=min_confidence,
        )
    except SpliceError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Print summary
    typer.echo()
    typer.echo("Splice complete!")
    typer.echo(f"  Ads removed: {stats['ads_removed']}")
    typer.echo(f"  Time removed: {format_duration(stats['time_removed'])}")
    typer.echo(f"  Original duration: {format_duration(stats['original_duration'])}")
    typer.echo(f"  New duration: {format_duration(stats['new_duration'])}")
    typer.echo()
    typer.echo(f"Output written to: {out}")


@app.command()
def worker(
    once: Annotated[
        bool,
        typer.Option("--once", help="Process one job and exit"),
    ] = False,
    daemon: Annotated[
        bool,
        typer.Option("--daemon", help="Run as daemon"),
    ] = False,
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Seconds between queue checks"),
    ] = 300,
    artifacts_dir: Annotated[
        Optional[str],
        typer.Option("--artifacts-dir", help="Directory to save detection artifacts"),
    ] = "data/artifacts",
) -> None:
    """Run the local worker to process podcast episodes."""
    from worker.daemon import WorkerDaemon

    # Get configuration from environment
    api_url = os.environ.get("API_URL")
    if not api_url:
        typer.echo("Error: API_URL environment variable not set", err=True)
        raise typer.Exit(1)

    api_key = os.environ.get("WORKER_API_KEY")
    if not api_key:
        typer.echo("Error: WORKER_API_KEY environment variable not set", err=True)
        raise typer.Exit(1)

    r2_access_key = os.environ.get("R2_ACCESS_KEY")
    r2_secret_key = os.environ.get("R2_SECRET_KEY")
    r2_bucket = os.environ.get("R2_BUCKET")
    r2_endpoint = os.environ.get("R2_ENDPOINT")

    if not all([r2_access_key, r2_secret_key, r2_bucket, r2_endpoint]):
        typer.echo("Error: R2 environment variables not set", err=True)
        typer.echo("Required: R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET, R2_ENDPOINT", err=True)
        raise typer.Exit(1)

    worker_daemon = WorkerDaemon(
        api_url=api_url,
        api_key=api_key,
        r2_access_key=r2_access_key,
        r2_secret_key=r2_secret_key,
        r2_bucket=r2_bucket,
        r2_endpoint=r2_endpoint,
        artifacts_dir=artifacts_dir,
    )

    if once:
        processed = worker_daemon.run_once()
        if processed:
            typer.echo("Processed one job")
        else:
            typer.echo("No jobs in queue")
    elif daemon:
        worker_daemon.run_daemon(interval=interval)
    else:
        typer.echo("Specify --once or --daemon mode")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show the version of AdNihilator."""
    typer.echo(f"AdNihilator v{__version__}")


if __name__ == "__main__":
    app()
