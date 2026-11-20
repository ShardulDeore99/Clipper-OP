"""Command line entry point: python -m clip_engine <authorized-local-media> [--ffmpeg PATH]."""
import argparse
import json
import logging

from .pipeline.highlight_pipeline import analyze_media


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze audio activity in an authorized local recording.")
    parser.add_argument("media", help="Local video or 16-bit PCM WAV file")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable (video input only)")
    parser.add_argument("--compact", action="store_true", help="Omit per-window feature rows")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = analyze_media(args.media, args.ffmpeg)
    if args.compact:
        analysis = result["audio_analysis"]
        result["audio_analysis"] = {k: v for k, v in analysis.items() if k != "features"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
