import asyncio
import sys
import torch
from pathlib import Path
import typer
from loguru import logger
import uvicorn
from typing import Optional, List, Dict, Any
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
import time
from concurrent.futures import ThreadPoolExecutor
import psutil
import shutil

from config.settings import ASRSettings
from core.pipeline import ASRPipeline
from api.endpoints import app
from utils.error_handler import handle_error, ASRException
from utils.validators import validate_audio_file

# Initialize console and CLI app
console = Console()
app_cli = typer.Typer(help="Advanced Multilingual ASR System")

class ProcessingManager:
    def __init__(self, settings: ASRSettings):
        """Initialize processing manager."""
        self.settings = settings
        self.device = torch.device(settings.DEVICE)
        self.pipeline: Optional[ASRPipeline] = None
        self._setup_logging()
        self.thread_pool = ThreadPoolExecutor(max_workers=settings.NUM_THREADS)

    def _setup_logging(self):
        """Configure logging system."""
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.remove()  # Remove default handler
        logger.add(
            self.settings.LOG_DIR / "asr.log",
            rotation="100 MB",
            retention="30 days",
            format=log_format,
            level="INFO",
            backtrace=True,
            diagnose=True
        )
        logger.add(sys.stderr, format=log_format, level="INFO")

    async def initialize_pipeline(self) -> None:
        """Initialize ASR pipeline with resource management."""
        try:
            if self.pipeline is None:
                # Check system resources
                available_memory = psutil.virtual_memory().available
                required_memory = self.settings.MIN_REQUIRED_MEMORY

                if available_memory < required_memory:
                    raise ASRException(
                        f"Insufficient memory. Required: {required_memory/1e9:.1f}GB, "
                        f"Available: {available_memory/1e9:.1f}GB"
                    )

                if self.device.type == 'cuda':
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    if gpu_memory < self.settings.MIN_REQUIRED_GPU_MEMORY:
                        raise ASRException(
                            f"Insufficient GPU memory. Required: "
                            f"{self.settings.MIN_REQUIRED_GPU_MEMORY/1e9:.1f}GB, "
                            f"Available: {gpu_memory/1e9:.1f}GB"
                        )

                self.pipeline = ASRPipeline(
                    device=self.device,
                    config=self.settings.to_dict()
                )
                logger.info(f"ASR Pipeline initialized on {self.device}")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            raise ASRException(f"Pipeline initialization failed: {str(e)}")

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.pipeline:
                await self.pipeline.cleanup()
            self.thread_pool.shutdown(wait=True)
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise

@app_cli.command()
def process_file(
    input_file: Path = typer.Argument(
        ...,
        help="Input audio file path",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        help="Output JSON file path"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        help="Configuration file path",
        exists=True
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output"
    )
):
    """Process single audio file with detailed progress tracking."""
    try:
        # Load settings
        settings = ASRSettings()
        if config_file:
            settings = ASRSettings.load_config(config_file)

        # Initialize processing manager
        manager = ProcessingManager(settings)

        async def run_processing():
            await manager.initialize_pipeline()

            try:
                # Validate audio file
                await validate_audio_file(str(input_file))

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    *Progress.get_default_columns(),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task(
                        "Processing audio...",
                        total=100
                    )

                    async def progress_callback(
                        progress_val: float,
                        status: str
                    ):
                        progress.update(
                            task,
                            completed=progress_val * 100,
                            description=status
                        )

                    start_time = time.time()
                    result = await manager.pipeline.process_file(
                        str(input_file),
                        progress_callback=progress_callback
                    )
                    processing_time = time.time() - start_time

                    # Add processing metadata
                    result['metadata'].update({
                        'processing_time': processing_time,
                        'input_file': str(input_file),
                        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'device': str(manager.device),
                        'settings_version': settings.version
                    })

                    # Save results
                    output_path = output_file or input_file.with_suffix('.json')
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    console.print(f"\n[green]Processing completed in {processing_time:.2f} seconds![/green]")
                    if verbose:
                        console.print(result)

            finally:
                await manager.cleanup()

        asyncio.run(run_processing())

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Processing failed: {str(e)}")
        raise typer.Exit(code=1)

@app_cli.command()
def batch_process(
    input_dir: Path = typer.Argument(
        ...,
        help="Input directory with audio files",
        exists=True,
        file_okay=False,
        dir_okay=True
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Output directory for results"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        help="Configuration file path",
        exists=True
    ),
    num_workers: int = typer.Option(
        4,
        help="Number of parallel workers",
        min=1,
        max=16
    ),
    file_pattern: str = typer.Option(
        "*",
        help="File pattern to match (e.g., *.wav)"
    )
):
    """Process multiple audio files with parallel processing."""
    try:
        settings = ASRSettings()
        if config_file:
            settings = ASRSettings.load_config(config_file)

        manager = ProcessingManager(settings)

        async def process_single_file(
            file_path: Path,
            progress: Progress,
            task_id: int
        ) -> Dict[str, Any]:
            try:
                await validate_audio_file(str(file_path))
                result = await manager.pipeline.process_file(str(file_path))
                
                output_path = output_dir / f"{file_path.stem}_result.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                progress.update(task_id, advance=1)
                return {'status': 'success', 'file': str(file_path)}
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                return {'status': 'failed', 'file': str(file_path), 'error': str(e)}

        async def batch_processor():
            await manager.initialize_pipeline()

            try:
                # Setup output directory
                output_dir.mkdir(parents=True, exist_ok=True)

                # Find audio files
                audio_files = [
                    f for f in input_dir.glob(f"**/{file_pattern}")
                    if f.suffix.lower() in {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
                ]

                if not audio_files:
                    raise ASRException(f"No audio files found in {input_dir}")

                results = []
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    *Progress.get_default_columns(),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    total_task = progress.add_task(
                        "Processing files...",
                        total=len(audio_files)
                    )

                    # Process in batches
                    batch_size = num_workers
                    for i in range(0, len(audio_files), batch_size):
                        batch = audio_files[i:i + batch_size]
                        tasks = [
                            process_single_file(
                                file_path,
                                progress,
                                total_task
                            )
                            for file_path in batch
                        ]
                        batch_results = await asyncio.gather(*tasks)
                        results.extend(batch_results)

                        # Clear GPU memory after each batch
                        if manager.device.type == 'cuda':
                            torch.cuda.empty_cache()

                # Generate summary
                successful = sum(1 for r in results if r['status'] == 'success')
                failed = sum(1 for r in results if r['status'] == 'failed')

                # Save batch processing report
                report = {
                    'summary': {
                        'total_files': len(audio_files),
                        'successful': successful,
                        'failed': failed,
                        'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'results': results
                }

                report_path = output_dir / 'batch_processing_report.json'
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)

                console.print(f"\n[green]Batch processing completed![/green]")
                console.print(f"Successful: {successful}")
                console.print(f"Failed: {failed}")
                console.print(f"Report saved to: {report_path}")

            finally:
                await manager.cleanup()

        asyncio.run(batch_processor())

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Batch processing failed: {str(e)}")
        raise typer.Exit(code=1)

@app_cli.command()
def start_server(
    host: str = typer.Option(
        "0.0.0.0",
        help="Server host"
    ),
    port: int = typer.Option(
        8000,
        help="Server port",
        min=1,
        max=65535
    ),
    workers: int = typer.Option(
        4,
        help="Number of worker processes",
        min=1,
        max=16
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        help="Configuration file path",
        exists=True
    ),
    reload: bool = typer.Option(
        False,
        help="Enable auto-reload for development"
    )
):
    """Start the ASR web server with monitoring."""
    try:
        settings = ASRSettings()
        if config_file:
            settings = ASRSettings.load_config(config_file)

        # Setup logging
        log_config = uvicorn.config.LOGGING_CONFIG
        log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"

        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            workers=workers,
            log_config=log_config,
            timeout_keep_alive=120,
            loop="auto",
            reload=reload,
            limit_concurrency=settings.MAX_CONCURRENT_REQUESTS,
            limit_max_requests=settings.MAX_REQUESTS_PER_WORKER
        )

        server = uvicorn.Server(config)
        server.run()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Server startup failed: {str(e)}")
        raise typer.Exit(code=1)

def main():
    """Entry point with Windows compatibility."""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        app_cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        sys.exit(0)

if __name__ == "__main__":
    main()