import asyncio
import sys
import torch
import gc
from pathlib import Path
import typer
from loguru import logger
import uvicorn
from typing import Optional, List, Dict, Any, Callable
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn
import time
from concurrent.futures import ThreadPoolExecutor
import psutil
import shutil
import os

from config.settings import ASRSettings
from config.validation import load_config, create_default_config, save_config
from core.pipeline import ASRPipeline
from api.endpoints import app
from utils.error_handler import handle_error, ASRException
from utils.validators import validate_audio_file
from utils.cache_manager import CacheManager

# Initialize console and CLI app
console = Console()
app_cli = typer.Typer(help="Advanced Multilingual ASR System with French Optimization")

class ProcessingManager:
    def __init__(self, settings: ASRSettings):
        """
        Initialize processing manager with enhanced resource management.
        
        Args:
            settings: ASR system settings
        """
        self.settings = settings
        self.device = torch.device(settings.DEVICE)
        self.pipeline: Optional[ASRPipeline] = None
        self._setup_logging()
        self.thread_pool = ThreadPoolExecutor(max_workers=settings.NUM_THREADS)
        self.cache_manager = self._initialize_cache_manager()
        
        # Track initialization status
        self.initialized = False
        
        # Memory monitoring settings
        self.memory_warning_threshold = settings.MEMORY_WARNING_THRESHOLD
        self.memory_critical_threshold = settings.MEMORY_CRITICAL_THRESHOLD

    def _setup_logging(self):
        """Configure advanced logging system with rotation and formatting."""
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        # Create log directory if it doesn't exist
        self.settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

        logger.remove()  # Remove default handler
        logger.add(
            self.settings.LOG_DIR / "asr.log",
            rotation="100 MB",
            retention="30 days",
            format=log_format,
            level=self.settings.LOG_LEVEL,
            backtrace=True,
            diagnose=True
        )
        logger.add(sys.stderr, format=log_format, level=self.settings.LOG_LEVEL)
        
        # Add error log for critical issues
        logger.add(
            self.settings.LOG_DIR / "error.log",
            rotation="50 MB",
            retention="60 days",
            format=log_format,
            level="ERROR",
            backtrace=True,
            diagnose=True,
            filter=lambda record: record["level"].name == "ERROR" or record["level"].name == "CRITICAL"
        )

    def _initialize_cache_manager(self) -> CacheManager:
        """Initialize cache manager with settings."""
        cache_dir = self.settings.CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        return CacheManager(
            cache_dir=str(cache_dir),
            max_size_gb=self.settings.MAX_CACHE_SIZE / (1024 * 1024 * 1024),  # Convert to GB
            ttl_hours=self.settings.CACHE_TTL / 3600,  # Convert to hours
            cleanup_interval=self.settings.CLEAN_CACHE_INTERVAL / 3600,  # Convert to hours
            memory_threshold=self.memory_warning_threshold
        )

    async def initialize_pipeline(self) -> None:
        """Initialize ASR pipeline with comprehensive resource management."""
        try:
            if self.pipeline is None:
                # Check system resources
                await self._check_system_resources()

                # Initialize pipeline with cache manager
                self.pipeline = ASRPipeline(
                    device=self.device,
                    cache_manager=self.cache_manager,
                    config=self.settings.to_dict()
                )
                
                logger.info(f"ASR Pipeline initialized on {self.device}")
                self.initialized = True
            else:
                logger.info("Pipeline already initialized")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            raise ASRException(f"Pipeline initialization failed: {str(e)}")

    async def _check_system_resources(self) -> None:
        """
        Check system resources with detailed reporting.
        
        Raises:
            ASRException: If resources are insufficient
        """
        # Check system memory
        vm = psutil.virtual_memory()
        available_memory = vm.available
        required_memory = self.settings.MIN_REQUIRED_MEMORY
        
        memory_percent = vm.percent
        if memory_percent > 90:
            raise ASRException(
                f"System memory usage is too high ({memory_percent}%). "
                f"Please close other applications and try again."
            )
        
        if available_memory < required_memory:
            raise ASRException(
                f"Insufficient memory. Required: {required_memory/1e9:.1f}GB, "
                f"Available: {available_memory/1e9:.1f}GB"
            )
        
        # Check GPU if using CUDA
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
                self.settings.DEVICE = "cpu"
                return
                
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_available = gpu_memory - gpu_memory_allocated
            
            if gpu_memory_available < self.settings.MIN_REQUIRED_GPU_MEMORY:
                raise ASRException(
                    f"Insufficient GPU memory. Required: "
                    f"{self.settings.MIN_REQUIRED_GPU_MEMORY/1e9:.1f}GB, "
                    f"Available: {gpu_memory_available/1e9:.1f}GB"
                )
        
        # Check disk space for cache and output
        cache_path = self.settings.CACHE_DIR
        if cache_path.exists():
            total, used, free = shutil.disk_usage(cache_path)
            if free < 5 * 1024 * 1024 * 1024:  # 5 GB minimum
                logger.warning(f"Low disk space on cache directory: {free/1e9:.1f}GB free")
                
        # Check if required directories are writable
        required_dirs = [
            self.settings.CACHE_DIR,
            self.settings.LOG_DIR,
            self.settings.UPLOAD_DIR,
            self.settings.RESULTS_DIR,
            self.settings.MODEL_DIR
        ]
        
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            if not os.access(directory, os.W_OK):
                raise ASRException(f"Directory not writable: {directory}")

    async def process_file(
        self, 
        file_path: str,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Process audio file with enhanced error handling and resource management.
        
        Args:
            file_path: Path to audio file
            config: Optional processing configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results
        """
        if not self.initialized:
            await self.initialize_pipeline()
        
        try:
            # Validate file
            validation_result = await validate_audio_file(file_path, strict=True)
            
            # Apply any warnings to config
            effective_config = config or {}
            if validation_result.get('warnings'):
                logger.warning(f"File validation warnings: {validation_result['warnings']}")
                # Add warnings to metadata
                if 'metadata' not in effective_config:
                    effective_config['metadata'] = {}
                effective_config['metadata']['validation_warnings'] = validation_result['warnings']
            
            # Add file metadata
            if 'metadata' not in effective_config:
                effective_config['metadata'] = {}
            effective_config['metadata']['file_format'] = validation_result.get('format')
            effective_config['metadata']['sample_rate'] = validation_result.get('sample_rate')
            effective_config['metadata']['duration'] = validation_result.get('duration')
            
            # Process file
            result = await self.pipeline.process_file(
                file_path,
                config=effective_config,
                progress_callback=progress_callback
            )
            
            # Monitor memory after processing
            await self._monitor_memory()
            
            return result
            
        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            error_result = handle_error(e, {"file_path": file_path})
            await self._emergency_cleanup()
            return error_result

    async def _monitor_memory(self) -> None:
        """Monitor memory usage and perform cleanup if needed."""
        try:
            # Check system memory
            memory_percent = psutil.virtual_memory().percent
            
            # Check GPU memory if applicable
            gpu_memory_percent = 0
            if self.device.type == 'cuda' and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_percent = (allocated / total) * 100
            
            # Log memory usage
            logger.debug(f"Memory usage - System: {memory_percent}%, GPU: {gpu_memory_percent}%")
            
            # Handle high memory usage
            if memory_percent > self.memory_critical_threshold or gpu_memory_percent > self.memory_critical_threshold:
                logger.warning("Critical memory usage detected, performing emergency cleanup")
                await self._emergency_cleanup()
            elif memory_percent > self.memory_warning_threshold or gpu_memory_percent > self.memory_warning_threshold:
                logger.info("High memory usage detected, performing routine cleanup")
                await self._routine_cleanup()
                
        except Exception as e:
            logger.warning(f"Memory monitoring failed: {str(e)}")

    async def _routine_cleanup(self) -> None:
        """Perform routine memory cleanup."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    async def _emergency_cleanup(self) -> None:
        """
        Perform emergency cleanup in case of memory issues.
        This is more aggressive than routine cleanup.
        """
        try:
            # Clear pipeline caches if initialized
            if self.pipeline:
                await self.pipeline.cleanup()
            
            # Clear CUDA cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            gc.collect()  # Double collection sometimes helps
            
            # Clear cache manager
            await self.cache_manager.cleanup()
            
            logger.info("Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {str(e)}")

    async def cleanup(self):
        """Cleanup all resources."""
        try:
            if self.pipeline:
                await self.pipeline.cleanup()
                
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Clear GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Clear cache
            await self.cache_manager.cleanup()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Cleanup completed successfully")
            self.initialized = False
            
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
    ),
    optimize_for_french: bool = typer.Option(
        True,
        "--french/--no-french",
        help="Enable/disable French optimization"
    )
):
    """Process single audio file with detailed progress tracking and French optimization."""
    try:
        # Load settings
        settings = ASRSettings()
        if config_file:
            settings = ASRSettings.load_config(config_file)

        # Apply command line overrides
        settings.FRENCH_OPTIMIZATION = optimize_for_french

        # Initialize processing manager
        manager = ProcessingManager(settings)

        async def run_processing():
            await manager.initialize_pipeline()

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task(
                        "Initializing...",
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

                    # Prepare processing configuration
                    process_config = {
                        'optimize_for_french': optimize_for_french,
                        'verbose_output': verbose
                    }

                    start_time = time.time()
                    result = await manager.process_file(
                        str(input_file),
                        config=process_config,
                        progress_callback=progress_callback
                    )
                    processing_time = time.time() - start_time

                    # Add additional metadata
                    if 'metadata' not in result:
                        result['metadata'] = {}
                        
                    result['metadata'].update({
                        'processing_time': processing_time,
                        'input_file': str(input_file),
                        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'device': str(manager.device),
                        'settings_version': settings.VERSION,
                        'french_optimization': optimize_for_french
                    })

                    # Save results
                    output_path = output_file or input_file.with_suffix('.json')
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    console.print(f"\n[green]Processing completed in {processing_time:.2f} seconds![/green]")
                    
                    # Show summary
                    if 'language_distribution' in result:
                        console.print("\n[bold]Language Distribution:[/bold]")
                        for lang in result['language_distribution']:
                            console.print(f"- {lang['language_name']}: {lang['percentage']}%")
                    
                    if 'speakers' in result:
                        console.print(f"\n[bold]Detected Speakers: {len(result['speakers'])}[/bold]")
                        for speaker_id, info in result['speakers'].items():
                            console.print(f"- {speaker_id}: {info['total_duration']:.2f}s ({info['avg_confidence']:.2f} confidence)")
                    
                    console.print(f"\nResults saved to: {output_path}")
                    
                    if verbose:
                        console.print("\n[bold]Full Result:[/bold]")
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
    ),
    optimize_for_french: bool = typer.Option(
        True,
        "--french/--no-french",
        help="Enable/disable French optimization"
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help="Continue processing if individual files fail"
    )
):
    """Process multiple audio files with parallel processing and French optimization."""
    try:
        # Resolve output directory
        if output_dir is None:
            output_dir = input_dir / "results"

        # Load settings
        settings = ASRSettings()
        if config_file:
            settings = ASRSettings.load_config(config_file)
            
        # Apply command line overrides
        settings.FRENCH_OPTIMIZATION = optimize_for_french

        # Adjust workers based on system resources
        cpu_count = os.cpu_count() or 4
        if num_workers > cpu_count:
            logger.warning(f"Requested {num_workers} workers, but only {cpu_count} CPUs available. Limiting to {cpu_count}.")
            num_workers = cpu_count

        manager = ProcessingManager(settings)

        async def process_single_file(
            file_path: Path,
            progress: Progress,
            task_id: int
        ) -> Dict[str, Any]:
            try:
                # Update progress description
                progress.update(task_id, description=f"Processing {file_path.name}")
                
                # Prepare file-specific config
                file_config = {
                    'optimize_for_french': optimize_for_french,
                    'metadata': {
                        'batch_processing': True,
                        'original_filename': file_path.name
                    }
                }
                
                # Define progress callback for this file
                async def file_progress_callback(prog: float, status: str):
                    # Update the main progress bar
                    pass  # We'll update the overall progress instead
                
                # Process the file
                result = await manager.process_file(
                    str(file_path),
                    config=file_config,
                    progress_callback=file_progress_callback
                )
                
                # Save the result
                output_path = output_dir / f"{file_path.stem}_result.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # Update progress
                progress.update(task_id, advance=1, description=f"Processed {file_path.name}")
                
                return {
                    'status': 'success',
                    'file': str(file_path),
                    'output': str(output_path),
                    'duration': result.get('metadata', {}).get('audio_duration', 0),
                    'language': result.get('language_distribution', [{}])[0].get('language', 'unknown') if result.get('language_distribution') else 'unknown',
                    'speakers': len(result.get('speakers', {}))
                }
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to process {file_path}: {error_msg}")
                progress.update(task_id, description=f"Failed: {file_path.name}")
                
                if not continue_on_error:
                    raise
                    
                return {
                    'status': 'failed',
                    'file': str(file_path),
                    'error': error_msg
                }

        async def batch_processor():
            await manager.initialize_pipeline()

            try:
                # Create output directory
                output_dir.mkdir(parents=True, exist_ok=True)

                # Find audio files
                file_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
                audio_files = [
                    f for f in input_dir.glob(f"**/{file_pattern}")
                    if f.suffix.lower() in file_extensions
                ]

                if not audio_files:
                    raise ASRException(f"No audio files found in {input_dir} with pattern {file_pattern}")

                results = []
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    total_task = progress.add_task(
                        f"Processing {len(audio_files)} files...",
                        total=len(audio_files)
                    )

                    # Process in batches for better memory management
                    batch_size = min(num_workers, 3)  # Limit to 3 files at once to manage memory
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

                        # Clean up memory after each batch
                        if manager.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        gc.collect()

                # Generate summary
                successful = sum(1 for r in results if r['status'] == 'success')
                failed = sum(1 for r in results if r['status'] == 'failed')

                # Calculate statistics for successful files
                languages = {}
                total_duration = 0
                total_speakers = 0
                
                for r in results:
                    if r['status'] == 'success':
                        lang = r.get('language', 'unknown')
                        languages[lang] = languages.get(lang, 0) + 1
                        total_duration += r.get('duration', 0)
                        total_speakers += r.get('speakers', 0)
                
                avg_speakers = total_speakers / successful if successful else 0

                # Save batch processing report
                report = {
                    'summary': {
                        'total_files': len(audio_files),
                        'successful': successful,
                        'failed': failed,
                        'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'total_duration': total_duration,
                        'languages': languages,
                        'average_speakers': avg_speakers
                    },
                    'results': results
                }

                report_path = output_dir / 'batch_processing_report.json'
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)

                # Display summary
                console.print(f"\n[green]Batch processing completed![/green]")
                console.print(f"[bold]Successful:[/bold] {successful} files")
                console.print(f"[bold]Failed:[/bold] {failed} files")
                
                if languages:
                    console.print("\n[bold]Languages detected:[/bold]")
                    for lang, count in languages.items():
                        console.print(f"- {lang}: {count} files")
                
                console.print(f"\n[bold]Total audio duration:[/bold] {total_duration:.2f} seconds")
                console.print(f"[bold]Average speakers per file:[/bold] {avg_speakers:.2f}")
                console.print(f"\nReport saved to: {report_path}")

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
    ),
    optimize_for_french: bool = typer.Option(
        True,
        "--french/--no-french",
        help="Enable/disable French optimization"
    )
):
    """Start the ASR web server with monitoring and French optimization."""
    try:
        # Load settings
        settings = ASRSettings()
        if config_file:
            settings = ASRSettings.load_config(config_file)
            
        # Apply command line overrides
        settings.FRENCH_OPTIMIZATION = optimize_for_french

        # Check if port is already in use
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
        except socket.error:
            console.print(f"[yellow]Warning: Port {port} is already in use.[/yellow]")
        finally:
            sock.close()

        # Setup logging
        log_config = uvicorn.config.LOGGING_CONFIG
        log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"

        # Prepare server config
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

        # Pre-server initialization
        console.print(f"[green]Starting ASR server on {host}:{port} with {workers} workers[/green]")
        console.print(f"[blue]French optimization: {'Enabled' if optimize_for_french else 'Disabled'}[/blue]")
        
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            console.print(f"[blue]Using GPU: {gpu_name}[/blue]")
        else:
            console.print(f"[yellow]Using CPU. GPU acceleration is not available.[/yellow]")

        # Start server
        server = uvicorn.Server(config)
        server.run()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Server startup failed: {str(e)}")
        raise typer.Exit(code=1)

@app_cli.command()
def create_config(
    output_file: Path = typer.Argument(
        Path("config/settings.json"),
        help="Output configuration file path"
    ),
    optimize_for_french: bool = typer.Option(
        True,
        "--french/--no-french",
        help="Enable/disable French optimization"
    ),
    device: str = typer.Option(
        "auto",
        help="Processing device (auto, cuda, cpu, cuda:0, etc.)"
    ),
    whisper_model: str = typer.Option(
        "large-v3",
        help="Whisper model size (tiny, base, small, medium, large-v3)"
    ),
    max_speakers: int = typer.Option(
        10,
        help="Maximum number of speakers",
        min=1,
        max=20
    )
):
    """Create a configuration file with customized settings."""
    try:
        # Create default config
        config = create_default_config()
        
        # Apply command line options
        config["french_optimization"] = optimize_for_french
        
        # Set device
        if device == "auto":
            config["hardware"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            config["hardware"]["device"] = device
            
        # Set model size
        config["model"]["whisper_model"] = whisper_model
        
        # Set speaker limit
        config["processing"]["max_speakers"] = max_speakers
        
        # Add system info
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "platform": sys.platform
        }
        
        config["system_info"] = system_info
        
        # Save configuration
        save_config(config, output_file)
        
        console.print(f"[green]Configuration created at {output_file}[/green]")
        console.print(f"French optimization: {'Enabled' if optimize_for_french else 'Disabled'}")
        console.print(f"Device: {config['hardware']['device']}")
        console.print(f"Whisper model: {config['model']['whisper_model']}")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Configuration creation failed: {str(e)}")
        raise typer.Exit(code=1)

@app_cli.command()
def system_check():
    """Check system compatibility and resource availability."""
    console.print("[bold blue]Multilingual ASR System Check[/bold blue]")
    console.print("Checking system requirements and resource availability...\n")
    
    # Check Python version
    python_version = sys.version_info
    console.print(f"[bold]Python Version:[/bold] {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        console.print("[red]  ✗ Python 3.8+ required[/red]")
    else:
        console.print("[green]  ✓ Python version compatible[/green]")
    
    # Check CPU
    cpu_count = psutil.cpu_count(logical=False)
    logical_cpus = psutil.cpu_count(logical=True)
    console.print(f"[bold]CPU:[/bold] {cpu_count} physical cores, {logical_cpus} logical cores")
    if cpu_count < 4:
        console.print("[yellow]  ⚠ At least 4 physical cores recommended for optimal performance[/yellow]")
    else:
        console.print("[green]  ✓ CPU meets requirements[/green]")
    
    # Check RAM
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    console.print(f"[bold]Memory:[/bold] {total_gb:.1f}GB total, {available_gb:.1f}GB available")
    if total_gb < 8:
        console.print("[red]  ✗ At least 8GB RAM required[/red]")
    elif total_gb < 16:
        console.print("[yellow]  ⚠ 16GB+ RAM recommended for optimal performance[/yellow]")
    else:
        console.print("[green]  ✓ Memory meets requirements[/green]")
    
    # Check GPU
    console.print("[bold]GPU Support:[/bold]")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        console.print(f"  Found {gpu_count} CUDA-compatible GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            console.print(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            if gpu_memory < 4:
                console.print("[yellow]    ⚠ At least 4GB VRAM recommended[/yellow]")
            elif gpu_memory < 8 and "large" in "large-v3":
                console.print("[yellow]    ⚠ At least 8GB VRAM recommended for large models[/yellow]")
            else:
                console.print("[green]    ✓ GPU memory sufficient[/green]")
    else:
        console.print("[yellow]  ✗ No CUDA-compatible GPU detected, using CPU only (slower processing)[/yellow]")
    
    # Check disk space
    console.print("[bold]Disk Space:[/bold]")
    app_dir = Path.cwd()
    total, used, free = shutil.disk_usage(app_dir)
    total_gb = total / (1024**3)
    free_gb = free / (1024**3)
    console.print(f"  {free_gb:.1f}GB free of {total_gb:.1f}GB total")
    if free_gb < 10:
        console.print("[yellow]  ⚠ At least 10GB free space recommended[/yellow]")
    else:
        console.print("[green]  ✓ Disk space sufficient[/green]")
    
    # Check for required directories
    console.print("[bold]Directory Structure:[/bold]")
    required_dirs = ["cache", "models", "uploads", "results", "logs"]
    for dirname in required_dirs:
        dir_path = Path(dirname)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            console.print(f"  Created directory: {dirname}")
        elif not os.access(dir_path, os.W_OK):
            console.print(f"[red]  ✗ Directory {dirname} is not writable[/red]")
        else:
            console.print(f"[green]  ✓ Directory {dirname} exists and is writable[/green]")
    
    # Check for model downloads
    console.print("[bold]Model Availability:[/bold]")
    models_required = {
        "Whisper": "~2-10GB depending on size",
        "PyAnnote": "~1GB", 
        "NLLB Translation": "~2.5GB",
        "French Wav2Vec2": "~1GB"
    }
    for model_name, size in models_required.items():
        console.print(f"  {model_name}: {size} required (will be downloaded on first use)")

    # Check for PyTorch compatibility
    console.print("[bold]PyTorch:[/bold]")
    console.print(f"  Version: {torch.__version__}")
    if torch.cuda.is_available():
        console.print(f"  CUDA Version: {torch.version.cuda}")
        console.print("[green]  ✓ PyTorch CUDA support available[/green]")
    else:
        console.print("[yellow]  ⚠ PyTorch CUDA support not available[/yellow]")
    
    # Overall assessment
    console.print("\n[bold]Overall Assessment:[/bold]")
    if not torch.cuda.is_available() or total_gb < 8:
        console.print("[yellow]System meets minimum requirements but may have performance limitations[/yellow]")
    else:
        console.print("[green]System meets recommended requirements for ASR processing[/green]")
    
    console.print("\n[bold]Recommended Next Steps:[/bold]")
    console.print("1. Run 'python main.py create-config' to generate a configuration file")
    console.print("2. Run 'python main.py process-file sample.wav' to process a single file")
    console.print("3. Run 'python main.py start-server' to start the API server")

@app_cli.command()
def cache_cleanup(
    force: bool = typer.Option(
        False, 
        "--force", 
        "-f", 
        help="Force complete cache removal"
    )
):
    """Clean up cache files and temporary data."""
    try:
        # Initialize basic settings
        settings = ASRSettings()
        cache_dir = settings.CACHE_DIR
        
        console.print("[bold]Cache Cleanup[/bold]")
        
        # Check cache size
        cache_size = 0
        file_count = 0
        for path in cache_dir.glob('**/*'):
            if path.is_file():
                cache_size += path.stat().st_size
                file_count += 1
        
        cache_size_gb = cache_size / (1024**3)
        console.print(f"Current cache size: {cache_size_gb:.2f}GB ({file_count} files)")
        
        if force:
            # Complete removal
            console.print("[yellow]Performing complete cache cleanup...[/yellow]")
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            console.print("[green]Cache directory completely cleared[/green]")
        else:
            # Selective cleanup
            console.print("Performing selective cache cleanup...")
            
            # Initialize cache manager
            cache_manager = CacheManager(
                cache_dir=str(cache_dir),
                max_size_gb=settings.MAX_CACHE_SIZE / (1024**3),
                ttl_hours=24,
                cleanup_interval=1
            )
            
            # Run cleanup
            asyncio.run(cache_manager.cleanup())
            
            # Calculate new size
            new_cache_size = 0
            new_file_count = 0
            for path in cache_dir.glob('**/*'):
                if path.is_file():
                    new_cache_size += path.stat().st_size
                    new_file_count += 1
            
            new_cache_size_gb = new_cache_size / (1024**3)
            saved = cache_size_gb - new_cache_size_gb
            
            console.print(f"[green]Cleanup completed: {saved:.2f}GB freed[/green]")
            console.print(f"New cache size: {new_cache_size_gb:.2f}GB ({new_file_count} files)")
        
        console.print("\nFor complete cache removal, use --force flag")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Cache cleanup failed: {str(e)}")
        raise typer.Exit(code=1)

def main():
    """Entry point with Windows compatibility and enhanced error handling."""
    # Windows-compatible event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Ensure critical directories exist
    for dirname in ["cache", "models", "uploads", "results", "logs"]:
        Path(dirname).mkdir(parents=True, exist_ok=True)
    
    try:
        # Banner
        console.print(
            "\n[bold blue]Multilingual ASR System with French Optimization[/bold blue]\n"
            "[cyan]Enhanced for French language processing[/cyan]\n"
        )
        
        # Run CLI app
        app_cli()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        sys.exit(0)
    except typer.Exit as e:
        sys.exit(e.exit_code)
    except Exception as e:
        console.print(f"\n[red bold]Unhandled error:[/red bold] {str(e)}")
        logger.critical(f"Unhandled error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()