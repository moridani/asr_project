from typing import Dict, Any, List, Optional, Union, Tuple
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Pipeline,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)
import numpy as np
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import json
import os
from dotenv import load_dotenv

load_dotenv()
class Translator:
    def __init__(
        self,
        device: torch.device,
        cache_manager: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize translator with optimized models and configurations.
        
        Args:
            device: torch device to use
            cache_manager: Cache manager instance
            config: Optional configuration parameters
        """
        self.device = device
        self.cache_manager = cache_manager
        self.config = self._load_config(config)
        self._initialize_models()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load and validate configuration."""
        default_config = {
            'model_paths': {
                'french': os.getenv("FRENCH_MODEL_PATH", "Helsinki-NLP/opus-mt-fr-en"),
                'nllb': os.getenv("NLLB_MODEL_PATH", "facebook/nllb-200-1.3B")
            },
            'batch_size': 8,
            'max_length': 512,
            'beam_size': 4,
            'cache_dir': 'cache/translation_models'
        }
        if config:
            default_config.update(config)
        return default_config

    def _initialize_models(self) -> None:
        """Initialize translation models with optimizations."""
        try:
            # French-specific model (highest priority)
            self.fr_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config['model_paths']['french'],
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
                cache_dir=Path(self.config['cache_dir']) / 'french'
            ).to(self.device)
            self.fr_tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_paths']['french'],
                cache_dir=Path(self.config['cache_dir']) / 'french'
            )

            # NLLB model for other languages
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config['model_paths']['nllb'],
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
                cache_dir=Path(self.config['cache_dir']) / 'nllb'
            ).to(self.device)
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_paths']['nllb'],
                cache_dir=Path(self.config['cache_dir']) / 'nllb'
            )

            logger.info("Translation models loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing translation models: {str(e)}")
            raise

    async def translate_segments(
        self,
        transcription: Dict[str, Any],
        diarization_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Translate non-English segments with optimized processing.
        
        Args:
            transcription: Transcription results
            diarization_result: Speaker diarization results
            
        Returns:
            Dictionary containing translations and metadata
        """
        try:
            start_time = time.time()
            translations = {}
            
            # Group segments by language for batch processing
            language_groups = self._group_segments_by_language(transcription['segments'])
            
            # Process each language group with progress tracking
            total_segments = sum(len(segs) for segs in language_groups.values())
            processed_segments = 0
            
            for language, segments in language_groups.items():
                if language.lower() in ['en', 'eng', 'english']:
                    continue
                
                # Process in optimized batches
                for i in range(0, len(segments), self.config['batch_size']):
                    batch = segments[i:i + self.config['batch_size']]
                    batch_translations = await self._translate_batch(batch, language)
                    translations.update(batch_translations)
                    
                    processed_segments += len(batch)
                    logger.info(f"Processed {processed_segments}/{total_segments} segments")
            
            processing_time = time.time() - start_time
            
            return {
                'translations': translations,
                'metadata': {
                    'processing_time': processing_time,
                    'languages_processed': list(language_groups.keys()),
                    'segment_count': len(translations),
                    'average_time_per_segment': processing_time / len(translations) if translations else 0
                }
            }

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise

    async def _translate_batch(
        self,
        segments: List[Dict[str, Any]],
        source_lang: str
    ) -> Dict[str, Dict[str, Any]]:
        """Translate a batch of segments with language-specific optimizations."""
        texts = [seg['text'] for seg in segments]
        segment_ids = [seg['id'] for seg in segments]
        
        # Select optimal translation approach based on language
        confidence = 0.85  # default confidence
        
        if source_lang.lower() in ['fr', 'fra', 'french']:
            translations = await self._translate_french(texts)
            confidence = 0.95
            model_used = 'helsinki-fr'
        else:
            translations = await self._translate_nllb_optimized(texts, source_lang)
            confidence = self._get_language_confidence(source_lang)
            model_used = 'nllb'
        
        return {
            seg_id: {
                'text': trans,
                'original': orig_text,
                'confidence': confidence,
                'model_used': model_used,
                'language': source_lang,
                'translation_parameters': self._get_translation_parameters(source_lang)
            }
            for seg_id, trans, orig_text in zip(segment_ids, translations, texts)
        }

    async def _translate_french(self, texts: List[str]) -> List[str]:
        """Optimized French translation."""
        loop = asyncio.get_event_loop()
        
        def _translate():
            try:
                inputs = self.fr_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config['max_length'],
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    translated = self.fr_model.generate(
                        **inputs,
                        max_length=self.config['max_length'],
                        num_beams=self.config['beam_size'],
                        length_penalty=1.0
                    )
                
                return self.fr_tokenizer.batch_decode(
                    translated,
                    skip_special_tokens=True
                )
                
            except Exception as e:
                logger.error(f"French translation failed: {str(e)}")
                return ["" for _ in texts]
        
        return await loop.run_in_executor(self.thread_pool, _translate)

    async def _translate_nllb_optimized(
        self,
        texts: List[str],
        source_lang: str
    ) -> List[str]:
        """Optimized NLLB translation for other languages."""
        loop = asyncio.get_event_loop()
        
        def _translate():
            try:
                src_lang = self._get_nllb_code(source_lang)
                params = self._get_translation_parameters(source_lang)
                
                inputs = self.nllb_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=params['max_length'],
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    translated = self.nllb_model.generate(
                        **inputs,
                        forced_bos_token_id=self.nllb_tokenizer.lang_code_to_id["eng_Latn"],
                        max_length=params['max_length'],
                        num_beams=params['num_beams'],
                        length_penalty=params['length_penalty'],
                        temperature=params['temperature']
                    )
                
                return self.nllb_tokenizer.batch_decode(
                    translated,
                    skip_special_tokens=True
                )
                
            except Exception as e:
                logger.error(f"NLLB translation failed: {str(e)}")
                return ["" for _ in texts]
        
        return await loop.run_in_executor(self.thread_pool, _translate)

    def _get_translation_parameters(self, lang: str) -> Dict[str, Any]:
        """Get optimized translation parameters for specific language."""
        base_params = {
            'max_length': self._get_optimal_length(lang),
            'num_beams': 3,
            'length_penalty': self._get_length_penalty(lang),
            'temperature': self._get_temperature(lang)
        }
        
        # Language-specific adjustments
        if lang.lower() in ['ar', 'ara', 'arabic']:
            base_params.update({
                'num_beams': 4,
                'temperature': 0.6
            })
        elif lang.lower() in ['hi', 'hin', 'hindi']:
            base_params.update({
                'num_beams': 4,
                'temperature': 0.7
            })
        elif lang.lower() in ['zh', 'chi', 'chinese']:
            base_params.update({
                'num_beams': 4,
                'length_penalty': 0.9
            })
            
        return base_params

    def _get_language_confidence(self, lang: str) -> float:
        """Get confidence score based on language support level."""
        confidence_map = {
            'ar': 0.90,  # Arabic
            'hi': 0.88,  # Hindi
            'zh': 0.87,  # Chinese
            'es': 0.86,  # Spanish
            'pt': 0.86,  # Portuguese
            'de': 0.85,  # German
            'it': 0.85,  # Italian
            'ru': 0.84,  # Russian
            'ja': 0.83,  # Japanese
            'ko': 0.83,  # Korean
            'tr': 0.82,  # Turkish
            'vi': 0.82,  # Vietnamese
            'th': 0.81,  # Thai
        }
        return confidence_map.get(lang.lower(), 0.80)

    def _get_optimal_length(self, lang: str) -> int:
        """Get optimal max length based on language characteristics."""
        length_map = {
            'ja': 384,   # Japanese needs shorter segments
            'ko': 384,   # Korean needs shorter segments
            'th': 384,   # Thai needs shorter segments
            'vi': 448,   # Vietnamese
            'zh': 448,   # Chinese
            'ar': 512,   # Arabic
            'default': 512
        }
        return length_map.get(lang.lower(), length_map['default'])

    def _get_length_penalty(self, lang: str) -> float:
        """Get optimal length penalty based on language structure."""
        penalty_map = {
            'ja': 0.8,  # Japanese tends to be more concise
            'ko': 0.8,  # Korean tends to be more concise
            'zh': 0.9,  # Chinese needs moderate penalty
            'vi': 1.1,  # Vietnamese might need expansion
            'th': 1.1,  # Thai might need expansion
            'ar': 1.2,  # Arabic might need more expansion
            'default': 1.0
        }
        return penalty_map.get(lang.lower(), penalty_map['default'])

    def _get_temperature(self, lang: str) -> float:
        """Get optimal temperature based on language complexity."""
        temp_map = {
            'ar': 0.6,  # More conservative for Arabic
            'hi': 0.7,  # More conservative for Hindi
            'zh': 0.7,  # More conservative for Chinese
            'ja': 0.8,  # More conservative for Japanese
            'ko': 0.8,  # More conservative for Korean
            'vi': 0.7,  # More conservative for Vietnamese
            'th': 0.7,  # More conservative for Thai
            'default': 0.6
        }
        return temp_map.get(lang.lower(), temp_map['default'])

    def _get_nllb_code(self, language: str) -> str:
        """Map language to NLLB code."""
        code_map = {
            'ar': 'ara_Arab',
            'hi': 'hin_Deva',
            'zh': 'zho_Hans',
            'ja': 'jpn_Jpan',
            'ko': 'kor_Hang',
            'th': 'tha_Thai',
            'vi': 'vie_Latn',
            'ru': 'rus_Cyrl',
            'default': 'eng_Latn'
        }
        return code_map.get(language.lower(), f"{language.lower()}_Latn")

    def _group_segments_by_language(
        self,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group segments by language for batch processing."""
        groups = {}
        for segment in segments:
            lang = segment.get('language', 'und')
            if lang not in groups:
                groups[lang] = []
            groups[lang].append(segment)
        return groups

    async def cleanup(self):
        """Cleanup resources and free memory."""
        try:
            self.thread_pool.shutdown(wait=True)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("Translation resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")