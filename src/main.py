# src/main.py
# YENİ VE TEMİZLENMİŞ HALİ

import sys
import os
import argparse
import logging
import cProfile
import pstats
import time
import traceback
from pathlib import Path

# --- Adım 1: Proje Kök Dizinini Bul ---
# Neden? Bu kod, projenin mutlak yolunu dinamik olarak belirler.
# Sizin ve diğer geliştiricilerin farklı dosya yollarına sahip olmasını çözer.
try:
    SCRIPT_DIR = Path(__file__).parent.absolute()
    PROJECT_ROOT = SCRIPT_DIR.parent # src'den bir üst klasöre (PRINTS) çık
    sys.path.insert(0, str(PROJECT_ROOT)) # Proje klasörünü PATH'e ekle
except Exception as e:
    print(f"FATAL ERROR determining paths: {e}")
    sys.exit(1)

# --- Adım 2: Sadece Gerekli Modülleri Import Et ---
# Neden? main.py'nin tek sorumluluğu programı başlatmaktır.
# İş mantığını (pipeline) ve yardımcı fonksiyonları (setup) buradan çağırır.
try:
    from src.pipeline import ProcessingPipeline # Ana iş akış sınıfımız
    from src.utils.setup import setup_logging # Logging ayarlama fonksiyonumuz
    # Diğer kütüphaneler (numpy, cv2) artık pipeline veya diğer alt modüllerden import ediliyor
except ImportError as e:
    print(f"FATAL ERROR during critical module import (Pipeline/Setup): {e}")
    print("Lütfen 'pipeline.py' ve 'src/utils/setup.py' dosyalarının varlığını kontrol edin.")
    traceback.print_exc()
    sys.exit(1)

# TensorFlow Ayarları (main.py'de kalmalı)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Ana `main` Fonksiyonu ---
# Neden? Bu fonksiyon, programın YÜRÜTME ADIMLARINI (execution steps) tanımlar.
def main(config_path: str, log_level: str, output_dir: Path, profile: bool):
    """
    Ana çalıştırma fonksiyonu. Loglamayı kurar ve ProcessingPipeline'ı başlatır.
    """
    profiler = None
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
        
    start_time = time.perf_counter()
    
    try:
        # 1. Logging'i başlat
        setup_logging(output_dir, log_level)
        
        # 2. Pipeline'ı oluştur
        # Bütün ağır işi bu sınıfa devrediyoruz.
        pipeline = ProcessingPipeline(
            config_path=config_path, 
            output_dir=output_dir, 
            project_root=PROJECT_ROOT
        )
        
        # 3. Pipeline'ı çalıştır
        pipeline.run()
        logging.info("Main execution finished successfully.")

    except Exception as e:
        logging.critical(f"An unexpected error occurred in main execution: {e}", exc_info=True)
    finally:
        total_time = time.perf_counter() - start_time
        logging.info("=" * 80 + f"\nPROCESSING COMPLETED IN {total_time:.2f} SECONDS\n" + "=" * 80)
        
        if profile and profiler:
            profiler.disable()
            try:
                # Profilleme sonuçlarını kaydet
                stats = pstats.Stats(profiler).sort_stats('cumtime')
                profile_path = output_dir / 'profile_stats.txt'
                with open(profile_path, 'w') as f:
                    stats.stream = f
                    stats.print_stats(30)
                logging.info(f"Profiling results saved to: {profile_path}")
            except Exception as e:
                logging.warning(f"Failed to save profiling results: {e}")

# --- if __name__ == "__main__": block ---
# Neden? Burası programın GİRİŞ NOKTASI'dır (entry point).
# Sadece argümanları ayrıştırmalı ve main() fonksiyonunu çağırmalıdır.
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Textile Color Analysis System using PSO-optimized DBN"
    )
    # Config argümanını artık zorunlu tutalım (default'u kaldırdım, main() içinde çözülüyordu)
    parser.add_argument('--config', type=str, required=True, help='Path to the specific pattern configuration YAML file')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Set logging level')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--profile', action='store_true', help='Enable detailed profiling output')
    
    args = parser.parse_args()

    # Çıktı dizinini belirle
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir).resolve()
    else:
        OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Config yolunu çöz (göreceli yolları mutlak hale getirir)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    
    if not config_path.exists():
         print(f"❌ Error: Configuration file not found: {config_path}")
         sys.exit(1)

    try:
        # Başlatma mesajları
        print(f"Starting Textile Color Analysis System...")
        # ... (Diğer başlangıç mesajları) ...
        print("-" * 60)

        # Ana fonksiyonu çağır
        main(
            config_path=str(config_path), 
            log_level=args.log_level, 
            output_dir=OUTPUT_DIR,
            profile=args.profile
        )

    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)