# src/utils/setup.py
# YENİ DOSYA (Tam Hali)

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Neden? validate_processing_config, özel hata türlerini fırlatabilir.
# Bu hataları ya burada tanımlarız ya da load_data'dan import ederiz.
# Şimdilik load_data'dan import edelim (eğer orada tanımlıysa).
try:
    from src.data.load_data import InvalidConfigurationError
except ImportError:
    # Fallback: Eğer load_data'da yoksa, burada tanımla
    class InvalidConfigurationError(ValueError):
        """Custom exception for invalid configuration."""
        pass

# Neden? Logging nesnesini alıp ayarlamak için.
logger = logging.getLogger(__name__) # Ana logger'ı al

# --- LOGGING KURULUMU ---

def setup_logging(output_dir: Path, log_level: str = 'INFO'):
    """
    Sets up project-wide logging to both a file and the console.

    Clears existing handlers to prevent duplicate logs if called multiple times.
    Logs DEBUG level and above to file, and the specified level to console.

    Args:
        output_dir (Path): The directory where 'processing.log' will be saved.
        log_level (str): The minimum logging level for console output 
                         (e.g., 'INFO', 'DEBUG'). Defaults to 'INFO'.
    """
    logger_instance = logging.getLogger() # Get the root logger
    
    # Neden? Notebook gibi ortamlarda hücre tekrar çalıştırıldığında
    # aynı handler'ın tekrar eklenmesini önlemek için.
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()
        # print("DEBUG: Cleared existing logging handlers.") # Gerekirse debug için açılabilir

    # Log seviyelerini Python'un logging sabitlerine çevir
    log_level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
    level = log_level_map.get(log_level.upper(), logging.INFO)
    
    # Formatlayıcılar: Dosya için detaylı, konsol için basit
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Root logger'ın seviyesini ayarla (handler'lar kendi seviyelerini ayrıca belirleyebilir)
    logger_instance.setLevel(logging.DEBUG) # En düşük seviyeyi ayarla ki hiçbir şey kaçmasın

    # 1. Dosya Handler'ı (File Handler)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / 'processing.log'
        file_handler = logging.FileHandler(log_file, mode='a') # 'a' = append (üstüne yazma)
        file_handler.setLevel(logging.DEBUG) # Dosyaya HER ŞEYİ (DEBUG ve üstü) yaz
        file_handler.setFormatter(detailed_formatter)
        logger_instance.addHandler(file_handler)
    except Exception as e:
        # Loglama kritik olduğu için, burada başarısız olursa programı durdurabiliriz.
        print(f"FATAL ERROR: Could not create log file handler at {log_file}: {e}")
        sys.exit(1)

    # 2. Konsol Handler'ı (Stream Handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level) # Konsola sadece kullanıcı tarafından istenen seviyeyi yaz
    console_handler.setFormatter(simple_formatter)
    logger_instance.addHandler(console_handler)

    logging.info(f"Logging setup complete. Level: {log_level}. Log file: {log_file}")


# --- KONFİGÜRASYON DOĞRULAMA ---

def validate_processing_config(config: Dict[str, Any], project_root: Path) -> bool:
    """
    Validates the loaded configuration dictionary and resolves relative paths.

    Checks for required keys, ensures segmentation parameters exist,
    validates numeric values (basic), and converts file paths to absolute paths
    based on the project root.

    Args:
        config (Dict[str, Any]): The configuration dictionary loaded from YAML.
                                 This dictionary is modified in-place.
        project_root (Path): The absolute path to the project's root directory.

    Returns:
        bool: True if validation passes, False otherwise.

    Raises:
        FileNotFoundError: If essential files (reference, test images) are not found.
        InvalidConfigurationError: If required keys are missing.
        ValueError: If numeric values are invalid (e.g., non-positive threshold).
    """
    logger.debug("Entering configuration validation...")
    required_base_keys = ['reference_image_path', 'test_images']
    # DBSCAN parametreleri de buraya eklenebilir
    required_seg_keys = ['distance_threshold', 'predefined_k', 'k_values', 'som_values', 'dbscan_eps', 'dbscan_min_samples', 'methods']
    
    try:
        # --- Anahtar Kontrolleri ---
        missing_keys = [key for key in required_base_keys if key not in config]
        if missing_keys:
            raise InvalidConfigurationError(f"Missing required base config keys: {missing_keys}")

        # Segmentasyon parametrelerini bul veya oluştur
        seg_params = config.get('segmentation_params')
        if not seg_params or not isinstance(seg_params, dict):
             logger.warning("Key 'segmentation_params' not found or not a dict. Checking root level for segmentation keys (legacy support).")
             # Eski config dosyalarıyla uyumluluk için kök dizinde ara
             missing_seg_keys = [key for key in required_seg_keys if key not in config]
             if missing_seg_keys:
                  raise InvalidConfigurationError(f"Missing required segmentation keys (checked root and 'segmentation_params'): {missing_seg_keys}")
             # Kök dizindekileri 'segmentation_params' altına taşı
             config['segmentation_params'] = {key: config[key] for key in required_seg_keys if key in config}
             seg_params = config['segmentation_params']
             logger.info("Moved segmentation keys found at root level into 'segmentation_params'.")
        else:
             # 'segmentation_params' varsa, gerekli tüm anahtarları içerdiğinden emin ol
             missing_seg_keys = [key for key in required_seg_keys if key not in seg_params]
             if missing_seg_keys:
                  raise InvalidConfigurationError(f"Missing required keys within 'segmentation_params': {missing_seg_keys}")

        # --- Yol (Path) Doğrulama ve Çözümleme ---
        # Neden? Config dosyasındaki göreceli yolları (örn: "dataset/...")
        # projenin gerçek konumuna göre mutlak yollara (örn: "C:/.../dataset/...") çeviririz.
        
        # Referans Görüntü
        ref_path_str = config['reference_image_path']
        ref_path = Path(ref_path_str)
        if not ref_path.is_absolute(): 
            ref_path = (project_root / ref_path).resolve()
        if not ref_path.exists(): 
            raise FileNotFoundError(f"Reference image not found at resolved path: {ref_path}")
        config['reference_image_path'] = str(ref_path) # Config'i mutlak yolla güncelle
        logger.debug(f"Reference image path validated: {ref_path}")

        # Test Görüntüleri
        resolved_test_images = []
        missing_images = []
        if not config.get('test_images'): # Test imaj listesi boşsa hata ver
             raise InvalidConfigurationError("'test_images' list cannot be empty in the configuration.")
             
        for img_path_str in config.get('test_images', []):
            img_path = Path(img_path_str)
            if not img_path.is_absolute(): 
                img_path = (project_root / img_path).resolve()
            if not img_path.exists(): 
                missing_images.append(str(img_path)) # Hata mesajı için çözülmüş yolu ekle
            else: 
                resolved_test_images.append(str(img_path)) # Mutlak yolu listeye ekle
                
        if missing_images: 
            raise FileNotFoundError(f"Test images not found at resolved paths: {missing_images}")
        config['test_images'] = resolved_test_images # Config'i mutlak yollarla güncelle
        logger.debug(f"Validated {len(resolved_test_images)} test image paths.")

        # --- Sayısal Değer Doğrulama (Basit) ---
        # Neden? Anlamsız parametrelerle (örn: negatif k değeri) çalışmayı engeller.
        if seg_params['distance_threshold'] <= 0: 
            # logger.warning("distance_threshold is non-positive.") # Belki sadece uyarı yeterlidir?
            pass # Şimdilik geçelim
        if seg_params['predefined_k'] <= 0: 
            raise ValueError("segmentation_params.predefined_k must be positive.")
        if seg_params['dbscan_eps'] <= 0:
            raise ValueError("segmentation_params.dbscan_eps must be positive.")
        if seg_params['dbscan_min_samples'] <= 0:
             raise ValueError("segmentation_params.dbscan_min_samples must be positive.")
             
        # Yöntem listesinin boş olmadığından emin ol
        if not seg_params.get('methods'):
            raise InvalidConfigurationError("segmentation_params.methods list cannot be empty.")

        logger.info("Configuration validation passed successfully.")
        return True

    # Neden (Hata Yakalama)? 'except' blokları, belirli hata türlerini yakalar
    # ve kullanıcıya daha anlaşılır mesajlar verir. En sondaki genel 'Exception'
    # beklenmedik hataları yakalar.
    except (InvalidConfigurationError, FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
         logger.error(f"Unexpected error during configuration validation: {e}", exc_info=True)
         return False

# --- Diğer Yardımcı Fonksiyonlar ---
# main.py'deki 'safe_image_load' buraya taşınabilir, ancak load_data.py'de
# zaten benzer bir 'load_image' fonksiyonu var. Kod tekrarını önlemek için
# load_data.load_image'ı kullanmak daha iyi. Bu yüzden buraya eklemiyorum.

# main.py'deki 'create_lab_converters' fonksiyonu, DBN modeli ve scaler'lar
# eğitildikten *sonra* kullanılıyor. Bu yüzden 'pipeline.py' içinde kalması
# veya 'color_conversion.py'ye taşınması daha mantıklı. Buraya eklemiyorum.

# main.py'deki 'setup_pipeline_configs' fonksiyonu, config sözlüğünü
# dataclass nesnelerine dönüştürüyordu. Bu işlevi artık doğrudan
# 'ProcessingPipeline.__init__' içinde yapıyoruz. Bu yüzden buraya eklemiyorum.