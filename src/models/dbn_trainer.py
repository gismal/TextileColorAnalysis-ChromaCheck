import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.config_types import TrainConfig
# Proje içi importlar
# pso_dbn'den DBN, DBNConfig, PSOConfig ve PSOOptimizer'ı import ediyoruz
from src.models.pso_dbn import DBN, DBNConfig, PSOConfig, PSOOptimizer 
# preprocess'ten efficient_data_sampling'i import ediyoruz
from src.data.sampling import efficient_data_sampling

logger = logging.getLogger(__name__)
class DBNTrainer:
    """
    DBN modelini eğitmek ve PSO ile optimize etmek için
    tüm mantığı kapsayan sınıf.
    """
    def __init__(self, dbn_config: DBNConfig, pso_config: PSOConfig, train_config: TrainConfig):
        """
        Trainer'ı konfigürasyon nesneleriyle başlatır.
        
        Args:
            dbn_config (DBNConfig): DBN mimarisi için ayarlar.
            pso_config (PSOConfig): PSO optimizasyonu için ayarlar.
            train_config (TrainConfig): Örnekleme ve test boyutu gibi eğitim ayarları.
        """
        self.dbn_config = dbn_config
        self.pso_config = pso_config
        self.train_config = train_config
        
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.dbn: Optional[DBN] = None
        self.x_train_scaled: Optional[np.ndarray] = None
        self.y_train_scaled: Optional[np.ndarray] = None
        logger.info("DBNTrainer initialized.")

    def train(self, rgb_data: List, lab_data: List) -> Tuple[DBN, Dict[str, MinMaxScaler]]:
        """
        Verilen ham RGB ve LAB verilerini kullanarak DBN modelini eğitir ve optimize eder.

        Args:
            rgb_data (List): Ham RGB görüntü verileri listesi.
            lab_data (List): Ham LAB görüntü verileri listesi.

        Returns:
            Tuple[DBN, Dict[str, MinMaxScaler]]: 
                Eğitilmiş DBN modeli ve 'scaler_x', 'scaler_y', 'scaler_y_ab'
                içeren bir sözlük (dictionary).
        """
        try:
            logger.info("Starting DBN training and PSO optimization process...")
            
            # 1. Veri Örnekleme (main.py'den taşındı)
            rgb_samples, lab_samples = efficient_data_sampling(
                rgb_data, lab_data, train_config=self.train_config
            )
            
            # 2. Veriyi Bölme (main.py'den taşındı)
            x_train, x_test, y_train, y_test = train_test_split(
                rgb_samples, lab_samples, 
                test_size=self.train_config.test_size, 
                random_state=self.train_config.random_state
            )
            logger.info(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

            # 3. Veriyi Ölçekleme (main.py'den taşındı)
            self._prepare_scalers(x_train, y_train)
            
            # 4. DBN Modelini Başlatma (main.py'den taşındı)
            self.dbn = DBN(input_size=3, output_size=3, config=self.dbn_config)
            sample_input = np.zeros((1, 3))
            self.dbn.model(sample_input)  # Model ağırlıklarını başlat
            initial_weights = self.dbn.model.get_weights()
            
            logger.info(f"Initial model has {len(initial_weights)} weight layers")
            for i, w in enumerate(initial_weights):
                logger.info(f"Layer {i}: shape {w.shape}, range [{w.min():.3f}, {w.max():.3f}]")

            # 5. PSO ile Optimizasyon (main.py'deki 'safe_pso_optimization' yerine geçti)
            optimized_weights = self._run_pso_with_retries(initial_weights)
            self.dbn.model.set_weights(optimized_weights)
            
            logger.info("DBN training and PSO optimization complete.")
            
            if not self.dbn or not self.scalers:
                 raise RuntimeError("Training failed to produce a model or scalers.")
                 
            return self.dbn, self.scalers

        except Exception as e:
            logger.error(f"Error during DBN training: {e}", exc_info=True)
            raise

    def _prepare_scalers(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Veri ölçekleyicileri oluşturur, eğitir ve sınıfta saklar."""
        logger.info("Preparing and fitting scalers...")
        
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))      # L kanalı (0, 1)
        scaler_y_ab = MinMaxScaler(feature_range=(0, 1))   # a,b kanalları (0, 1)
        
        # x_train_scaled'i _run_pso_with_retries'de kullanmak için self olarak saklıyoruz
        self.x_train_scaled = scaler_x.fit_transform(x_train)
        
        y_l_scaled = scaler_y.fit_transform(y_train[:, [0]])
        y_ab_scaled = scaler_y_ab.fit_transform(y_train[:, 1:])
        # y_train_scaled'i _run_pso_with_retries'de kullanmak için self olarak saklıyoruz
        self.y_train_scaled = np.hstack((y_l_scaled, y_ab_scaled))
        
        self.scalers = {
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'scaler_y_ab': scaler_y_ab
        }
        logger.info(f"Scaled training data - X: {self.x_train_scaled.shape}, Y: {self.y_train_scaled.shape}")
        logger.info(f"X range: [{self.x_train_scaled.min():.3f}, {self.x_train_scaled.max():.3f}]")
        logger.info(f"Y range: [{self.y_train_scaled.min():.3f}, {self.y_train_scaled.max():.3f}]")

    def _run_pso_with_retries(self, initial_weights: List[np.ndarray]) -> List[np.ndarray]:
        """PSO optimizasyonunu dener ve başarısız olursa yeniden dener."""
        
        current_weights = initial_weights
        
        for attempt in range(self.train_config.pso_retries):
            try:
                logger.info(f"PSO optimization attempt {attempt + 1}/{self.train_config.pso_retries}")
                
                # Doğrudan modern PSOOptimizer sınıfını kullanıyoruz
                optimizer = PSOOptimizer(self.pso_config)
                
                if self.dbn is None or self.x_train_scaled is None or self.y_train_scaled is None:
                    raise RuntimeError("DBN or scaled data is not initialized.")
                
                optimized_weights = optimizer.optimize(
                    self.dbn, self.x_train_scaled, self.y_train_scaled
                )
                
                if any(np.isnan(w).any() or np.isinf(w).any() for w in optimized_weights):
                    raise ValueError("PSO produced invalid weights (NaN or Inf)")
                
                logger.info("PSO optimization completed successfully")
                return optimized_weights
                
            except Exception as e:
                logger.warning(f"PSO attempt {attempt + 1} failed: {e}")
                if attempt < self.train_config.pso_retries - 1:
                    logger.info("Retrying with slightly different initialization...")
                    # Ağırlıklara küçük bir gürültü ekleyerek yeniden deneme
                    current_weights = [w + np.random.normal(0, 0.001, w.shape) for w in current_weights]
                    if self.dbn:
                        self.dbn.model.set_weights(current_weights)
                else:
                    logger.error("All PSO attempts failed. Using last valid weights.")
        
        # Tüm denemeler başarısız olursa son geçerli ağırlıkları (veya başlangıç) döndür
        logger.warning("Returning last known valid weights (or initial weights) after all PSO retries failed.")
        return current_weights