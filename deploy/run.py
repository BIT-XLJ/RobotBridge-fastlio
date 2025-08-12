import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig

from loguru import logger

@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="base"
)
def main(cfg: DictConfig) -> None:
    
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, 'eval.log')
    logger.remove()
    logger.add(hydra_log_path, level='DEBUG')

    console_log_level = os.environ.get('LOGURU_LEVEL', 'INFO').upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logger.info(f'Log saved to {hydra_log_path}')

    agent = instantiate(cfg.agent)
    agent.run()


if __name__=="__main__":
    main()