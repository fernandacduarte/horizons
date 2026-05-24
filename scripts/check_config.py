"""Smoke test: load the default config via Hydra and print it."""
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print("Config loaded successfully. Contents:\n")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
