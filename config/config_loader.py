import yaml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Configuration loader from YAML files"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize config loader
        
        Args:
            config_path: Path to config.yaml file
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration values"""
        for key, value in updates.items():
            keys = key.split('.')
            config = self.config
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
    
    def save(self, output_path: str) -> None:
        """Save configuration to file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.get(key)
    
    def __str__(self) -> str:
        """String representation"""
        return yaml.dump(self.config, default_flow_style=False)


# Global config instance
_global_config = None


def get_config(config_path: str = None) -> ConfigLoader:
    """Get or create global config instance"""
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigLoader(config_path)
    
    return _global_config
