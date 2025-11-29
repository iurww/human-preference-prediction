from .configs import CONFIG, print_config
from .random_seed import set_random_seeds


set_random_seeds(CONFIG['seed'])