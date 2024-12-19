import logging
from colorama import Fore, Style, init

# Definisci i colori per ogni livello di log
COLORS = {
    # 'INFO': Fore.CYAN + Style.BRIGHT,
    'DEBUG': Fore.GREEN + Style.BRIGHT,
    'WARNING': Fore.YELLOW + Style.BRIGHT,
    'ERROR': Fore.RED + Style.BRIGHT,
    'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        # Ottiene il colore per il livello di log specifico
        log_color = COLORS.get(record.levelname, "")
        log_message = super().format(record)
        return f"{log_color}{log_message}{Style.RESET_ALL}"

class Logger:
    def __init__(self, level=logging.INFO):
        # Inizializza Colorama per i colori cross-platform
        init(autoreset=True)

        # Configura il logger principale
        self.logger = logging.getLogger("ColorLogger")
        self.logger.setLevel(level)

        # Configura il gestore di stream (console)
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter("[%(levelname)s] %(message)s"))
        
        # Aggiunge il gestore solo una volta
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def set_level(self, level=logging.INFO):
        """Modifica il livello di logging."""
        self.logger.setLevel(level)
        
    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)