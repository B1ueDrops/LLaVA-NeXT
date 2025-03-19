class Logger:
    @staticmethod
    def info(message):
        print(f'\033[92m[INFO]\033[0m: {message}')
    
    @staticmethod
    def debug(message):
        print(f'\033[94m[DEBUG]\033[0m: {message}')
