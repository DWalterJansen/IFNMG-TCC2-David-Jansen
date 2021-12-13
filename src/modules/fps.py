import time # Lib para calcular o FPS

# Classe para auxiliar no cálculo do FPS durante a execução
class FPS:

    def __init__(self):
        self.ptime = 0 # Tempo do processo anterior (previos)
        self.ctime = 0 # Tempo do processo atual (current)

    def update_ptime(self):
        self.ptime = self.ctime

    def update_ctime(self):
        self.ctime = time.time()

    def compute(self) -> int:
        self.update_ctime()
        fps = 1 / (self.ctime - self.ptime)
        self.update_ptime()
        return int(fps)
