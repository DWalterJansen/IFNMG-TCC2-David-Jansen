from typing import Any, Dict
import argparse # Lib para argumentos de execução

def argsParser() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--src", help="Fonte para detecção: 0: webcam, 1: vídeo, 2: imagem", required=True, metavar="", type=int)
    ap.add_argument("-p", "--path", help="Caminho para o vídeo ou imagem", required=False, metavar="", type=str)

    args = vars(ap.parse_args())

    if not (0 <= args['src'] and args['src'] <= 3) :
        ap.error("--src deve ser 0, 1 ou 2")
    elif args['src'] != 0 and args['path'] is None :
        ap.error("--path requerido para detecção em vídeo ou imagem")

    return args