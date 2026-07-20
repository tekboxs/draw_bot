# Draw Bot

Bot que transforma uma imagem em movimentos de mouse — ele "desenha" a imagem
em qualquer programa que aceite desenho livre (Paint, quadros online, jogos etc.)
controlando o cursor.

## Como funciona

O caminho da imagem até o desenho é:

1. **Imagem → pontos** — lê a imagem em tons de cinza e pega os pixels escuros.
2. **Agrupar em traços** — junta pontos vizinhos em grupos (union-find + KD-Tree).
3. **Ordenar e simplificar** — para cada traço, ordena os pontos numa rota curta
   (vizinho mais próximo) e remove pontos redundantes (Douglas-Peucker).
4. **Desenhar** — move o mouse traço a traço, descendo e levantando a "caneta".

## Estrutura

```
draw_bot/
├── main.py            # todo o pipeline (imagem → desenho)
├── requirements.txt   # dependências
├── images/            # imagens de exemplo
└── README.md
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

1. Ajuste `IMAGEM` e `LIMIAR` no bloco `__main__` do `main.py`.
2. Rode:
   ```bash
   python main.py
   ```
3. Posicione o mouse no canto onde o desenho deve começar.
4. Pressione **`d`** para iniciar.

### Parâmetros úteis

| Parâmetro    | O que faz                                                        |
|--------------|------------------------------------------------------------------|
| `LIMIAR`     | Quanto maior, mais pixels contam como traço (0–255).             |
| `AMOSTRAGEM` | Pula pixels para reduzir a densidade (1 = usa todos).            |
