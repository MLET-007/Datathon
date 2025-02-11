from datetime import datetime
import matplotlib.pyplot as plt

# função para converter um timestamp em milissegundos para uma data formatada
def converter_timestamp(timestamp):
    return datetime.fromtimestamp(int(timestamp) / 1000).strftime('%d/%m/%Y %H:%M')

# função para processar a coluna com múltiplos timestamps
def processar_timestamps(linha):
    # Dividir a string em timestamps individuais
    timestamps = linha.split(',')
    # Converter cada timestamp para o formato de data
    datas_formatadas = [converter_timestamp(ts) for ts in timestamps]
    # Juntar as datas formatadas em uma única string, separadas por ';'
    return ','.join(datas_formatadas)

# função para contar os valores separados por vírgula
def contar_valores(string):
    return len(string.split(','))

def plotar_histograma_com_estatisticas(coluna, bins=10, cor_hist='blue', alpha_hist=0.7):
    """
    Plota um histograma de uma coluna de dados com marcações para:
    - Primeiro quantil (Q1)
    - Mediana (Q2)
    - Terceiro quantil (Q3)
    - Média
    - Desvio padrão (média - std e média + std)
    - Valor máximo

    Parâmetros:
    -----------
    coluna : pd.Series ou array-like
        Coluna de dados para plotar o histograma.
    bins : int, opcional
        Número de intervalos (bins) no histograma. Padrão é 10.
    cor_hist : str, opcional
        Cor do histograma. Padrão é 'blue'.
    alpha_hist : float, opcional
        Transparência do histograma. Padrão é 0.7.
    """
    # Calcular as estatísticas
    q1 = coluna.quantile(0.25)  # Primeiro quantil (Q1)
    q2 = coluna.quantile(0.50)  # Segundo quantil (Mediana, Q2)
    q3 = coluna.quantile(0.75)  # Terceiro quantil (Q3)
    media = coluna.mean()       # Média
    std = coluna.std()          # Desvio padrão
    maximo = coluna.max()       # Valor máximo

    # Plotar o histograma
    plt.hist(coluna, bins=bins, edgecolor='black', color=cor_hist, alpha=alpha_hist, label='Histograma')

    # Adicionar linhas verticais para as estatísticas
    plt.axvline(q1, color='red', linestyle='--', label=f'Q1: {q1:.2f}')
    plt.axvline(q2, color='green', linestyle='--', label=f'Mediana (Q2): {q2:.2f}')
    plt.axvline(q3, color='blue', linestyle='--', label=f'Q3: {q3:.2f}')
    plt.axvline(media, color='purple', linestyle='-', label=f'Média: {media:.2f}')
    plt.axvline(media - std, color='orange', linestyle=':', label=f'Média - Std: {media - std:.2f}')
    plt.axvline(media + std, color='orange', linestyle=':', label=f'Média + Std: {media + std:.2f}')
    plt.axvline(maximo, color='black', linestyle='-.', label=f'Máximo: {maximo:.2f}')

    # Adicionar título e rótulos
    plt.title('Histograma com Estatísticas')
    plt.xlabel('Valores')
    plt.ylabel('Frequência')

    # Adicionar legenda
    plt.legend()

    # Mostrar o gráfico
    plt.show()