import pandas as pd
import re
import numpy as np
from collections import Counter
import nltk
# Importação opcional para uso futuro, 
# mas não usaremos diretamente para evitar erros
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# Configuração inicial do NLTK - baixar recursos necessários
print("Baixando recursos do NLTK necessários...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Função para carregar dados do Excel
def carregar_planilha(caminho_arquivo):
    try:
        # Tente carregar o arquivo
        df = pd.read_excel(caminho_arquivo)
        print(f"Planilha carregada com sucesso. Encontradas {len(df)} linhas.")
        return df
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None

# Função para pré-processar o texto
def preprocessar_texto(texto):
    if pd.isna(texto):
        return ""
    
    # Converter para minúsculas
    texto = str(texto).lower()
    
    # Remover caracteres especiais
    texto = re.sub(r'[^\w\s]', ' ', texto)
    
    # Tokenização simples (sem usar word_tokenize para evitar problemas com o idioma)
    tokens = texto.split()
    
    # Lista de stopwords em português
    stop_words_pt = ['a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'até', 
                    'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 
                    'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'essa', 
                    'essas', 'esse', 'esses', 'esta', 'estas', 'este', 'estes', 'eu', 'foi', 'foram', 
                    'há', 'isso', 'isto', 'já', 'lhe', 'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 
                    'meus', 'minha', 'minhas', 'muito', 'muitos', 'na', 'não', 'nas', 'nem', 'no', 'nos', 
                    'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o', 'os', 'ou', 'para', 
                    'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'se', 'sem', 
                    'seu', 'seus', 'sua', 'suas', 'são', 'só', 'também', 'te', 'tem', 'têm', 'teu', 'teus', 
                    'tu', 'tua', 'tuas', 'um', 'uma', 'você', 'vocês', 'vos', 'à', 'às']
    
    # Remover stopwords
    tokens = [w for w in tokens if not w in stop_words_pt and len(w) > 2]
    
    return ' '.join(tokens)

# Definir as categorias existentes e novas
def definir_categorias():
    # Categorias existentes (exemplo - você deve ajustar conforme suas categorias reais)
    categorias_existentes = [
        "Material de Escritório", "Equipamentos de Informática", "Materiais de Limpeza",
        "Móveis e Utensílios", "Material de Copa e Cozinha", "Material Elétrico",
        "Material Hidráulico", "Ferramentas", "Material de Construção",
        "Material de Segurança", "Material de Decoração", "Material Hospitalar",
        "Material de Laboratório", "Material Esportivo", "Material Didático",
        "Material de Jardinagem", "Material Automotivo", "Material de Embalagem"
    ]
    
    # Novas categorias a serem criadas (exemplo - ajuste conforme necessário)
    novas_categorias = [
        "Equipamentos de Áudio e Vídeo", "Material de Telecomunicações", 
        "Equipamentos de Medição", "Material Gráfico", "Uniformes e Vestuário",
        "Material para Eventos", "Suprimentos para Impressoras", 
        "Equipamentos de Refrigeração", "Acessórios para Computadores",
        "Material de Sinalização", "Equipamentos de Proteção Individual", 
        "Material de Acondicionamento"
    ]
    
    todas_categorias = categorias_existentes + novas_categorias
    return todas_categorias

# Função para extrair características dos textos
def extrair_caracteristicas(textos_preprocessados):
    # Criar um bag of words simples
    todas_palavras = []
    for texto in textos_preprocessados:
        palavras = texto.split()
        todas_palavras.extend(palavras)
    
    # Contar frequência de palavras
    contador = Counter(todas_palavras)
    palavras_comuns = [palavra for palavra, count in contador.most_common(500)]
    
    # Criar características para cada texto
    caracteristicas = []
    for texto in textos_preprocessados:
        palavras = set(texto.split())
        feats = {}
        for palavra in palavras_comuns:
            feats[palavra] = (palavra in palavras)
        caracteristicas.append(feats)
    
    return caracteristicas

# Definição de categorias com termos, pesos e expressões contextuais
def criar_dicionario_categorias():
    return {
        "Material de Escritório": {
            "termos_fortes": {  # Termos que são muito específicos desta categoria (peso alto)
                "grampeador": 10, "clips": 10, "fichário": 10, "borracha": 8, 
                "agenda executiva": 10, "pasta suspensa": 10, "pasta az": 10,
                "papel a4": 10, "papel oficio": 10, "marcador permanente": 8
            },
            "termos": {  # Termos gerais da categoria (peso médio)
                "caneta": 5, "papel": 3, "pasta": 3, "arquivo": 4, "etiqueta": 4, 
                "lápis": 5, "caderno": 5, "agenda": 5, "corretivo": 5, "clipe": 5,
                "grampo": 5, "post it": 6, "porta caneta": 6, "lapiseira": 6
            },
            "expressoes": [  # Expressões com múltiplas palavras
                "material escritorio", "artigo escritorio", "escritorio papelaria"
            ],
            "contexto": {  # Palavras que quando aparecem juntas são indicativas
                "papel": ["sulfite", "a4", "oficio", "carta", "almaço"],
                "caneta": ["esferografica", "gel", "marca texto", "hidrográfica"],
                "pasta": ["suspensa", "az", "catalogo", "arquivo"]
            }
        },
        
        "Equipamentos de Informática": {
            "termos_fortes": {
                "computador": 10, "notebook": 10, "monitor": 9, "mouse": 8, "teclado": 8,
                "impressora": 10, "scanner": 10, "servidor": 10, "roteador": 10, 
                "switch": 10, "nobreak": 10, "cpu": 10, "desktop": 10, "workstation": 10
            },
            "termos": {
                "estabilizador": 6, "hd": 6, "ssd": 7, "pendrive": 7, "memória ram": 8,
                "placa mãe": 8, "processador": 7, "periférico": 6, "bluetooth": 5,
                "gabinete": 6, "fonte": 3, "wi-fi": 6, "internet": 4
            },
            "expressoes": [
                "equipamento informatica", "acessorio computador", "hardware computacional"
            ],
            "contexto": {
                "cabo": ["hdmi", "vga", "usb", "rede", "dados", "ethernet", "dvi"],
                "placa": ["vídeo", "rede", "mãe", "som", "pci"],
                "fonte": ["alimentação", "atx", "computador", "pc"]
            }
        },
        
        "Materiais de Limpeza": {
            "termos_fortes": {
                "detergente": 9, "desinfetante": 10, "vassoura": 10, "rodo": 10, 
                "água sanitária": 10, "sabão em pó": 10, "alvejante": 10, "saco lixo": 10
            },
            "termos": {
                "sabão": 5, "limpador": 7, "pano": 3, "cera": 5, "lustra": 6, 
                "esponja": 5, "mop": 8, "limpeza": 6, "higienização": 7, "multiuso": 7,
                "limpa": 5, "lixeira": 8, "limpador": 7
            },
            "expressoes": [
                "material limpeza", "produto limpeza", "artigo limpeza", "higiene limpeza"
            ],
            "contexto": {
                "pano": ["chão", "limpeza", "prato", "microfibra"],
                "saco": ["lixo", "resíduo"],
                "limpador": ["vidro", "multiuso", "banheiro", "piso"]
            }
        },
        
        "Móveis e Utensílios": {
            "termos_fortes": {
                "mesa escritório": 10, "cadeira giratória": 10, "armário": 9, 
                "estante": 8, "gaveteiro": 10, "escrivaninha": 10, "sofá": 10
            },
            "termos": {
                "mesa": 6, "cadeira": 6, "poltrona": 8, "prateleira": 6, 
                "bancada": 7, "suporte": 3, "móvel": 8, "estofado": 7,
                "longarina": 9, "divisória": 7, "balcão": 7
            },
            "expressoes": [
                "mobiliario escritorio", "movel escritorio", "movel corporativo", "mobilia"
            ],
            "contexto": {
                "mesa": ["escritório", "trabalho", "reunião", "atendimento"],
                "cadeira": ["ergonômica", "escritório", "giratória", "fixa"],
                "armário": ["aço", "madeira", "mdf", "alto", "baixo"]
            }
        },
        
        "Material de Copa e Cozinha": {
            "termos_fortes": {
                "xícara": 10, "prato": 9, "talher": 10, "garrafa térmica": 10, 
                "cafeteira": 10, "microondas": 10, "açucareiro": 10, "porta copos": 10
            },
            "termos": {
                "copo": 7, "garrafa": 5, "térmica": 6, "café": 4, "açúcar": 6, 
                "chaleira": 8, "filtro": 3, "bandeja": 6, "copa": 8, "cozinha": 7,
                "adoçante": 8, "colher": 7, "jarra": 8
            },
            "expressoes": [
                "material copa", "utensilio cozinha", "artigo copa", "acessorio copa"
            ],
            "contexto": {
                "copo": ["descartável", "vidro", "água", "café"],
                "filtro": ["café", "água"],
                "bandeja": ["servir", "refeição", "copa"]
            }
        },
        
        "Material Elétrico": {
            "termos_fortes": {
                "lâmpada led": 10, "disjuntor": 10, "interruptor": 10, "tomada": 10, 
                "reator": 10, "luminária": 10, "eletroduto": 10, "quadro distribuição": 10
            },
            "termos": {
                "lâmpada": 7, "fio": 5, "extensão": 6, "plug": 7, 
                "elétrico": 7, "conduíte": 8, "resistência": 6, "instalação": 4,
                "led": 5, "fluorescente": 7, "fase": 6
            },
            "expressoes": [
                "material eletrico", "componente eletrico", "acessorio eletrico", "instalacao eletrica"
            ],
            "contexto": {
                "cabo": ["elétrico", "energia", "pp", "flexível", "força"],
                "fio": ["elétrico", "condutor", "energia", "mm"],
                "tomada": ["elétrica", "embutir", "sobrepor"]
            }
        },
        
        "Material Hidráulico": {
            "termos_fortes": {
                "válvula": 9, "registro": 9, "torneira": 10, "sifão": 10, 
                "flange": 10, "bóia": 10, "hidrômetro": 10, "engate": 10
            },
            "termos": {
                "cano": 7, "tubo": 5, "conexão": 6, "vedação": 7, 
                "adesivo": 3, "fita": 2, "hidráulico": 8, "água": 3,
                "pvc": 6, "esgoto": 7, "joelho": 7, "tê": 7
            },
            "expressoes": [
                "material hidraulico", "encanamento agua", "sistema hidraulico", 
                "instalacao hidraulica"
            ],
            "contexto": {
                "tubo": ["pvc", "água", "esgoto", "hidráulico"],
                "conexão": ["hidráulica", "pvc", "água"],
                "fita": ["veda rosca", "teflon", "vedação"]
            }
        },
        
        "Ferramentas": {
            "termos_fortes": {
                "alicate": 10, "martelo": 10, "furadeira": 10, "serra": 9, 
                "chave fenda": 10, "esmerilhadeira": 10, "parafusadeira": 10,
                "marreta": 10, "picareta": 10, "formão": 10, "serrote": 10
            },
            "termos": {
                "chave": 7, "parafuso": 5, "broca": 8, "porca": 6, 
                "trena": 8, "nível": 7, "esquadro": 8, "ferramenta": 8,
                "pá": 7, "enxada": 9, "talhadeira": 9, "lima": 7
            },
            "expressoes": [
                "ferramenta manual", "ferramenta eletrica", "equipamento ferramenta", "kit ferramenta"
            ],
            "contexto": {
                "chave": ["fenda", "philips", "allen", "inglesa", "boca"],
                "serra": ["circular", "mármore", "tico-tico", "manual"],
                "cabo": ["madeira", "ferramenta", "emborrachado"]
            }
        },
        
        "Material de Construção": {
            "termos_fortes": {
                "cimento": 10, "tijolo": 10, "telha": 10, "argamassa": 10, 
                "gesso": 9, "concreto": 10, "betoneira": 10, "massa corrida": 10
            },
            "termos": {
                "areia": 7, "madeira": 5, "prego": 6, "massa": 4, 
                "pedra": 6, "brita": 8, "construção": 7, "obra": 6,
                "impermeabilizante": 8, "reboco": 8, "cal": 7, "bloco": 7
            },
            "expressoes": [
                "material construcao", "material obra", "insumo construcao", "produto construcao"
            ],
            "contexto": {
                "massa": ["corrida", "acrílica", "cimentícia", "reboco"],
                "prego": ["com cabeça", "sem cabeça", "polegadas"],
                "madeira": ["construção", "obra", "eucalipto", "pinus"]
            }
        },
        
        "Material de Segurança": {
            "termos_fortes": {
                "câmera segurança": 10, "alarme": 10, "sensor presença": 10, "cofre": 10, 
                "fechadura": 9, "cadeado": 9, "extintor": 10, "dvr": 10
            },
            "termos": {
                "câmera": 6, "sensor": 6, "controle": 3, "segurança": 6, 
                "sirene": 8, "trava": 7, "vigilância": 8, "monitoramento": 7,
                "cftv": 10, "ip": 3, "digital": 2
            },
            "expressoes": [
                "sistema seguranca", "equipamento seguranca", "dispositivo seguranca", 
                "controle acesso"
            ],
            "contexto": {
                "câmera": ["vigilância", "monitoramento", "segurança", "cftv"],
                "controle": ["acesso", "remoto", "segurança"],
                "sensor": ["presença", "movimento", "abertura"]
            }
        },
        
        "Material de Decoração": {
            "termos_fortes": {
                "quadro decorativo": 10, "cortina": 10, "tapete": 10, "vaso ornamental": 10, 
                "luminária decorativa": 10, "espelho decorativo": 10, "ornamento": 10
            },
            "termos": {
                "quadro": 6, "enfeite": 7, "almofada": 8, "espelho": 6, 
                "moldura": 7, "relógio": 5, "decoração": 8, "decorativo": 8,
                "estátua": 8, "porta retrato": 8, "objeto": 2
            },
            "expressoes": [
                "item decorativo", "artigo decoracao", "peca decorativa", "adorno"
            ],
            "contexto": {
                "quadro": ["decorativo", "parede", "decoração", "pintura"],
                "vaso": ["decorativo", "cerâmica", "flores", "ornamental"],
                "relógio": ["parede", "decorativo", "mesa"]
            }
        },
        
        "Material Hospitalar": {
            "termos_fortes": {
                "seringa": 10, "agulha": 10, "luva procedimento": 10, "esparadrapo": 10, 
                "atadura": 10, "esfigmomanômetro": 10, "estetoscópio": 10, "maca": 10,
                "cateter": 10, "desfibrilador": 10, "bisturi": 10
            },
            "termos": {
                "hospitalar": 8, "gaze": 8, "máscara": 4, "curativo": 8, 
                "médico": 6, "saúde": 3, "procedimento": 5, "enfermagem": 7,
                "ambulatorial": 8, "cirúrgico": 8, "descartável": 3
            },
            "expressoes": [
                "material hospitalar", "material medico", "equipamento hospitalar", 
                "produto hospitalar"
            ],
            "contexto": {
                "luva": ["procedimento", "cirúrgica", "látex", "nitrilo"],
                "máscara": ["cirúrgica", "hospitalar", "procedimento"],
                "termômetro": ["clínico", "digital", "infravermelho"]
            }
        },
        
        "Material de Laboratório": {
            "termos_fortes": {
                "pipeta": 10, "microscópio": 10, "lâmina": 8, "proveta": 10, 
                "béquer": 10, "erlenmeyer": 10, "centrífuga": 10, "balança analítica": 10
            },
            "termos": {
                "tubo ensaio": 9, "balança": 5, "estufa": 6, "laboratório": 8, 
                "análise": 5, "ensaio": 7, "reagente": 9, "experimento": 7,
                "científico": 7, "calibração": 6
            },
            "expressoes": [
                "material laboratorio", "equipamento laboratorio", "vidraria laboratorio", 
                "instrumento analise"
            ],
            "contexto": {
                "tubo": ["ensaio", "laboratório", "falcon", "coleta"],
                "balança": ["analítica", "precisão", "digital", "laboratório"],
                "lâmina": ["microscópio", "análise", "laboratório"]
            }
        },
        
        "Material Esportivo": {
            "termos_fortes": {
                "bola oficial": 10, "rede esportiva": 10, "raquete": 10, "kimono": 10, 
                "tatame": 10, "peso academia": 10, "esteira": 10, "bicicleta ergométrica": 10
            },
            "termos": {
                "bola": 7, "rede": 4, "uniforme": 5, "jogo": 3, 
                "esporte": 7, "tênis": 5, "apito": 6, "atletismo": 8,
                "fitness": 8, "academia": 7, "esportivo": 7, "treino": 6
            },
            "expressoes": [
                "material esportivo", "equipamento esportivo", "artigo esporte", 
                "acessorio esportivo"
            ],
            "contexto": {
                "bola": ["futebol", "vôlei", "basquete", "oficial"],
                "uniforme": ["esportivo", "time", "treino"],
                "rede": ["vôlei", "futebol", "tênis", "esportiva"]
            }
        },
        
        "Material Didático": {
            "termos_fortes": {
                "livro didático": 10, "apostila": 10, "lousa": 10, "giz escolar": 10, 
                "pincel quadro": 10, "globo terrestre": 10, "atlas": 10, "mapa escolar": 10
            },
            "termos": {
                "livro": 5, "giz": 8, "pincel": 4, "marcador": 5, 
                "quadro": 4, "educativo": 8, "pedagógico": 9, "didático": 9,
                "escolar": 8, "ensino": 7, "educação": 6, "aprendizagem": 8
            },
            "expressoes": [
                "material didatico", "recurso pedagogico", "kit educativo", 
                "material escolar"
            ],
            "contexto": {
                "pincel": ["quadro", "branco", "atômico", "escolar"],
                "livro": ["didático", "escolar", "pedagógico", "ensino"],
                "quadro": ["branco", "negro", "aviso", "escolar"]
            }
        },
        
        "Material de Jardinagem": {
            "termos_fortes": {
                "regador": 10, "cortador grama": 10, "vaso planta": 10, "adubo": 10, 
                "tesoura poda": 10, "pulverizador": 9, "mangueira jardim": 10, "rastelo": 10
            },
            "termos": {
                "terra": 5, "vaso": 5, "planta": 7, "semente": 8, 
                "mangueira": 5, "jardim": 8, "tesoura": 4, "poda": 9,
                "gramado": 8, "flor": 7, "jardinagem": 9, "jardineiro": 8
            },
            "expressoes": [
                "material jardinagem", "ferramenta jardim", "equipamento jardinagem", 
                "acessorio jardim"
            ],
            "contexto": {
                "tesoura": ["poda", "jardim", "jardinagem"],
                "vaso": ["planta", "flor", "jardim", "cerâmica"],
                "terra": ["vegetal", "adubada", "jardim", "plantio"]
            }
        },
        
        "Material Automotivo": {
            "termos_fortes": {
                "óleo motor": 10, "filtro ar": 10, "filtro óleo": 10, "pneu": 10, 
                "bateria automotiva": 10, "macaco hidráulico": 10, "chave roda": 10, "vela ignição": 10
            },
            "termos": {
                "óleo": 6, "filtro": 5, "bateria": 6, "auto": 7, 
                "carro": 7, "moto": 7, "veículo": 7, "combustível": 8,
                "motor": 6, "automotivo": 8, "freio": 8, "automóvel": 7
            },
            "expressoes": [
                "peca automotiva", "acessorio automotivo", "componente veiculo", 
                "suprimento automotivo"
            ],
            "contexto": {
                "óleo": ["motor", "lubrificante", "sintético", "automotivo"],
                "filtro": ["ar", "óleo", "combustível", "cabine"],
                "bateria": ["automotiva", "carro", "veículo"]
            }
        },
        
        "Material de Embalagem": {
            "termos_fortes": {
                "caixa papelão": 10, "fita adesiva": 9, "papel bolha": 10, "stretch": 10, 
                "etiqueta adesiva": 9, "embalagem plástica": 10, "embalagem presente": 10
            },
            "termos": {
                "caixa": 6, "fita": 4, "papelão": 8, "plástico": 4, 
                "saco": 5, "embalagem": 8, "pacote": 6, "sacola": 7,
                "papel": 3, "bolha": 8, "envelope": 7, "isopor": 7
            },
            "expressoes": [
                "material embalagem", "suprimento embalagem", "produto embalagem", 
                "artigo embalagem"
            ],
            "contexto": {
                "caixa": ["papelão", "embalagem", "presente", "transporte"],
                "fita": ["adesiva", "empacotamento", "dupla face", "lacre"],
                "papel": ["presente", "embrulho", "kraft", "bolha"]
            }
        },
        
        # Novas categorias
        "Equipamentos de Áudio e Vídeo": {
            "termos_fortes": {
                "caixa som": 10, "microfone": 10, "projetor": 10, "datashow": 10, 
                "alto-falante": 10, "amplificador": 10, "mesa som": 10, "filmadora": 10
            },
            "termos": {
                "som": 6, "caixa

# Função para categorizar itens com base nas descrições
def categorizar_itens(df, coluna_descricao='B'):
    # Pré-processar as descrições
    descricoes = df[coluna_descricao].apply(preprocessar_texto)
    
    # Criar dicionário de palavras-chave por categoria
    dicionario_categorias = criar_dicionario_categorias()
    todas_categorias = list(dicionario_categorias.keys())
    
    # Inicializar coluna de nova categoria
    df['Nova Categoria'] = 'Não Categorizado'
    
    # Categorizar cada item
    for idx, desc in enumerate(descricoes):
        if not desc:  # Se a descrição estiver vazia
            continue
            
        pontuacao_categorias = {cat: 0 for cat in todas_categorias}
        
        # Contar ocorrências de palavras-chave em cada descrição
        for categoria, palavras_chave in dicionario_categorias.items():
            for palavra in palavras_chave:
                if palavra in desc:
                    pontuacao_categorias[categoria] += 1
        
        # Encontrar a categoria com maior pontuação
        if max(pontuacao_categorias.values()) > 0:
            melhor_categoria = max(pontuacao_categorias.items(), key=lambda x: x[1])[0]
            df.at[idx, 'Nova Categoria'] = melhor_categoria
    
    return df

# Função principal
def main():
    # Solicitar o caminho do arquivo
    caminho_arquivo = input("Digite o caminho do arquivo Excel (.xlsx): ")
    
    # Carregar a planilha
    df = carregar_planilha(caminho_arquivo)
    
    if df is not None:
        # Exibir informações iniciais
        print("\nPrimeiras linhas da planilha:")
        print(df.head())
        
        # Verificar colunas disponíveis
        print("\nColunas disponíveis:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        
        # Solicitar a coluna que contém as descrições
        coluna_descricao = input("\nDigite o nome ou índice da coluna que contém as descrições dos itens: ")
        
        # Converter índice numérico para nome da coluna se necessário
        try:
            coluna_idx = int(coluna_descricao)
            coluna_descricao = df.columns[coluna_idx]
        except ValueError:
            pass
        
        # Categorizar os itens
        print("\nCategorizando itens...")
        df_categorizado = categorizar_itens(df, coluna_descricao)
        
        # Exibir estatísticas de categorização
        contagem_categorias = df_categorizado['Nova Categoria'].value_counts()
        print("\nDistribuição de itens por categoria:")
        print(contagem_categorias)
        
        # Salvar o resultado
        caminho_saida = caminho_arquivo.replace('.xlsx', '_categorizado.xlsx')
        df_categorizado.to_excel(caminho_saida, index=False)
        print(f"\nPlanilha categorizada salva como: {caminho_saida}")

if __name__ == "__main__":
    main()
