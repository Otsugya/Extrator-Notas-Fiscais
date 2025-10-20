from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import PyPDF2
import json
import sqlite3
from datetime import datetime
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configurar Gemini AI
GEMINI_API_KEY = "AIzaSyAvag5ZD7lFydA4NVcM6a6AsMjUaSfmk7A"
genai.configure(api_key=GEMINI_API_KEY)

CATEGORIAS_DESPESAS = {
    "INSUMOS AGRÍCOLAS": ["sementes", "fertilizantes", "defensivos agrícolas", "corretivos"],
    "MANUTENÇÃO E OPERAÇÃO": ["combustíveis", "lubrificantes", "peças", "parafusos", "componentes mecânicos", "manutenção", "pneus", "filtros", "correias", "ferramentas", "utensílios", "diesel", "óleo"],
    "RECURSOS HUMANOS": ["mão de obra", "salários", "encargos"],
    "SERVIÇOS OPERACIONAIS": ["frete", "transporte", "colheita", "secagem", "armazenagem", "pulverização", "aplicação"],
    "INFRAESTRUTURA E UTILIDADES": ["energia elétrica", "arrendamento", "construções", "reformas", "materiais de construção", "material hidráulico"],
    "ADMINISTRATIVAS": ["honorários", "contábeis", "advocatícios", "agronômicos", "despesas bancárias", "financeiras"],
    "SEGUROS E PROTEÇÃO": ["seguro agrícola", "seguro de ativos", "seguro prestamista"],
    "IMPOSTOS E TAXAS": ["ITR", "IPTU", "IPVA", "INCRA-CCIR"],
    "INVESTIMENTOS": ["aquisição de máquinas", "implementos", "veículos", "imóveis", "infraestrutura rural"]
}

def init_db():
    conn = sqlite3.connect('sistema.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS Pessoas (
        idPessoas INTEGER PRIMARY KEY AUTOINCREMENT,
        tipo VARCHAR(45),
        razaosocial VARCHAR(150),
        fantasia VARCHAR(150),
        documento VARCHAR(45),
        status VARCHAR(45) DEFAULT 'ATIVO'
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS Classificacao (
        idClassificacao INTEGER PRIMARY KEY AUTOINCREMENT,
        tipo VARCHAR(45),
        descricao VARCHAR(150),
        status VARCHAR(45) DEFAULT 'ATIVO'
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS MovimentoContas (
        idMovimentoContas INTEGER PRIMARY KEY AUTOINCREMENT,
        tipo VARCHAR(45),
        numeronotafiscal VARCHAR(45),
        dataemissao DATE,
        descricao VARCHAR(300),
        status VARCHAR(45) DEFAULT 'ATIVO',
        valortotal DECIMAL(10,2),
        Pessoas_idFornecedorCliente INTEGER,
        Pessoas_idFaturado INTEGER,
        FOREIGN KEY (Pessoas_idFornecedorCliente) REFERENCES Pessoas(idPessoas),
        FOREIGN KEY (Pessoas_idFaturado) REFERENCES Pessoas(idPessoas)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS ParcelasContas (
        idParcelasContas INTEGER PRIMARY KEY AUTOINCREMENT,
        Identificacao VARCHAR(45),
        datavencimento DATE,
        valorparcela DECIMAL(10,2),
        valorpago DECIMAL(10,2),
        valorsaldo DECIMAL(10,2),
        statusparcela VARCHAR(45),
        MovimentoContas_idMovimentoContas INTEGER,
        FOREIGN KEY (MovimentoContas_idMovimentoContas) REFERENCES MovimentoContas(idMovimentoContas)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS MovimentoContas_has_Classificacao (
        MovimentoContas_idMovimentoContas INTEGER,
        Classificacao_idClassificacao INTEGER,
        PRIMARY KEY (MovimentoContas_idMovimentoContas, Classificacao_idClassificacao),
        FOREIGN KEY (MovimentoContas_idMovimentoContas) REFERENCES MovimentoContas(idMovimentoContas),
        FOREIGN KEY (Classificacao_idClassificacao) REFERENCES Classificacao(idClassificacao)
    )''')
    
    conn.commit()
    conn.close()

init_db()

def extrair_texto_pdf(arquivo_pdf):
    try:
        pdf_reader = PyPDF2.PdfReader(arquivo_pdf)
        return "\n".join([p.extract_text() for p in pdf_reader.pages])
    except Exception as e:
        return f"Erro ao extrair texto do PDF: {str(e)}"

def classificar_despesa(descricao_produtos):
    descricao_lower = descricao_produtos.lower()
    for categoria, palavras_chave in CATEGORIAS_DESPESAS.items():
        if any(palavra in descricao_lower for palavra in palavras_chave):
            return categoria
    return "OUTRAS DESPESAS"

def processar_nota_fiscal_gemini(texto_pdf):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""Analise o seguinte texto de uma nota fiscal e extraia as informações em formato JSON:

        {texto_pdf}

        Retorne APENAS um JSON válido com a seguinte estrutura:
        {{
            "numero": "número da nota fiscal",
            "serie": "série da nota fiscal",
            "dataEmissao": "data de emissão no formato YYYY-MM-DD",
            "fornecedor": {{
                "razaoSocial": "razão social do fornecedor",
                "fantasia": "nome fantasia do fornecedor",
                "cnpj": "CNPJ do fornecedor"
            }},
            "faturado": {{
                "nomeCompleto": "nome completo do cliente/faturado",
                "cpf": "CPF do cliente (se disponível, senão null)"
            }},
            "itens": [
                {{
                    "descricao": "descrição do produto/serviço",
                    "quantidade": "quantidade"
                }}
            ],
            "parcelas": [
                {{
                    "numero": 1,
                    "dataVencimento": "data de vencimento no formato YYYY-MM-DD",
                    "valor": "valor da parcela em número"
                }}
            ],
            "valorTotal": "valor total da nota em número",
            "classificacaoDespesa": ["categoria da despesa baseada nos produtos"]
        }}

        Se alguma informação não estiver disponível, use null. Para datas, use sempre o formato YYYY-MM-DD."""
        
        response = model.generate_content(prompt)
        resposta_texto = response.text.strip().replace('```json', '').replace('```', '').strip()
        dados_extraidos = json.loads(resposta_texto)
        
        descricao_produtos = " ".join([item.get("descricao", "") for item in dados_extraidos.get("itens", [])])
        categoria_automatica = classificar_despesa(descricao_produtos)
        
        if not dados_extraidos.get("classificacaoDespesa") or dados_extraidos["classificacaoDespesa"] == [None]:
            dados_extraidos["classificacaoDespesa"] = [categoria_automatica]
        
        return dados_extraidos
    except Exception as e:
        return {"erro": f"Erro ao processar com Gemini: {str(e)}"}

def consultar_ou_criar_pessoa(tipo, razao_social, documento):
    conn = sqlite3.connect('sistema.db')
    c = conn.cursor()
    
    c.execute('SELECT idPessoas FROM Pessoas WHERE documento = ? AND tipo = ?', (documento, tipo))
    resultado = c.fetchone()
    
    if resultado:
        conn.close()
        return {"existe": True, "id": resultado[0], "tipo": tipo, "razaoSocial": razao_social, "documento": documento}
    
    c.execute('INSERT INTO Pessoas (tipo, razaosocial, documento) VALUES (?, ?, ?)', (tipo, razao_social, documento))
    conn.commit()
    novo_id = c.lastrowid
    conn.close()
    
    return {"existe": False, "id": novo_id, "tipo": tipo, "razaoSocial": razao_social, "documento": documento}

def consultar_ou_criar_classificacao(descricao):
    conn = sqlite3.connect('sistema.db')
    c = conn.cursor()
    
    c.execute('SELECT idClassificacao FROM Classificacao WHERE descricao = ? AND tipo = ?', (descricao, 'DESPESA'))
    resultado = c.fetchone()
    
    if resultado:
        conn.close()
        return {"existe": True, "id": resultado[0], "descricao": descricao}
    
    c.execute('INSERT INTO Classificacao (tipo, descricao) VALUES (?, ?)', ('DESPESA', descricao))
    conn.commit()
    novo_id = c.lastrowid
    conn.close()
    
    return {"existe": False, "id": novo_id, "descricao": descricao}

def criar_movimento(dados, id_fornecedor, id_faturado, ids_classificacao):
    conn = sqlite3.connect('sistema.db')
    c = conn.cursor()
    
    descricao_itens = ", ".join([item.get("descricao", "") for item in dados.get("itens", [])])
    
    c.execute('''INSERT INTO MovimentoContas 
        (tipo, numeronotafiscal, dataemissao, descricao, valortotal, Pessoas_idFornecedorCliente, Pessoas_idFaturado)
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
        ('APAGAR', dados.get('numero'), dados.get('dataEmissao'), descricao_itens, 
         dados.get('valorTotal'), id_fornecedor, id_faturado))
    
    id_movimento = c.lastrowid
    
    for parcela in dados.get('parcelas', []):
        identificacao = f"NF{dados.get('numero')}-P{parcela.get('numero', 1)}"
        c.execute('''INSERT INTO ParcelasContas 
            (Identificacao, datavencimento, valorparcela, valorsaldo, statusparcela, MovimentoContas_idMovimentoContas)
            VALUES (?, ?, ?, ?, ?, ?)''',
            (identificacao, parcela.get('dataVencimento'), parcela.get('valor'), 
             parcela.get('valor'), 'ABERTO', id_movimento))
    
    for id_class in ids_classificacao:
        c.execute('INSERT INTO MovimentoContas_has_Classificacao VALUES (?, ?)', (id_movimento, id_class))
    
    conn.commit()
    conn.close()
    
    return id_movimento

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({"erro": "Nenhum arquivo PDF foi enviado"}), 400
        
        arquivo = request.files['pdf']
        if not arquivo.filename or not arquivo.filename.lower().endswith('.pdf'):
            return jsonify({"erro": "Arquivo deve ser um PDF"}), 400
        
        texto_pdf = extrair_texto_pdf(BytesIO(arquivo.read()))
        if texto_pdf.startswith("Erro"):
            return jsonify({"erro": texto_pdf}), 400
        
        dados_extraidos = processar_nota_fiscal_gemini(texto_pdf)
        return jsonify(dados_extraidos)
    except Exception as e:
        return jsonify({"erro": f"Erro interno do servidor: {str(e)}"}), 500

@app.route('/processar', methods=['POST'])
def processar_dados():
    try:
        dados = request.json
        
        # Consultar/Criar FORNECEDOR
        fornecedor = consultar_ou_criar_pessoa(
            'FORNECEDOR',
            dados['fornecedor']['razaoSocial'],
            dados['fornecedor']['cnpj']
        )
        
        # Consultar/Criar FATURADO
        faturado = consultar_ou_criar_pessoa(
            'FATURADO',
            dados['faturado']['nomeCompleto'],
            dados['faturado'].get('cpf', 'SEM_CPF')
        )
        
        # Consultar/Criar DESPESAS
        classificacoes = []
        for despesa in dados.get('classificacaoDespesa', []):
            classificacoes.append(consultar_ou_criar_classificacao(despesa))
        
        # Criar MOVIMENTO
        id_movimento = criar_movimento(
            dados,
            fornecedor['id'],
            faturado['id'],
            [c['id'] for c in classificacoes]
        )
        
        return jsonify({
            "sucesso": True,
            "mensagem": "REGISTRO FOI LANÇADO COM SUCESSO",
            "fornecedor": fornecedor,
            "faturado": faturado,
            "despesas": classificacoes,
            "movimentoId": id_movimento
        })
    except Exception as e:
        return jsonify({"erro": f"Erro ao processar: {str(e)}"}), 500

@app.route('/categorias')
def get_categorias():
    return jsonify(list(CATEGORIAS_DESPESAS.keys()))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)