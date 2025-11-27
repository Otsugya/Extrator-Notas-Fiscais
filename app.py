from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import PyPDF2
import json
import sqlite3
from datetime import datetime
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    
    # Tabela para armazenar embeddings (RAG com Embeddings)
    c.execute('''CREATE TABLE IF NOT EXISTS Embeddings (
        idEmbedding INTEGER PRIMARY KEY AUTOINCREMENT,
        tipo_entidade VARCHAR(45),
        id_entidade INTEGER,
        texto_original TEXT,
        embedding TEXT,
        data_criacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    
    # Criar embedding para a nova pessoa
    criar_embedding_pessoa(novo_id, tipo, razao_social, documento)
    
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
    
    # Criar embedding para a nova classificação
    criar_embedding_classificacao(novo_id, descricao)
    
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
    
    # Criar embedding para o novo movimento
    criar_embedding_movimento(id_movimento, dados.get('numero'), dados.get('dataEmissao'), 
                            descricao_itens, dados.get('valorTotal'))
    
    return id_movimento

# ==================== RAG - SISTEMA DE BUSCA INTELIGENTE ====================

def gerar_embedding_texto(texto):
    """Gera embedding usando Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        # Usando o modelo para gerar uma representação vetorial simplificada
        response = model.generate_content(f"Crie um resumo de 50 palavras sobre: {texto}")
        resumo = response.text
        
        # Convertendo para vetor numérico simples (baseado em hash)
        import hashlib
        hash_obj = hashlib.sha256(resumo.encode())
        hash_bytes = hash_obj.digest()
        embedding = [int(b) / 255.0 for b in hash_bytes[:128]]  # 128 dimensões
        
        return embedding
    except:
        # Fallback: criar embedding baseado em frequência de palavras
        palavras = texto.lower().split()
        vocab = list(set(palavras))[:128]
        embedding = [palavras.count(palavra) / len(palavras) for palavra in vocab]
        # Normalizar para 128 dimensões
        while len(embedding) < 128:
            embedding.append(0.0)
        return embedding[:128]

def criar_embedding_pessoa(id_pessoa, tipo, razao_social, documento):
    """Cria embedding para uma pessoa"""
    conn = sqlite3.connect('sistema.db')
    c = conn.cursor()
    
    texto = f"Pessoa {tipo}: {razao_social}, Documento: {documento}"
    embedding = gerar_embedding_texto(texto)
    
    c.execute('''INSERT INTO Embeddings (tipo_entidade, id_entidade, texto_original, embedding)
                 VALUES (?, ?, ?, ?)''',
              ('PESSOA', id_pessoa, texto, json.dumps(embedding)))
    
    conn.commit()
    conn.close()

def criar_embedding_classificacao(id_classificacao, descricao):
    """Cria embedding para uma classificação"""
    conn = sqlite3.connect('sistema.db')
    c = conn.cursor()
    
    texto = f"Classificação de despesa: {descricao}"
    embedding = gerar_embedding_texto(texto)
    
    c.execute('''INSERT INTO Embeddings (tipo_entidade, id_entidade, texto_original, embedding)
                 VALUES (?, ?, ?, ?)''',
              ('CLASSIFICACAO', id_classificacao, texto, json.dumps(embedding)))
    
    conn.commit()
    conn.close()

def criar_embedding_movimento(id_movimento, numero_nf, data_emissao, descricao, valor_total):
    """Cria embedding para um movimento"""
    conn = sqlite3.connect('sistema.db')
    c = conn.cursor()
    
    texto = f"Movimento nota fiscal {numero_nf}, data {data_emissao}, descrição: {descricao}, valor: {valor_total}"
    embedding = gerar_embedding_texto(texto)
    
    c.execute('''INSERT INTO Embeddings (tipo_entidade, id_entidade, texto_original, embedding)
                 VALUES (?, ?, ?, ?)''',
              ('MOVIMENTO', id_movimento, texto, json.dumps(embedding)))
    
    conn.commit()
    conn.close()

def buscar_dados_banco():
    """Busca todos os dados do banco para contexto RAG"""
    conn = sqlite3.connect('sistema.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Buscar pessoas
    c.execute('SELECT * FROM Pessoas')
    pessoas = [dict(row) for row in c.fetchall()]
    
    # Buscar classificações
    c.execute('SELECT * FROM Classificacao')
    classificacoes = [dict(row) for row in c.fetchall()]
    
    # Buscar movimentos com joins
    c.execute('''
        SELECT 
            m.*,
            f.razaosocial as fornecedor_nome,
            f.documento as fornecedor_doc,
            ft.razaosocial as faturado_nome,
            ft.documento as faturado_doc
        FROM MovimentoContas m
        LEFT JOIN Pessoas f ON m.Pessoas_idFornecedorCliente = f.idPessoas
        LEFT JOIN Pessoas ft ON m.Pessoas_idFaturado = ft.idPessoas
    ''')
    movimentos = [dict(row) for row in c.fetchall()]
    
    # Buscar parcelas
    c.execute('SELECT * FROM ParcelasContas')
    parcelas = [dict(row) for row in c.fetchall()]
    
    conn.close()
    
    return {
        "pessoas": pessoas,
        "classificacoes": classificacoes,
        "movimentos": movimentos,
        "parcelas": parcelas
    }

def rag_simples(pergunta):
    """RAG Simples - busca por palavras-chave e contexto direto"""
    dados = buscar_dados_banco()
    
    # Criar contexto com todos os dados
    contexto = f"""
    DADOS DO SISTEMA:
    
    PESSOAS CADASTRADAS ({len(dados['pessoas'])}):
    {json.dumps(dados['pessoas'], indent=2, ensure_ascii=False)}
    
    CLASSIFICAÇÕES DE DESPESAS ({len(dados['classificacoes'])}):
    {json.dumps(dados['classificacoes'], indent=2, ensure_ascii=False)}
    
    MOVIMENTOS FINANCEIROS ({len(dados['movimentos'])}):
    {json.dumps(dados['movimentos'], indent=2, ensure_ascii=False)}
    
    PARCELAS ({len(dados['parcelas'])}):
    {json.dumps(dados['parcelas'], indent=2, ensure_ascii=False)}
    """
    
    # Usar LLM para responder baseado no contexto
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""Você é um assistente especializado em análise financeira e contábil.
        
Baseado EXCLUSIVAMENTE nos dados abaixo, responda a pergunta do usuário de forma clara e detalhada.
Se os dados não contiverem informação suficiente, informe isso ao usuário.

{contexto}

PERGUNTA DO USUÁRIO: {pergunta}

INSTRUÇÕES:
- Responda em português brasileiro
- Seja específico e use números/valores quando disponíveis
- Se não houver dados suficientes, seja honesto sobre isso
- Organize a resposta de forma clara
- Use formatação markdown para melhor legibilidade
"""
        
        response = model.generate_content(prompt)
        return {
            "sucesso": True,
            "resposta": response.text,
            "metodo": "RAG Simples",
            "contexto_usado": True
        }
    except Exception as e:
        return {
            "sucesso": False,
            "erro": f"Erro ao processar pergunta: {str(e)}"
        }

def rag_embeddings(pergunta):
    """RAG com Embeddings - busca por similaridade semântica"""
    try:
        # Gerar embedding da pergunta
        embedding_pergunta = gerar_embedding_texto(pergunta)
        
        # Buscar embeddings armazenados
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        c.execute('SELECT * FROM Embeddings')
        embeddings_db = c.fetchall()
        conn.close()
        
        if not embeddings_db:
            return {
                "sucesso": False,
                "erro": "Nenhum embedding encontrado no banco. Adicione dados primeiro."
            }
        
        # Calcular similaridade
        similaridades = []
        for emb in embeddings_db:
            id_emb, tipo_ent, id_ent, texto_orig, emb_json, data_cria = emb
            embedding_db = json.loads(emb_json)
            
            # Calcular similaridade de cosseno
            sim = cosine_similarity([embedding_pergunta], [embedding_db])[0][0]
            similaridades.append({
                "id": id_emb,
                "tipo": tipo_ent,
                "id_entidade": id_ent,
                "texto": texto_orig,
                "similaridade": float(sim)
            })
        
        # Ordenar por similaridade
        similaridades.sort(key=lambda x: x['similaridade'], reverse=True)
        top_5 = similaridades[:5]
        
        # Buscar dados completos das entidades mais relevantes
        dados_relevantes = buscar_dados_entidades(top_5)
        
        # Criar contexto com dados mais relevantes
        contexto = f"""
        DADOS MAIS RELEVANTES PARA A PERGUNTA:
        
        {json.dumps(dados_relevantes, indent=2, ensure_ascii=False)}
        
        SIMILARIDADES ENCONTRADAS:
        {json.dumps([{"tipo": t["tipo"], "similaridade": f"{t['similaridade']:.2%}", "texto": t["texto"]} for t in top_5], indent=2, ensure_ascii=False)}
        """
        
        # Usar LLM para responder
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""Você é um assistente especializado em análise financeira e contábil.

Baseado nos dados mais relevantes encontrados por busca semântica, responda a pergunta do usuário.

{contexto}

PERGUNTA DO USUÁRIO: {pergunta}

INSTRUÇÕES:
- Responda em português brasileiro
- Use os dados mais relevantes encontrados
- Seja específico e detalhado
- Use formatação markdown
- Mencione o nível de relevância dos dados quando apropriado
"""
        
        response = model.generate_content(prompt)
        return {
            "sucesso": True,
            "resposta": response.text,
            "metodo": "RAG com Embeddings",
            "similaridades": top_5,
            "contexto_usado": True
        }
    except Exception as e:
        return {
            "sucesso": False,
            "erro": f"Erro ao processar com embeddings: {str(e)}"
        }

def buscar_dados_entidades(entidades_relevantes):
    """Busca dados completos das entidades encontradas como relevantes"""
    conn = sqlite3.connect('sistema.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    resultado = {
        "pessoas": [],
        "classificacoes": [],
        "movimentos": []
    }
    
    for ent in entidades_relevantes:
        if ent['tipo'] == 'PESSOA':
            c.execute('SELECT * FROM Pessoas WHERE idPessoas = ?', (ent['id_entidade'],))
            pessoa = c.fetchone()
            if pessoa:
                resultado['pessoas'].append(dict(pessoa))
                
        elif ent['tipo'] == 'CLASSIFICACAO':
            c.execute('SELECT * FROM Classificacao WHERE idClassificacao = ?', (ent['id_entidade'],))
            classif = c.fetchone()
            if classif:
                resultado['classificacoes'].append(dict(classif))
                
        elif ent['tipo'] == 'MOVIMENTO':
            c.execute('''
                SELECT 
                    m.*,
                    f.razaosocial as fornecedor_nome,
                    ft.razaosocial as faturado_nome
                FROM MovimentoContas m
                LEFT JOIN Pessoas f ON m.Pessoas_idFornecedorCliente = f.idPessoas
                LEFT JOIN Pessoas ft ON m.Pessoas_idFaturado = ft.idPessoas
                WHERE m.idMovimentoContas = ?
            ''', (ent['id_entidade'],))
            movimento = c.fetchone()
            if movimento:
                resultado['movimentos'].append(dict(movimento))
    
    conn.close()
    return resultado

# ==================== ROTAS ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crud')
def crud():
    return render_template('crud.html')

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
        
        fornecedor = consultar_ou_criar_pessoa(
            'FORNECEDOR',
            dados['fornecedor']['razaoSocial'],
            dados['fornecedor']['cnpj']
        )
        
        faturado = consultar_ou_criar_pessoa(
            'FATURADO',
            dados['faturado']['nomeCompleto'],
            dados['faturado'].get('cpf', 'SEM_CPF')
        )
        
        classificacoes = []
        for despesa in dados.get('classificacaoDespesa', []):
            classificacoes.append(consultar_ou_criar_classificacao(despesa))
        
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

@app.route('/banco-dados')
def get_banco_dados():
    try:
        dados = buscar_dados_banco()
        return jsonify(dados)
    except Exception as e:
        return jsonify({"erro": f"Erro ao buscar dados: {str(e)}"}), 500

@app.route('/zerar-banco', methods=['POST'])
def zerar_banco():
    try:
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        
        c.execute('DELETE FROM MovimentoContas_has_Classificacao')
        c.execute('DELETE FROM ParcelasContas')
        c.execute('DELETE FROM MovimentoContas')
        c.execute('DELETE FROM Classificacao')
        c.execute('DELETE FROM Pessoas')
        c.execute('DELETE FROM Embeddings')
        
        c.execute('DELETE FROM sqlite_sequence WHERE name="Pessoas"')
        c.execute('DELETE FROM sqlite_sequence WHERE name="Classificacao"')
        c.execute('DELETE FROM sqlite_sequence WHERE name="MovimentoContas"')
        c.execute('DELETE FROM sqlite_sequence WHERE name="ParcelasContas"')
        c.execute('DELETE FROM sqlite_sequence WHERE name="Embeddings"')
        
        conn.commit()
        conn.close()
        
        return jsonify({"sucesso": True, "mensagem": "Banco de dados zerado com sucesso!"})
    except Exception as e:
        return jsonify({"erro": f"Erro ao zerar banco: {str(e)}"}), 500

# ==================== ROTAS RAG ====================

@app.route('/rag-simples', methods=['POST'])
def consulta_rag_simples():
    try:
        dados = request.json
        pergunta = dados.get('pergunta', '')
        
        if not pergunta:
            return jsonify({"erro": "Pergunta não fornecida"}), 400
        
        resultado = rag_simples(pergunta)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"erro": f"Erro ao processar RAG Simples: {str(e)}"}), 500

@app.route('/rag-embeddings', methods=['POST'])
def consulta_rag_embeddings():
    try:
        dados = request.json
        pergunta = dados.get('pergunta', '')
        
        if not pergunta:
            return jsonify({"erro": "Pergunta não fornecida"}), 400
        
        resultado = rag_embeddings(pergunta)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"erro": f"Erro ao processar RAG Embeddings: {str(e)}"}), 500

@app.route('/reconstruir-embeddings', methods=['POST'])
def reconstruir_embeddings():
    """Reconstrói todos os embeddings do banco de dados"""
    try:
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        
        # Limpar embeddings existentes
        c.execute('DELETE FROM Embeddings')
        
        # Criar embeddings para pessoas
        c.execute('SELECT * FROM Pessoas')
        pessoas = c.fetchall()
        for pessoa in pessoas:
            criar_embedding_pessoa(pessoa[0], pessoa[1], pessoa[2], pessoa[4])
        
        # Criar embeddings para classificações
        c.execute('SELECT * FROM Classificacao')
        classificacoes = c.fetchall()
        for classif in classificacoes:
            criar_embedding_classificacao(classif[0], classif[2])
        
        # Criar embeddings para movimentos
        c.execute('SELECT * FROM MovimentoContas')
        movimentos = c.fetchall()
        for mov in movimentos:
            criar_embedding_movimento(mov[0], mov[2], mov[3], mov[4], mov[6])
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "sucesso": True,
            "mensagem": f"Embeddings reconstruídos: {len(pessoas)} pessoas, {len(classificacoes)} classificações, {len(movimentos)} movimentos"
        })
    except Exception as e:
        return jsonify({"erro": f"Erro ao reconstruir embeddings: {str(e)}"}), 500
@app.route('/api/pessoas', methods=['GET'])
def get_pessoas():
    try:
        conn = sqlite3.connect('sistema.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        tipo = request.args.get('tipo')
        status = request.args.get('status', 'ATIVO')
        search = request.args.get('search', '')
        
        query = 'SELECT * FROM Pessoas WHERE status = ?'
        params = [status]
        
        if tipo:
            query += ' AND tipo = ?'
            params.append(tipo)
        
        if search:
            query += ' AND (razaosocial LIKE ? OR documento LIKE ?)'
            params.extend([f'%{search}%', f'%{search}%'])
        
        c.execute(query, params)
        pessoas = [dict(row) for row in c.fetchall()]
        conn.close()
        
        return jsonify(pessoas)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/pessoas/<int:id>', methods=['GET'])
def get_pessoa(id):
    try:
        conn = sqlite3.connect('sistema.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM Pessoas WHERE idPessoas = ?', (id,))
        pessoa = c.fetchone()
        conn.close()
        
        if pessoa:
            return jsonify(dict(pessoa))
        return jsonify({"erro": "Pessoa não encontrada"}), 404
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/pessoas', methods=['POST'])
def create_pessoa():
    try:
        dados = request.json
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO Pessoas (tipo, razaosocial, fantasia, documento, status)
                     VALUES (?, ?, ?, ?, ?)''',
                  (dados['tipo'], dados['razaosocial'], dados.get('fantasia', ''),
                   dados['documento'], 'ATIVO'))
        
        novo_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Criar embedding
        criar_embedding_pessoa(novo_id, dados['tipo'], dados['razaosocial'], dados['documento'])
        
        return jsonify({"sucesso": True, "id": novo_id, "mensagem": "Pessoa criada com sucesso"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/pessoas/<int:id>', methods=['PUT'])
def update_pessoa(id):
    try:
        dados = request.json
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        
        c.execute('''UPDATE Pessoas 
                     SET razaosocial = ?, fantasia = ?, documento = ?
                     WHERE idPessoas = ?''',
                  (dados['razaosocial'], dados.get('fantasia', ''),
                   dados['documento'], id))
        
        conn.commit()
        conn.close()
        
        return jsonify({"sucesso": True, "mensagem": "Pessoa atualizada com sucesso"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/pessoas/<int:id>', methods=['DELETE'])
def delete_pessoa(id):
    try:
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        c.execute('UPDATE Pessoas SET status = ? WHERE idPessoas = ?', ('INATIVO', id))
        conn.commit()
        conn.close()
        
        return jsonify({"sucesso": True, "mensagem": "Pessoa excluída com sucesso"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# CRUD CLASSIFICAÇÃO
@app.route('/api/classificacoes', methods=['GET'])
def get_classificacoes():
    try:
        conn = sqlite3.connect('sistema.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        tipo = request.args.get('tipo')
        status = request.args.get('status', 'ATIVO')
        search = request.args.get('search', '')
        
        query = 'SELECT * FROM Classificacao WHERE status = ?'
        params = [status]
        
        if tipo:
            query += ' AND tipo = ?'
            params.append(tipo)
        
        if search:
            query += ' AND descricao LIKE ?'
            params.append(f'%{search}%')
        
        c.execute(query, params)
        classificacoes = [dict(row) for row in c.fetchall()]
        conn.close()
        
        return jsonify(classificacoes)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/classificacoes', methods=['POST'])
def create_classificacao():
    try:
        dados = request.json
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO Classificacao (tipo, descricao, status)
                     VALUES (?, ?, ?)''',
                  (dados['tipo'], dados['descricao'], 'ATIVO'))
        
        novo_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Criar embedding
        criar_embedding_classificacao(novo_id, dados['descricao'])
        
        return jsonify({"sucesso": True, "id": novo_id, "mensagem": "Classificação criada com sucesso"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/classificacoes/<int:id>', methods=['PUT'])
def update_classificacao(id):
    try:
        dados = request.json
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        
        c.execute('UPDATE Classificacao SET descricao = ? WHERE idClassificacao = ?',
                  (dados['descricao'], id))
        
        conn.commit()
        conn.close()
        
        return jsonify({"sucesso": True, "mensagem": "Classificação atualizada com sucesso"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/classificacoes/<int:id>', methods=['DELETE'])
def delete_classificacao(id):
    try:
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        c.execute('UPDATE Classificacao SET status = ? WHERE idClassificacao = ?', ('INATIVO', id))
        conn.commit()
        conn.close()
        
        return jsonify({"sucesso": True, "mensagem": "Classificação excluída com sucesso"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# CRUD MOVIMENTOS
@app.route('/api/movimentos', methods=['GET'])
def get_movimentos():
    try:
        conn = sqlite3.connect('sistema.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        status = request.args.get('status', 'ATIVO')
        search = request.args.get('search', '')
        
        query = '''SELECT m.*, 
                   f.razaosocial as fornecedor_nome,
                   ft.razaosocial as faturado_nome
                   FROM MovimentoContas m
                   LEFT JOIN Pessoas f ON m.Pessoas_idFornecedorCliente = f.idPessoas
                   LEFT JOIN Pessoas ft ON m.Pessoas_idFaturado = ft.idPessoas
                   WHERE m.status = ?'''
        params = [status]
        
        if search:
            query += ' AND (m.numeronotafiscal LIKE ? OR m.descricao LIKE ?)'
            params.extend([f'%{search}%', f'%{search}%'])
        
        c.execute(query, params)
        movimentos = [dict(row) for row in c.fetchall()]
        conn.close()
        
        return jsonify(movimentos)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/movimentos', methods=['POST'])
def create_movimento():
    try:
        dados = request.json
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO MovimentoContas 
                     (tipo, numeronotafiscal, dataemissao, descricao, valortotal, 
                      Pessoas_idFornecedorCliente, Pessoas_idFaturado, status)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (dados['tipo'], dados['numeronotafiscal'], dados['dataemissao'],
                   dados['descricao'], dados['valortotal'], 
                   dados.get('Pessoas_idFornecedorCliente'), 
                   dados.get('Pessoas_idFaturado'), 'ATIVO'))
        
        novo_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Criar embedding
        criar_embedding_movimento(novo_id, dados['numeronotafiscal'], 
                                dados['dataemissao'], dados['descricao'], 
                                dados['valortotal'])
        
        return jsonify({"sucesso": True, "id": novo_id, "mensagem": "Movimento criado com sucesso"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/movimentos/<int:id>', methods=['PUT'])
def update_movimento(id):
    try:
        dados = request.json
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        
        c.execute('''UPDATE MovimentoContas 
                     SET numeronotafiscal = ?, dataemissao = ?, descricao = ?, 
                         valortotal = ?
                     WHERE idMovimentoContas = ?''',
                  (dados['numeronotafiscal'], dados['dataemissao'],
                   dados['descricao'], dados['valortotal'], id))
        
        conn.commit()
        conn.close()
        
        return jsonify({"sucesso": True, "mensagem": "Movimento atualizado com sucesso"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route('/api/movimentos/<int:id>', methods=['DELETE'])
def delete_movimento(id):
    try:
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        c.execute('UPDATE MovimentoContas SET status = ? WHERE idMovimentoContas = ?', 
                  ('INATIVO', id))
        conn.commit()
        conn.close()
        
        return jsonify({"sucesso": True, "mensagem": "Movimento excluído com sucesso"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# SCRIPT DE GERAÇÃO DE DADOS DE TESTE
@app.route('/api/gerar-dados-teste', methods=['POST'])
def gerar_dados_teste():
    try:
        conn = sqlite3.connect('sistema.db')
        c = conn.cursor()
        
        # Gerar 200 pessoas
        tipos_pessoa = ['FORNECEDOR', 'CLIENTE', 'FATURADO']
        nomes = ['João Silva', 'Maria Santos', 'Pedro Oliveira', 'Ana Costa', 'Carlos Souza']
        empresas = ['Agro Tech Ltda', 'Rural Solutions', 'Campo Verde SA', 'Sementes Brasil']
        
        for i in range(200):
            tipo = tipos_pessoa[i % 3]
            is_empresa = i % 2 == 0
            razao = empresas[i % len(empresas)] if is_empresa else nomes[i % len(nomes)]
            doc = f"{10000000000000 + i}" if is_empresa else f"{10000000000 + i}"
            
            c.execute('''INSERT INTO Pessoas (tipo, razaosocial, fantasia, documento, status)
                         VALUES (?, ?, ?, ?, ?)''',
                      (tipo, f"{razao} {i+1}", f"Fantasia {i+1}", doc, 'ATIVO'))
        
        # Gerar 200 classificações
        categorias = ['INSUMOS AGRÍCOLAS', 'MANUTENÇÃO', 'RECURSOS HUMANOS', 
                      'SERVIÇOS', 'INFRAESTRUTURA', 'VENDAS', 'EXPORTAÇÃO']
        
        for i in range(100):
            c.execute('''INSERT INTO Classificacao (tipo, descricao, status)
                         VALUES (?, ?, ?)''',
                      ('DESPESA', f"{categorias[i % len(categorias)]} - Item {i+1}", 'ATIVO'))
        
        for i in range(100):
            c.execute('''INSERT INTO Classificacao (tipo, descricao, status)
                         VALUES (?, ?, ?)''',
                      ('RECEITA', f"{categorias[i % len(categorias)]} - Item {i+100}", 'ATIVO'))
        
        # Gerar 200 movimentos
        import random
        from datetime import datetime, timedelta
        
        for i in range(200):
            data = datetime.now() - timedelta(days=random.randint(1, 365))
            tipo = 'APAGAR' if i % 2 == 0 else 'ARECEBER'
            valor = round(random.uniform(100, 50000), 2)
            
            c.execute('''INSERT INTO MovimentoContas 
                         (tipo, numeronotafiscal, dataemissao, descricao, valortotal, 
                          Pessoas_idFornecedorCliente, Pessoas_idFaturado, status)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (tipo, f"NF{10000+i}", data.strftime('%Y-%m-%d'),
                       f"Descrição movimento {i+1}", valor, 
                       random.randint(1, 200), random.randint(1, 200), 'ATIVO'))
        
        conn.commit()
        conn.close()
        
        return jsonify({"sucesso": True, "mensagem": "600 registros de teste criados com sucesso"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    
        return render_template('crud.html')

if __name__ == '__main__':
    app.run()