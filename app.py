from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import PyPDF2
import json
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configurar Gemini AI
GEMINI_API_KEY = "AIzaSyAvag5ZD7lFydA4NVcM6a6AsMjUaSfmk7A"  # Substitua pela sua API key
genai.configure(api_key=GEMINI_API_KEY)

# Categorias de despesas
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

def extrair_texto_pdf(arquivo_pdf):
    """Extrai texto do arquivo PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(arquivo_pdf)
        texto_completo = ""
        
        for pagina in pdf_reader.pages:
            texto_completo += pagina.extract_text() + "\n"
        
        return texto_completo
    except Exception as e:
        return f"Erro ao extrair texto do PDF: {str(e)}"

def classificar_despesa(descricao_produtos):
    """Classifica a despesa baseada na descrição dos produtos"""
    descricao_lower = descricao_produtos.lower()
    
    for categoria, palavras_chave in CATEGORIAS_DESPESAS.items():
        if any(palavra in descricao_lower for palavra in palavras_chave):
            return categoria
    
    return "OUTRAS DESPESAS"

def processar_nota_fiscal_gemini(texto_pdf):
    """Processa a nota fiscal usando Gemini AI"""
    try:
        # Usar o modelo mais recente disponível
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Analise o seguinte texto de uma nota fiscal e extraia as informações em formato JSON:

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

        Se alguma informação não estiver disponível, use null. Para datas, use sempre o formato YYYY-MM-DD.
        """
        
        response = model.generate_content(prompt)
        
        # Extrair apenas o JSON da resposta
        resposta_texto = response.text.strip()
        
        # Remover possíveis marcações de código
        if resposta_texto.startswith('```json'):
            resposta_texto = resposta_texto.replace('```json', '').replace('```', '').strip()
        elif resposta_texto.startswith('```'):
            resposta_texto = resposta_texto.replace('```', '').strip()
        
        # Tentar parsear o JSON
        dados_extraidos = json.loads(resposta_texto)
        
        # Classificar a despesa baseada nos itens
        descricao_produtos = " ".join([item.get("descricao", "") for item in dados_extraidos.get("itens", [])])
        categoria_automatica = classificar_despesa(descricao_produtos)
        
        # Adicionar classificação automática se não houver
        if not dados_extraidos.get("classificacaoDespesa") or dados_extraidos["classificacaoDespesa"] == [None]:
            dados_extraidos["classificacaoDespesa"] = [categoria_automatica]
        
        return dados_extraidos
        
    except json.JSONDecodeError as e:
        return {"erro": f"Erro ao processar JSON: {str(e)}", "resposta_bruta": resposta_texto}
    except Exception as e:
        return {"erro": f"Erro ao processar com Gemini: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({"erro": "Nenhum arquivo PDF foi enviado"}), 400
        
        arquivo = request.files['pdf']
        
        if arquivo.filename == '':
            return jsonify({"erro": "Nenhum arquivo selecionado"}), 400
        
        if not arquivo.filename.lower().endswith('.pdf'):
            return jsonify({"erro": "Arquivo deve ser um PDF"}), 400
        
        # Extrair texto do PDF
        texto_pdf = extrair_texto_pdf(BytesIO(arquivo.read()))
        
        if texto_pdf.startswith("Erro"):
            return jsonify({"erro": texto_pdf}), 400
        
        # Processar com Gemini
        dados_extraidos = processar_nota_fiscal_gemini(texto_pdf)
        
        return jsonify(dados_extraidos)
        
    except Exception as e:
        return jsonify({"erro": f"Erro interno do servidor: {str(e)}"}), 500

@app.route('/categorias')
def get_categorias():
    return jsonify(list(CATEGORIAS_DESPESAS.keys()))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)