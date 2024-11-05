import pandas as pd
import joblib

from flask import Flask, request, jsonify

app = Flask(__name__)

def create_features(df):
    # Calculando taxas de crescimento
    df['cliques_growth'] = (df['Cliques_30d'] - df['Cliques_7d']) / df['Cliques_7d']
    df['aberturas_growth'] = (df['Aberturas_30d'] - df['Aberturas_7d']) / df['Aberturas_7d']
    df['leads_growth'] = (df['Leads_30d'] - df['Leads_7d']) / df['Leads_7d']
    
    # Calculando taxas de conversão
    df['conv_rate_7d'] = df['Leads_7d'] / df['Envios_7d']
    df['conv_rate_30d'] = df['Leads_30d'] / df['Envios_30d']
    
    # Calculando engajamento
    df['engagement_7d'] = (df['Cliques_7d'] + df['Aberturas_7d']) / df['Envios_7d']
    df['engagement_30d'] = (df['Cliques_30d'] + df['Aberturas_30d']) / df['Envios_30d']
    
    return df

feature_columns = [
    'cliques_growth', 'aberturas_growth', 'leads_growth',
    'conv_rate_7d', 'conv_rate_30d', 'engagement_7d', 'engagement_30d'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recebendo dados
        data = request.get_json()
        
        # Criando DataFrame
        input_df = pd.DataFrame(data, index=[0])
        
        # Criando features
        input_df = create_features(input_df)
        
        # Selecionando features relevantes
        input_features = input_df[feature_columns]
        
        # Scaling
        input_scaled = scaler.transform(input_features)
        
        # Fazendo predição
        prediction = rf.predict(input_scaled)[0]
        probability = rf.predict_proba(input_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability[1]),
            'message': 'Campanha apresentará melhoria' if prediction == 1 else 'Campanha não apresentará melhoria'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Carregando modelo e scaler salvos
    with open('campaign_model.joblib', 'rb') as f:
        rf = joblib.load(f)
    with open('scaler.joblib', 'rb') as f:
        scaler = joblib.load(f)
    app.run(debug=True)