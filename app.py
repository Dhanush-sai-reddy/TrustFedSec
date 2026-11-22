import hashlib, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from config import Config
from src.dataset import prepare
from src.coordinator import run

st.set_page_config(page_title='TrustFed-Sec', page_icon='🛡️', layout='wide')
st.markdown('''<style>.stApp {background:#071321;color:#d7e8ee} [data-testid="stSidebar"]{background:#0d2030} h1,h2,h3{color:#ecfbff} [data-testid="stMetric"]{background:#102637;padding:15px;border-radius:10px;border:1px solid #1c5261} .stButton button{border:1px solid #25b7b0;color:#8ff5ec} </style>''', unsafe_allow_html=True)
st.title('🛡 TrustFed-Sec')
st.caption('Trust-aware, poisoning-resistant federated intrusion detection · CPU simulation')
st.info('Five simulated clients by default train local logistic models. A coordinator inspects their updates against a trusted validation set and softly adjusts their aggregation weights. This is an attack simulation, not live network monitoring.')
with st.sidebar:
    st.header('Experiment controls')
    dataset = st.radio('Dataset', ['Synthetic demonstration', 'Local UNSW-NB15 CSV'])
    files = st.file_uploader('Upload one or more UNSW-NB15 CSV files', type=['csv'], accept_multiple_files=True) if dataset.startswith('Local') else []
    partition = st.selectbox('Client partition', ['IID', 'Non-IID'])
    n = st.slider('Clients', 2, 10, 5)
    bad = st.slider('Compromised clients', 0, n, min(2,n))
    attack = st.selectbox('Attack type', ['sign flip', 'label flip'])
    strength = st.slider('Sign-flip multiplier', 0.0, 15.0, 5.0, 0.5)
    fraction = st.slider('Label poisoning fraction', 0.0, 1.0, 0.5, 0.05)
    rounds = st.slider('Communication rounds', 1, 30, 10)
    seed = st.number_input('Random seed', min_value=0, max_value=1000000, value=42)
    with st.expander('Trust parameters'):
        beta = st.slider('History smoothing β', 0.0, 0.95, 0.6, 0.05)
        norm = st.slider('Log norm deviation threshold', 0.2, 3.0, 1.0, 0.1)
        cosine = st.slider('Cosine threshold', -0.5, 0.9, 0.2, 0.1)
        val = st.slider('Validation loss threshold', 0.005, 0.2, 0.02, 0.005, format='%.3f')
        floor = st.slider('Minimum trust weight', 0.01, 0.5, 0.1, 0.01)
    if st.button('Reset results', use_container_width=True):
        st.session_state.pop('results', None); st.rerun()
cfg = Config(seed=int(seed),clients=n,compromised=bad,attack=attack,strength=strength,poison_fraction=fraction,rounds=rounds,partition=partition,beta=beta,norm_threshold=norm,cosine_threshold=cosine,validation_threshold=val,trust_floor=floor)
file_data = [(f.name, f.getvalue()) for f in files]
key = hashlib.sha256((repr(cfg)+dataset).encode()+b''.join(name.encode()+content for name,content in file_data)).hexdigest()
if 'results' not in st.session_state: st.session_state.results = {}
current = st.session_state.results.get(key, {})
if st.session_state.results and not current: st.warning('Settings or dataset changed. Previous results are hidden to prevent comparisons across configurations.')
st.subheader('Run scenarios')
st.caption('All runs with the same controls use the same data split, partitions, initial model and seeds. Poisoned and trust-aware runs use the same compromised clients and attack.')
buttons = st.columns(3)
selected = None
for col, name in zip(buttons, ['Normal FedAvg','Poisoned FedAvg','TrustFed-Sec']):
    if col.button('Run '+name, use_container_width=True): selected=name
if selected:
    if dataset.startswith('Local') and not files: st.error('Upload at least one CSV containing the binary label column.')
    else:
        try:
            with st.spinner('Preparing data and running rounds…'):
                data=prepare(cfg, files if files else None)
                progress=st.progress(0, text='Starting')
                live=st.empty()
                def update(r, rows, metrics):
                    progress.progress(r/cfg.rounds, text=f'Round {r}/{cfg.rounds} · validation loss {metrics["loss"]:.4f} · accuracy {metrics["accuracy"]:.3f} · macro F1 {metrics["macro_f1"]:.3f}')
                    live.dataframe(pd.DataFrame(rows)[['client','status','trust','aggregation_weight','update_norm','cosine','validation_loss_delta']], hide_index=True, use_container_width=True)
                result=run(data,cfg,selected,update)
                current[selected]=result; st.session_state.results[key]=current
                out=Path('results'); out.mkdir(exist_ok=True)
                (out / f'{datetime.now():%Y%m%d_%H%M%S_%f}_{selected.replace(" ","_")}.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
                st.success(f'{selected} completed and saved to results/.')
        except Exception as e: st.error(f'Experiment could not run: {e}')
if current:
    first=next(iter(current.values())); cols=st.columns(4)
    for c, label, value in zip(cols,['Dataset','Samples','Features after encoding','Clients'],[first['source'],f"{first['samples']:,}",first['features'],n]): c.metric(label,value)
    st.subheader('Held-out test results')
    table=[]
    for name,result in current.items(): table.append({'Scenario':name, **result['test'], 'seconds':result['seconds'], 'communication_bytes':result['communication_bytes']})
    st.dataframe(pd.DataFrame(table).set_index('Scenario').round(4), use_container_width=True)
    st.caption('Attack recall is recall for label 1; communication is estimated float64 parameter bytes sent in both directions, excluding protocol overhead. Timing is measured on this machine.')
    hist=pd.concat([pd.DataFrame(v['history']).assign(Scenario=k) for k,v in current.items()],ignore_index=True)
    a,b=st.columns(2)
    for col, metric in [(a,'accuracy'),(b,'macro_f1')]:
        fig=px.line(hist,x='round',y=metric,color='Scenario',markers=True,template='plotly_dark',title='Validation '+metric.replace('_',' ')); col.plotly_chart(fig,use_container_width=True)
    fig=px.line(hist,x='round',y='loss',color='Scenario',markers=True,template='plotly_dark',title='Validation loss'); st.plotly_chart(fig,use_container_width=True)
    st.subheader('Client inspection')
    name=st.selectbox('Scenario',list(current)); result=current[name]
    round_no=st.select_slider('Round',options=list(range(1,cfg.rounds+1)),value=cfg.rounds)
    rows=pd.DataFrame(result['clients']); rows=rows[rows['round']==round_no]
    st.dataframe(rows.round(4),hide_index=True,use_container_width=True)
    client=st.selectbox('Explain client',rows['client'].tolist()); row=rows[rows.client==client].iloc[0]
    st.write(f"Client {client} has trust **{row['trust']:.3f}** and aggregation weight **{row['aggregation_weight']:.1%}**. Its update norm is {row['update_norm']:.3f}, cosine similarity to the median update is {row['cosine']:.3f}, and its candidate validation loss changed by {row['validation_loss_delta']:+.4f}. Status: **{row['status']}**. Trust combines these signals with previous rounds; a review flag is not proof of malicious activity.")
    if name != 'TrustFed-Sec': st.caption('Trust values are diagnostic in this FedAvg scenario; aggregation uses sample counts only.')
    st.download_button('Download selected result JSON',json.dumps(result,indent=2),file_name='trustfed_result.json',mime='application/json')
else: st.caption('Run a scenario to display measured results. No results are preloaded.')
