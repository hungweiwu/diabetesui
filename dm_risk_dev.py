import streamlit as st
import pandas as pd
import joblib
import warnings
import plotly.express as px
import plotly.graph_objects as go
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
import os


warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", 
                   page_icon="📊",
                   initial_sidebar_state="expanded",    
                   menu_items={
                       'Get Help': 'https://www.extremelycoolapp.com/help',
                       'Report a bug': "https://www.extremelycoolapp.com/bug",
                       'About': "# This is a header. This is an *extremely* cool app!"
                   })
st.title("aDCSI Score Calculator")

category_colors = {
    "Ocular": "rgba(255,0,0,0.5)",
    "Renal": "rgba(138,92,7,1)",
    "Neurological": "rgba(255,255,0,0.5)",
    "Cerebrovascular": "rgba(9,9,139,1)",
    "Cardiovascular": "rgba(3,88,3,1)",
    "Peripheral Vascular": "rgba(88,87,87,1)",
    "Metabolic": "rgba(87,3,87,1)"
}


def display_icd_markdown(details):
    badges = []
    for category, icd_set in details.items():
        if icd_set:  # 有 ICD 才顯示
            color = category_colors.get(category, "#808080")  # default gray
            for icd_point in icd_set:
                code, point = icd_point.split(":")
                # point_color = "green" if point == '1' else "violet"
                badge_html = f"""
                <span style="
                    display:inline-block;
                    background-color:{color};
                    color:white;
                    border-radius:5px;
                    padding:3px 6px;
                    margin:2px;
                    font-size:12px;
                ">{category}</span>
                <span style="
                    display:inline-block;
                    background-color:#808080;
                    color:white;
                    border-radius:5px;
                    padding:3px 6px;
                    margin:2px;
                    font-size:12px;
                ">{code}</span>
                """
                badges.append(badge_html)
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    # st.markdown("<div style='white-space:nowrap'>" + "".join(badges) + "</div>", unsafe_allow_html=True)

def display_icd_markdown(details):
    badges = []
    for category, icd_set in details.items():
        if icd_set:
            color = category_colors.get(category, "#808080")  # default gray
            for icd_point in icd_set:
                code, point = icd_point.split(":")
                badge_html = (
                    # f'<span style="display:inline-block;background-color:{color};color:white;'
                    # f'border-radius:5px;padding:3px 6px;margin:2px;font-size:12px;">{category}</span>'
                    f'<span style="display:inline-block;background-color:{color};color:white;'
                    f'border-radius:5px;padding:3px 6px;margin:2px;font-size:12px;">{code}</span>'
                )
                badges.append(badge_html)
    st.markdown(" ".join(badges), unsafe_allow_html=True)

def calc_aDCSI_scores(ICD_list):
    Label_1 = {"E08.3": 1, "E09.3": 1, "E10.3": 1, "E11.3": 1, "E12.3": 1, "E13.3": 1, "E14.3": 1,
               "H28.0": 1, "H33.0": 1, "H35.0": 1, "H36.0": 1, "H35.2": 1, "H35.3": 1, "H35.6": 1, "H35.8": 1, 
               "H43.1": 2, "H54.0": 2, "H54.4": 2,
               "250.5": 1, "362.01": 1, "362.1": 1, "362.83": 1, "362.53": 1, "362.81": 1, "362.82": 1,
               "362.02": 2, "361": 2, "369": 2, "379.23": 2 }
    Label_2 = {"E08.2": 1, "E09.2": 1,"E10.2": 1, "E11.2": 1, "E12.2": 1, "E13.2": 1, "E14.2": 1, 
               "N03": 1, "N04": 1, "N05": 1, "N08.3": 1, "N17": 2, 
               "N18.0": 2, "N18.1": 1, "N18.2": 1, "N18.3": 1, "N18.4": 2, "N18.5": 2, "N18.6": 2, "N18.7": 2, "N18.8": 2, "N18.9": 1, "N19": 2, 
               "Z49": 2, "Z99.2": 2, "T82.4": 2,
               "250.4": 1, "580": 1, "581": 1, "581.81": 1, "582": 1, "583": 1,
               "585": 2, "586": 2, "593.9": 2 }  
    Label_3 = {"E08.4": 1, "E09.4": 1, "E10.4": 1, "E11.4": 1, "E12.4": 1, "E13.4": 1, "E14.4": 1,
               "G59.0": 1, "G60.9": 1, "G63.2": 1, "G73.0": 1, "G99.0": 1, 
               "H49.0": 1, "H49.1": 1, "H49.2": 1, "M14.2": 1, "M14.6": 1,
               "356.9": 1, "250.6": 1, "358.1": 1, "951.0": 1, "951.1": 1, "951.3": 1,
               "354": 1, "355": 1, "713.5": 1, "357.2": 1 }   
    Label_4 = {"G45": 1, "I61": 1, "I63": 1, "I64": 1, "I69": 1,
               "435": 1, "431": 2, "433": 2, "434": 2, "436": 2 } 
    Label_5 = {"I11.0": 2, "I13.0": 2, "I20": 1, "I21": 2, "I22": 2, "I23": 2, "I24": 1, "I25": 1, 
               "I46": 2, "I48": 2, "I49.0": 2, "I50": 2, "I70": 1, "I71": 1,
               "440": 1, "411": 1, "413": 1, "414": 1, "429.2": 1, 
               "410": 2, "427.1": 2, "427.3": 2, "427.4": 2, "427.5": 2, "412": 2, "428": 2, "440.23": 2, "440.24": 2, "441": 2 }
    Label_6 = {"E08.5": 1, "E09.5": 1,  "E10.5": 1, "E11.5": 1, "E12.5": 1, "E13.5": 1, "E14.5": 1,
               "E08.52": 2, "E09.52": 2, "E10.52": 2, "E11.52": 2, "E13.52": 2, 
               "E08.621": 2, "E09.621": 2, "E10.621": 2, "E11.621": 2, "E13.621": 2,
               "E08.622": 2, "E09.622": 2, "E10.622": 2, "E11.622": 2, "E13.622": 2, 
               "I70.23": 2, "I70.24": 2, "I70.25": 2, "I70.26": 2, "I72.4": 1, "I73.9": 1, "I74.3": 2, "I79.2": 1, 
               "L97": 2, "L98.4": 2, "A48.0": 2, "R02": 2,
               "250.7": 1, "442.3": 1, "443.81": 1, "443.9": 1, "892.1": 1,
               "444.22": 2, "785.4": 2, "0.4": 2, "707.1": 2 }
    Label_7 = {"E10.0": 2, "E11.0": 2, "E12.0": 2, "E13.0": 2, "E14.0": 2, 
               "E08.1": 2, "E09.1": 2, "E10.1": 2, "E11.1": 2, "E12.1": 2, "E13.1": 2, "E14.1": 2, 
               "E08.641": 2, "E09.641": 2, "E10.641": 2, "E11.641": 2, "E13.641": 2,
               "250.1": 2, "250.2": 2, "250.3": 2 }                 

    aDCSI = {
        "Ocular": (Label_1, 2),
        "Renal": (Label_2, 2),
        "Neurological": (Label_3, 1), 
        "Cerebrovascular": (Label_4, 2),
        "Cardiovascular": (Label_5, 2),
        "Peripheral Vascular": (Label_6, 2),
        "Metabolic": (Label_7, 2),
    }

    scores, details = {}, {}
    total_score = 0

    for category, (label_dict, max_score) in aDCSI.items():
        score = 0
        cat_detail = []
        for code in ICD_list:
            for icd_key, point in label_dict.items():
                if icd_key in code:
                    score += point
                    cat_detail.append(f"{code}:{point}")

        score = min(score, max_score)
        scores[category] = score
        total_score += score
        details[category] = set(cat_detail)

    scores["Total"] = total_score
    return scores, details


def display_patient_info(patient_df):
    if patient_df.empty:
        st.warning("No patient data found.")
        return
    patient_info = patient_df.iloc[0]
    st.subheader("Patient Information")
    st.markdown(f"""
    <div style='font-size:19px;'>
        <ul>
            <li><b>PID:</b> {patient_info['PID']}</li>
            <li><b>Age:</b> {patient_info['Age']}</li>
            <li><b>Sex:</b> {patient_info['Sex']}</li>
            <li><b>Race:</b> {patient_info['Race']}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



def plot_trends(results_df):
    fig = go.Figure()
    fig.update_layout(width=300, height=250)
    fig.add_trace(go.Scatter(x=results_df["year"], y=results_df["score"],
                             name="aDCSI Score", mode="lines+markers", line=dict(color="rgba(40, 75, 150, 0.8)"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=results_df["year"], y=results_df["mortality"],
                             name="Mortality (%)", mode="lines+markers", line=dict(color="rgba(150, 5, 50, 0.8)"), yaxis="y2"))
    fig.update_layout(
        xaxis=dict(title="Time"),
        yaxis=dict(title="aDCSI Score", side="left", range=[0, max(results_df["score"].max() + 1, 5)]),
        yaxis2=dict(title="Mortality (%)", side="right", overlaying="y", range=[0, 100]),
        legend=dict(x=0.1, y=1.1, orientation="h"),
        margin=dict(t=50, b=50)
    )
    return fig

def display_severity(severity):
    """
    根據 severity 顯示不同顏色和 icon
    """
    severity_map = {
        "Low": {"color": "rgba(0, 150, 0, 0.7)", "icon": "✅", "recommendation": "Continue regular follow-up and lifestyle management."},
        "Medium": {"color": "rgba(220, 90, 0, 0.7)", "icon": "⚠️", "recommendation": "Consider intensifying monitoring and reviewing medications."},
        "High": {"color": "rgba(220, 0, 0, 0.7)", "icon": "🚨", "recommendation": "Immediate intervention required."}
    }

    info = severity_map.get(severity, {"color": "gray", "icon": "❔", "recommendation": "No recommendation available."})

    st.markdown(f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 40px;
            padding: 50px;
            text-align: center;
            background-color: #f9f9f9;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
        ">
            <div style="font-size:30px; font-weight:bold; color:{info['color']}">
                {info['icon']} {severity} risk
            </div>
            <div style="font-size:18px; margin-top:5px; color:#333;">
                {info['recommendation']}
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* 全域字體大小 */
    html, body, [class*="css"]  {
        font-size: 18px;  /* 改大整體字體 */
    }

    /* metric 的字體大小 */
    .stMetric {
        font-size: 18px !important;
    }

    """,
    unsafe_allow_html=True
)


# uploaded_file = st.sidebar.file_uploader("Please upload CSV file", type=["csv"]) # 資料不須另外上傳
uploaded_file = "dm_risk_data.csv"

if uploaded_file:
    
    df = pd.read_csv(uploaded_file)
    if not {"PID", "condition_source_value", "condition_start_date"}.issubset(df.columns):
        st.error("CSV must have columns: PID, condition_source_value, condition_start_date")
    else:

        pid_list = [""] + df["PID"].unique().tolist()
        def clear_chat_history():
            st.session_state.messages = []

        selected_pid = st.selectbox(
            "Select Patient's PID", 
            pid_list,
            on_change=clear_chat_history,
            key="selected_pid"
        )

        if selected_pid:
          
            patient_df = df[df["PID"] == selected_pid].copy()
            patient_df["condition_start_date"] = pd.to_datetime(patient_df["condition_start_date"])
            patient_df["time_group"] = patient_df["condition_start_date"].dt.to_period("Y").astype(str)
            patient_df['Age'] = (patient_df['time_group'].astype(int) - patient_df['year_of_birth']).max()
            patient_df['Sex'] = patient_df['gender_source_value'].map({'M': 'Male', 'F': 'Female'})
            patient_df['Race'] = patient_df['race_concept_name']

            maincol1, maincol2= st.columns([1, 0.6], border=False)
            with maincol1:
                col1, col2= st.columns([1, 1.5])
                with col1:
                    display_patient_info(patient_df)

        
                results, breakdowns, seen_icds = [], {}, set()
                clf = joblib.load('clf.pkl')
                for t, group in patient_df.groupby("time_group"):

                    seen_icds.update(group["condition_source_value"].tolist())
                    scores, details = calc_aDCSI_scores(list(seen_icds))
    
                    total = scores["Total"]
                    level = "Low" if total <= 2 else "Medium" if total <= 4 else "High"
                    prob = clf.predict_proba([[total]]).squeeze()[-1]
                    percent = round(prob*100, 2)
                    results.append({"year": t, "score": total, "severity": level, "mortality": percent})
                    breakdowns[t] = (scores, details)

                results_df = pd.DataFrame(results).sort_values(by="year", ascending=False, ignore_index=True)

                with col2:
                    st.subheader("Dynamic aDCSI Scores")
                    fig = plot_trends(results_df)
                    tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
                    tab1.plotly_chart(fig, use_container_width=True)
                    tab2.dataframe(results_df, height=250)

        
                st.subheader("Patient Summary")

        
                selected_time = st.selectbox("Select Time for Breakdown", results_df["year"].tolist())

                if selected_time in breakdowns:
                    scores, details = breakdowns[selected_time]


                    df_scores = pd.DataFrame(
                        [(k, v) for k, v in scores.items() if k != "Total"],
                        columns=["Category", "Score"]
                    )
                    df_scores_no_total = df_scores.copy()


                    subcol1, subcol2= st.columns(2)
                    with subcol1:

                        fig = px.bar(
                            df_scores,
                            x="Score",
                            y="Category",
                            orientation="h",
                            text="Score",
                            color="Category", 
                            color_discrete_map=category_colors
                        )

                        fig.update_layout(
                            title="aDCSI Breakdown by Category",
                            xaxis=dict(
                                range=[0, 2],
                                tickmode="linear",
                                dtick=1
                            ),
                            height=345,
                            showlegend=False,
                            margin=dict(l=100, r=50, t=30, b=30)
                        )

                        con1 = st.container(border=True, height=355)
                        
                        with con1:
                            st.plotly_chart(fig, use_container_width=True)


                if selected_time in breakdowns:
                    scores, details = breakdowns[selected_time]

                    record = next((r for r in results if r["year"] == selected_time), None)
                    if record:
                        score = record["score"]
                        severity = record["severity"]
                        percent = round(clf.predict_proba([[score]]).squeeze()[-1] * 100, 2)

        
                    time_list = [r['year'] for r in results]
                    prev_time = time_list[0]
                    record_2 = next((r for r in results if r["year"] == prev_time), None)

                    if record_2:
                        score_2 = record_2["score"]
                        severity_2 = record_2["severity"]
                        percent_2 = round(clf.predict_proba([[score_2]]).squeeze()[-1] * 100, 2)
                    else:
                        score_2, percent_2 = None, None

                
                    recommendation_dict = {
                        "Low": "Continue regular follow-up and lifestyle management.",
                        "Medium": "Consider intensifying monitoring and reviewing medications.",
                        "High": "Immediate intervention required."
                    }
                    recommendation = recommendation_dict.get(severity, "No recommendation available.")

            
                    delta_value = score - score_2 if score_2 is not None else None
                    percent_delta = percent - percent_2 if percent_2 is not None else None

                    mortality_delta = f"{round(percent_delta, 2)}%"
                    with subcol2:
                        a, b = st.columns(2)
                        a.metric(
                            label="aDCSI Score",
                            value=score,
                            border=True,
                            help="Score interpretation\n - Low: 0–2\n- Medium: 3–4\n- High: ≥5"
                        )

                        b.metric(
                            label="Mortality (%)",
                            value=f"{percent}%",
                            border=True,
                            help="Mortality estimated using univariable logistic regression with aDCSI as predictor."

                        )
                        
                        display_severity(severity)

            with maincol2:

                model_name = st.selectbox(
                    "Select LLM Model",
                    options=["gpt-oss:20b", "llama3.1:latest", "gemma3:12b", "richardyoung/llama-medx_v32:latest"],
                    on_change=clear_chat_history,)
                
                llm = ChatOllama(model=model_name) 

                def load_knowledge_base(file_path="knowledge_base.txt"):
               
                    try:
                        if os.path.exists(file_path):
                            with open(file_path, "r", encoding="utf-8") as f:
                                return f.read()
                        else:
                            st.warning(f"⚠️ 找不到知識庫檔案: {file_path}")
                            return ""
                    except Exception as e:
                        st.error(f"❌ 讀取知識庫時發生錯誤: {e}")
                        return ""


                system_prompt = (
                    "你是一位專精於糖尿病衛教的AI助理。"
                    "你的主要語言為繁體中文,次要語言為英文。"
                    "請根據使用者的提問,提供有關糖尿病及其併發症的衛教資訊。"
                    "你的回答應該簡潔明瞭,並且易於理解。"
                    "避免使用過於專業的術語,除非使用者特別要求。"
                )

                rag_knowledge = load_knowledge_base()
                # rag_knowledge = False
                if rag_knowledge:
                    system_prompt += f"{rag_knowledge}\n\n請在回答時適時參考上述資料,用易懂的方式解釋。"
                patient_context = f"""
                    
當前分析的病人資訊(若使用者有詢問當前病人相關資訊請參考以下內容):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【基本資料】
- 病人編號: {selected_pid}
- 年齡: {patient_df['Age'].iloc[0]} 歲
- 性別: {patient_df['Sex'].iloc[0]}
- 種族: {patient_df['Race'].iloc[0]}

【最新評估結果】(時間: {selected_time})
- aDCSI 總分: {score} 分
- 嚴重度等級: {severity}
- 預估死亡率: {percent}%

【併發症分數明細】
"""
                
             
                for category, cat_score in scores.items():
                    if category != "Total":
                        patient_context += f"- {category}: {cat_score} 分\n"
                
     
                patient_context += "\n【診斷代碼 (ICD) 詳細資訊】\n"
                for category, icd_list in details.items():
                    if icd_list:
                        patient_context += f"\n{category}:\n"
                        for icd_code in icd_list:
                            patient_context += f"  • {icd_code}\n"
                
        
                if len(results) > 1:
                    patient_context += f"\n【歷史趨勢】\n"
                    patient_context += f"- 總共有 {len(results)} 個時間點的記錄\n"
                    if delta_value is not None:
                        patient_context += f"- aDCSI 分數變化: {'+' if delta_value > 0 else ''}{delta_value}\n"
                    if percent_delta is not None:
                        patient_context += f"- 死亡率變化: {'+' if percent_delta > 0 else ''}{round(percent_delta, 2)}%\n"
                
                patient_context += f"\n【臨床建議】\n{recommendation}\n"
                patient_context += "\n【補充資訊】死亡率是使用單變量邏輯斯迴歸模型估算,以 aDCSI 分數為預測變數。\n"
                patient_context += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"



                system_prompt += patient_context

                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("placeholder", "{chat_history}"),
                    ("human", "{question}")
                ])


                chain = chat_prompt | llm

                st.subheader("💬 Chatbot")

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                chat_container = st.container(height=690)  
                with chat_container:    
                    with st.expander("📋 使用須知與免責聲明", expanded=False):
                        st.markdown("""
                        **使用須知:**
                        - 此 AI 助理基於糖尿病衛教知識庫提供資訊
                        - 回答內容僅供參考,不構成醫療建議
                        - 請勿將 AI 建議作為診斷或治療依據
                        
                        **免責聲明:**
                        - 本系統無法取代醫師、營養師或衛教師的專業判斷
                        - 任何用藥、飲食或運動調整,請務必諮詢您的醫療團隊
                        - 如遇緊急狀況(如嚴重低血糖、意識不清),請立即就醫
                        
                        **資料隱私:**
                        - 對話內容僅用於本次諮詢,不會被儲存或分享
                        """)

                with chat_container:
           
                    for msg in st.session_state.messages:
                        with st.chat_message(msg["role"]):
                            st.write(msg["content"])
                

                if user_input := st.chat_input("輸入你的問題"):
            
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    
                    chat_history = []

                    for msg in st.session_state.messages[:-1]: 
                        if msg["role"] == "user":
                            chat_history.append(("human", msg["content"]))
                        else:
                            chat_history.append(("assistant", msg["content"]))
        
                    def stream_response():
                        full_response = ""
                        for chunk in chain.stream({
                            "question": user_input,
                            "chat_history": chat_history
                        }):
                            content = chunk.content
                            full_response += content
                            yield content
                    
                        st.session_state.messages.append({"role": "assistant", "content": full_response})


                    with chat_container:

                        with st.chat_message("user"):
                            st.write(user_input)
                        
                        with st.chat_message("assistant"):
                            st.write_stream(stream_response())

        
        else:
            st.info("ℹ️ Please select a patient PID")
else:
    st.info("Please upload CSV file.")