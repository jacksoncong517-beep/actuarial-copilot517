# ==========================================
# Actuarial Copilot Pro
# AI Actuarial Assistant
# ==========================================

import os
import numpy as np
import pandas as pd
import pdfplumber
import re
import gradio as gr

from numpy_financial import irr
from numpy.linalg import svd

from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ==========================================
# 生命表读取
# ==========================================

life_pdf = "shengmingbiao.pdf"

mortality = {}

try:
    with pdfplumber.open(life_pdf) as pdf:
        page = pdf.pages[0]
        text = page.extract_text()

    for line in text.split("\n"):
        nums = re.findall(r"\d+\.\d+|\d+", line)

        if len(nums) >= 3:
            age = int(nums[0])
            q = float(nums[1])
            mortality[age] = q

except:
    # fallback
    for a in range(20,100):
        mortality[a] = 0.002


# ==========================================
# 精算计算模块
# ==========================================

def calc_irr(cashflows):

    try:
        irr_val = irr(cashflows)
        return f"IRR = {irr_val:.4f}"

    except:
        return "IRR无法计算"


def expected_claim(age, benefit):

    qx = mortality.get(age,0.002)

    return benefit * qx


def net_premium(age, benefit, interest):

    qx = mortality.get(age,0.002)

    return benefit * qx / (1 + interest)


def calc_nbv(premium, claim, expense, interest):

    cf = premium - claim - expense

    return cf / (1 + interest)


def profit_projection(age, benefit, term, interest, lapse, expense, commission):

    qx = mortality.get(age,0.002)

    premium = benefit * qx / (1 + interest)

    cashflows = []

    for t in range(term):

        claim = benefit * qx
        exp = premium * expense
        comm = premium * commission if t == 0 else 0
        lapse_loss = premium * lapse

        profit = premium - claim - exp - comm - lapse_loss

        cashflows.append(profit)

    pv = sum(cf/(1+interest)**(i+1) for i,cf in enumerate(cashflows))

    irr_val = irr([-benefit] + cashflows)

    txt = "\n".join([f"Year {i+1}: {cf:.2f}" for i,cf in enumerate(cashflows[:10])])

    return f"""
Net Premium: {premium:.2f}

Cashflows:
{txt}

NBV: {pv:.2f}

IRR: {irr_val:.4f}
"""


# ==========================================
# Lee Carter（简化）
# ==========================================

def lee_carter_forecast():

    ages = sorted(mortality.keys())

    mx = np.array([mortality[a] for a in ages])

    logmx = np.log(mx + 1e-6)

    U,S,Vt = svd(logmx.reshape(-1,1))

    k = S[0]

    forecast = np.exp(logmx + k*0.01)

    result = "\n".join([
        f"{a}岁预测死亡率 {forecast[i]:.6f}"
        for i,a in enumerate(ages[:10])
    ])

    return result


# ==========================================
# LLM
# ==========================================

def llm(prompt):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content


# ==========================================
# AI Agent
# ==========================================

def actuarial_agent(q):

    if "IRR" in q:
        return "IRR是内部收益率，用于衡量保险现金流回报。"

    if "死亡率" in q:
        return lee_carter_forecast()

    if "IFRS" in q:

        prompt = f"""
你是保险精算专家，请解释以下IFRS17问题：

{q}
"""

        return llm(prompt)

    return llm(q)


# ==========================================
# UI
# ==========================================

def premium_ui(age,benefit,interest):

    p = net_premium(age,benefit,interest)

    return f"净保费 {p:.2f}"


def claim_ui(age,benefit):

    c = expected_claim(age,benefit)

    return f"期望赔付 {c:.2f}"


def nbv_ui(premium,claim,expense,interest):

    nbv = calc_nbv(premium,claim,expense,interest)

    return f"NBV {nbv:.2f}"


def irr_ui(c0,c1,c2,c3,c4,c5):

    cashflows=[c0,c1,c2,c3,c4,c5]

    return calc_irr(cashflows)


def profit_ui(age,benefit,term,interest,lapse,expense,commission):

    return profit_projection(age,benefit,term,interest,lapse,expense,commission)


def lc_ui():

    return lee_carter_forecast()


def ask(q):

    return actuarial_agent(q)


# ==========================================
# Web App
# ==========================================

with gr.Blocks(title="Actuarial Copilot Pro") as demo:

    gr.Markdown("# Actuarial Copilot Pro")
    gr.Markdown("AI精算助手")

    gr.Markdown("## AI保险问答")

    q = gr.Textbox(lines=3)
    btn = gr.Button("提问")
    ans = gr.Textbox(lines=10)

    btn.click(ask,q,ans)

    gr.Markdown("---")

    with gr.Tab("保费计算"):

        age = gr.Number(label="年龄",value=30)
        benefit = gr.Number(label="保额",value=100000)
        interest = gr.Number(label="利率",value=0.03)

        btn = gr.Button("计算")
        out = gr.Textbox()

        btn.click(premium_ui,[age,benefit,interest],out)


    with gr.Tab("期望赔付"):

        age2 = gr.Number(label="年龄",value=30)
        benefit2 = gr.Number(label="保额",value=100000)

        btn2 = gr.Button("计算")
        out2 = gr.Textbox()

        btn2.click(claim_ui,[age2,benefit2],out2)


    with gr.Tab("NBV"):

        p = gr.Number(label="保费")
        c = gr.Number(label="赔付")
        e = gr.Number(label="费用")
        r = gr.Number(label="利率",value=0.03)

        btn3 = gr.Button("计算")
        out3 = gr.Textbox()

        btn3.click(nbv_ui,[p,c,e,r],out3)


    with gr.Tab("IRR"):

        c0 = gr.Number(label="Year0",value=-1000)
        c1 = gr.Number(label="Year1",value=200)
        c2 = gr.Number(label="Year2",value=300)
        c3 = gr.Number(label="Year3",value=300)
        c4 = gr.Number(label="Year4",value=200)
        c5 = gr.Number(label="Year5",value=200)

        btn_irr = gr.Button("计算IRR")
        irr_out = gr.Textbox()

        btn_irr.click(irr_ui,[c0,c1,c2,c3,c4,c5],irr_out)


    with gr.Tab("Profit Testing"):

        age3 = gr.Number(label="年龄",value=30)
        benefit3 = gr.Number(label="保额",value=100000)
        term = gr.Number(label="期限",value=20)

        interest3 = gr.Number(label="利率",value=0.03)
        lapse = gr.Number(label="退保率",value=0.1)
        expense = gr.Number(label="费用率",value=0.2)
        comm = gr.Number(label="佣金率",value=0.5)

        btn4 = gr.Button("运行")
        out4 = gr.Textbox(lines=12)

        btn4.click(
            profit_ui,
            [age3,benefit3,term,interest3,lapse,expense,comm],
            out4
        )


    with gr.Tab("死亡率预测"):

        btn5 = gr.Button("运行Lee Carter")
        out5 = gr.Textbox(lines=10)

        btn5.click(lc_ui,None,out5)


# ==========================================
# Railway启动
# ==========================================

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT",7860))
)
