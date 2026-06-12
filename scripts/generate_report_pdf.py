# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin

ROOT = Path.cwd()
OUT = ROOT / 'data' / 'reports'
ASSETS = OUT / 'report_assets'
ASSETS.mkdir(parents=True, exist_ok=True)
PDF = OUT / 'knowledge_graph_final_report.pdf'
MD = OUT / 'knowledge_graph_final_report.md'
FONT = r'C:\Windows\Fonts\msyh.ttc'
FONT_BOLD = r'C:\Windows\Fonts\msyhbd.ttc'
CODE_FONT = r'C:\Windows\Fonts\consola.ttf'
A4 = (1240, 1754)
M = 88
W = A4[0] - 2*M
INK=(31,41,55); MUTED=(86,98,115); TEAL=(15,118,110); BLUE=(37,99,235); GREEN=(22,163,74); ORANGE=(217,119,6); RED=(220,38,38); LIGHT=(248,250,252)
report=json.loads((OUT/'graph_report_final.json').read_text(encoding='utf-8'))
examples=json.loads((OUT/'report_app_examples.json').read_text(encoding='utf-8'))

def ft(size,bold=False,code=False):
    p = CODE_FONT if code and Path(CODE_FONT).exists() else (FONT_BOLD if bold else FONT)
    return ImageFont.truetype(p,size)

def wrap(draw,text,font,maxw):
    lines=[]
    for para in str(text).split('\n'):
        cur=''
        for ch in para:
            t=cur+ch
            if draw.textlength(t,font=font)<=maxw or not cur: cur=t
            else: lines.append(cur); cur=ch
        lines.append(cur)
    return lines

def draw_card(draw,box,title,body,color):
    x1,y1,x2,y2=box
    draw.rounded_rectangle(box,radius=18,fill=LIGHT,outline=color,width=4)
    draw.text((x1+22,y1+22),title,font=ft(25,True),fill=color)
    yy=y1+62
    for line in body.split('\n'):
        draw.text((x1+22,yy),line,font=ft(19),fill=INK); yy+=30

def architecture():
    img=Image.new('RGB',(1100,650),'white'); d=ImageDraw.Draw(img)
    draw_card(d,(60,80,300,205),'数据源','10000条JD\nO*NET/本地词表',BLUE)
    draw_card(d,(430,80,680,205),'构建流水线','弱标注/实体抽取\n关系抽取/融合',TEAL)
    draw_card(d,(815,80,1040,205),'图谱存储','CSV文件\nNeo4j适配层',GREEN)
    draw_card(d,(230,360,510,505),'API服务','FastAPI\nGraphStore统一接口',ORANGE)
    draw_card(d,(640,360,930,505),'网页应用','可视化/推荐\n差距/路径/画像',RED)
    for a,b in [((305,145),(425,145)),((685,145),(812,145)),((910,210),(790,358)),((555,210),(390,358)),((512,433),(638,433))]:
        d.line((*a,*b),fill=MUTED,width=5); d.polygon([(b[0],b[1]),(b[0]-15,b[1]-9),(b[0]-15,b[1]+9)],fill=MUTED)
    d.text((60,585),'图1 系统架构：岗位数据进入构建流水线，生成图谱并通过网页应用提供下游任务。',font=ft(20),fill=MUTED)
    p=ASSETS/'architecture.png'; img.save(p); return p

def pipeline():
    img=Image.new('RGB',(1100,620),'white'); d=ImageDraw.Draw(img)
    d.text((45,50),'知识图谱构建流程',font=ft(36,True),fill=TEAL)
    steps=['原始岗位文本','标题清洗','词表索引','实体抽取','关系抽取','实体对齐融合','图谱导出']
    colors=[BLUE,BLUE,TEAL,TEAL,ORANGE,GREEN,RED]
    x=45; y=160
    for i,s in enumerate(steps):
        d.rounded_rectangle((x,y,x+130,y+95),radius=14,fill=LIGHT,outline=colors[i],width=3)
        for j,line in enumerate(wrap(d,s,ft(22,True),105)[:2]): d.text((x+15,y+22+j*28),line,font=ft(22,True),fill=colors[i])
        if i<len(steps)-1:
            d.line((x+135,y+48,x+183,y+48),fill=MUTED,width=4); d.polygon([(x+183,y+48),(x+168,y+39),(x+168,y+57)],fill=MUTED)
        x+=150
    d.text((45,320),f"完整输入：10000条岗位；弱标注样本：35962条；gazetteer：9168条；抽取关系：{report['num_relations']}条。",font=ft(21),fill=INK)
    d.text((45,370),'核心关系：REQUIRES_SKILL、USES_TOOL、REQUIRES_KNOWLEDGE；增强关系：SIMILAR_TO、PREREQUISITE_OF。',font=ft(21),fill=INK)
    d.text((45,540),'图2 构建流程与统计信息。',font=ft(20),fill=MUTED)
    p=ASSETS/'pipeline.png'; img.save(p); return p

def app_result():
    img=Image.new('RGB',(1100,700),'white'); d=ImageDraw.Draw(img)
    d.rounded_rectangle((35,35,1065,665),radius=22,fill=(15,23,42))
    d.text((70,70),'下游应用运行结果',font=ft(36,True),fill='white')
    d.text((70,130),'输入技能：Java, Spring, SQL',font=ft(24),fill=(191,219,254))
    y=190; d.text((80,y),'岗位推荐 Top-5',font=ft(27,True),fill=(134,239,172)); y+=50
    for i,r in enumerate(examples['recommend_java']['recommendations'][:5],1):
        d.text((90,y),f"{i}. {r['occupation']} | 命中:{r['hit_count']} 分数:{r['score']}",font=ft(22),fill=(226,232,240)); y+=42
    y+=25; d.text((80,y),'技能差距示例：Python Developer + 已掌握 Python',font=ft(25,True),fill=(253,186,116)); y+=45
    d.text((90,y),'输出：missing_skills = []，说明当前输入覆盖该岗位已抽取核心技能。',font=ft(22),fill=(226,232,240))
    d.text((70,630),'图3 由真实 API 输出渲染的应用结果截图。',font=ft(20),fill=(148,163,184))
    p=ASSETS/'app_result.png'; img.save(p); return p

arch=architecture(); pipe=pipeline(); app=app_result()
items=[]
def H(t): items.append(('h',t))
def P(t): items.append(('p',t))
def C(t): items.append(('c',t))
def F(path,cap): items.append(('f',path,cap))
def T(rows,widths): items.append(('t',rows,widths))

H('封面')
P('小组项目名称：职业技能知识图谱构建与应用系统【待确认】')
P('小组成员：吕天白（学号【待补充】）、赵金波（学号【待补充】）。如课程要求列出全部成员，请继续补充。')
P('报告撰写时间：2026年6月12日')
P('仓库地址：git@github.com:blackblake/Knowledge-Engineering.git（待提交当前代码覆盖远程后作为最终仓库地址）')
H('1 小组分工')
T([['成员','主要任务','对应报告小节'],['吕天白','总体方案设计、数据处理、实体/关系抽取、实体对齐融合、图谱构建、存储后端、API与下游任务实现、报告主体撰写','2.1-2.7，3.1-3.2'],['赵金波','参与可视化页面展示与前端交互相关工作','2.7，3.1']], [140,650,260])
T([['成员','贡献占比','电子签'],['吕天白','80%','吕天白'],['赵金波','20%【根据总和100%暂列，待确认】','赵金波【待确认】']], [220,390,440])
P('说明：课程要求贡献占比总和为100%。已知吕天白贡献80%，赵金波参与可视化工作，因此报告草稿暂列赵金波20%；若实际比例不同，应在提交前修改。')
H('2 知识图谱构建')
H('2.1 数据源、获取方法和原始数据统计信息')
P('本项目面向招聘岗位场景构建职业-技能知识图谱。主要数据源为本地处理后的岗位描述样本 jd_sample.jsonl，以及由 O*NET/本地规则整理得到的 formal_gazetteer.jsonl。岗位数据包含岗位标题、语言和岗位描述文本；词表覆盖职业、技能、工具、知识等实体类型。')
T([['数据文件','用途','统计'],['data/processed/crf/jd_sample.jsonl','岗位标题与岗位描述原始输入','10000条岗位记录'],['data/processed/full_weak_ner.jsonl','弱监督实体抽取/构图输入','35962条句级样本'],['data/processed/full_gazetteer.jsonl','实体规范词表与别名','9168条规范词条']], [390,400,260])
F(pipe,'图2 知识图谱构建流程')
H('2.2 实体抽取的输入、方法和结果')
P('实体抽取输入为岗位标题与岗位描述句子。系统先清洗岗位标题作为 OCC 职业实体，再用 gazetteer 建立 n-gram 索引，对岗位描述进行最长匹配，识别 SKL 技能、TOL 工具、KNW 知识等实体。该方法可解释、可复现，并能保证 Java、Python、React、Spring 等关键技术词进入图谱。')
C("""def _find_term_mentions(text, term_index):
    tokens, token_spans = tokenize_with_spans(text)
    for start in range(len(tokens)):
        for end in range(min(len(tokens), start + MAX_TERM_TOKENS), start, -1):
            entry = term_index.get(tuple(t.casefold() for t in tokens[start:end]))
            if entry:
                # 输出实体 mention: text/start/end/type/canonical_id
                ...""")
P('实体抽取结果：生成35962条弱监督NER样本，覆盖9168条gazetteer词条中的4176个可索引术语。职业标题经过编号、奖金信息、括号信息等清洗，减少了脏职业节点。')
H('2.3 关系抽取的输入、方法和结果')
P('关系抽取输入为已标注实体的句子。每条样本至少包含一个 OCC 职业实体，系统将句中其他实体与最近的职业实体建立关系。不同实体类型映射到不同关系：SKL -> REQUIRES_SKILL，TOL -> USES_TOOL，KNW -> REQUIRES_KNOWLEDGE。')
C("""_RELATION_MAP = {'SKL':'REQUIRES_SKILL','KNW':'REQUIRES_KNOWLEDGE','TOL':'USES_TOOL'}
def extract_relations(example):
    spans = extract_entity_spans(example)
    occupations = [s for s in spans if s.entity_type == 'OCC']
    for span in spans:
        relation = _RELATION_MAP.get(span.entity_type)
        closest = min(occupations, key=lambda occ: abs(occ.start_token - span.start_token))""")
T([['关系类型','数量'],['REQUIRES_SKILL',report['relation_counts']['REQUIRES_SKILL']],['USES_TOOL',report['relation_counts']['USES_TOOL']],['REQUIRES_KNOWLEDGE',report['relation_counts']['REQUIRES_KNOWLEDGE']],['SIMILAR_TO',report['relation_counts']['SIMILAR_TO']],['PREREQUISITE_OF',report['relation_counts']['PREREQUISITE_OF']]], [520,260])
H('2.4 实体消岐的输入、方法和结果')
P(f"实体消岐输入为关系抽取阶段产生的实体文本与gazetteer规范词条。系统先使用规范化精确匹配，再使用fuzzy matching进行相似度对齐，并在融合阶段合并同一canonical_id下的别名和重复关系。对齐结果：head coverage={report['alignment']['head_coverage']}，tail coverage={report['alignment']['tail_coverage']}；融合后实体{report['fusion']['merged_entities']}个，融合关系{report['fusion']['merged_relations']}条，合并别名{report['fusion']['merged_alt_names']}个。")
H('2.5 事件抽取')
P('本项目未单独实现事件抽取模块。原因是课程项目核心目标是职业-技能知识图谱及其下游应用，岗位文本中的事件结构较弱，主要信息以实体和实体关系形式表达。')
H('2.6 其他：知识推理与图谱增强')
P('系统在基础抽取关系之外构建 SIMILAR_TO 职业相似关系和 PREREQUISITE_OF 技能先修关系。SIMILAR_TO 根据职业技能集合重叠度生成，用于成长路径规划；PREREQUISITE_OF 根据技能全局出现频次构造启发式先修链，用于展示知识推理能力。')
H('2.7 知识图谱的规模、存储和可视化结果')
T([['指标','数值'],['节点数',report['num_nodes']],['边数',report['num_edges']],['抽取关系数',report['num_relations']],['存储后端','CSV文件 + Neo4jGraphStore适配层'],['网页可视化','FastAPI + vis-network；默认展示高证据子图，避免百万边前端卡顿']], [430,620])
F(arch,'图1 系统架构与存储/可视化方式')
H('3 基于知识图谱的应用')
H('3.1 系统基本功能')
P('系统提供一个本地网页应用，包含图谱可视化和四类下游任务：岗位推荐、技能差距分析、职业成长路径规划和岗位画像查询。用户输入技能列表或目标职业后，前端调用FastAPI接口，后端基于GraphStore从完整图谱中检索与计算结果。')
F(app,'图3 下游任务运行结果示例')
T([['功能','输入','输出'],['岗位推荐','技能列表，例如Java, Spring, SQL','推荐相关岗位及命中数、分数'],['技能差距','目标岗位 + 已掌握技能','仍缺少的技能/工具/知识'],['成长路径','源岗位 + 目标岗位','基于SIMILAR_TO的路径'],['岗位画像','岗位名称','该岗位关联技能、工具、知识和证据数']], [250,360,450])
H('3.2 如何使用知识图谱的实现细节')
P('网页应用不是静态写死结果，而是通过统一GraphStore接口使用知识图谱。CSVGraphStore和Neo4jGraphStore暴露相同方法：summary、graph_raw、occupation_profile、recommend、skill_gap和growth_path。因此系统可以在CSV后端与Neo4j后端之间切换。')
C("""def recommend(self, skills, top_k):
    skill_set = {_normalize(skill) for skill in skills}
    for edge in self.edges:
        if edge['relation'] in {'REQUIRES_SKILL','USES_TOOL','REQUIRES_KNOWLEDGE'}:
            # 对 OCC -> 技能/工具/知识 边累积命中权重
            ...
    return ranked[:top_k]""")
P('以岗位推荐为例，系统只统计OCC职业节点指向SKL/TOL/KNW节点的关系边。输入技能被规范化后与图谱节点名称匹配，命中边按evidence_count的log权重累积，并综合命中技能数、标题命中和证据强度排序。')
P('以技能差距为例，系统查询目标职业的REQUIRES_SKILL边，与用户已掌握技能集合做差集，输出缺失技能、是否核心要求、证据数和熟练度信息。')
H('提交说明')
P('本报告已按课程要求组织为：封面、小组分工、知识图谱构建、基于知识图谱的应用。最终提交形式为PDF。当前草稿中仍需人工确认的字段包括：项目正式名称、成员学号、赵金波贡献占比/电子签，以及是否需要列出除吕天白和赵金波以外的全部成员。')
P('GitHub仓库地址要求为git@github.com:blackblake/Knowledge-Engineering.git。由于覆盖远程仓库属于有风险操作，需要在提交前确认当前工作区内容、远程权限和是否允许force push。')

# markdown
md=['# 职业技能知识图谱构建与应用系统结课报告\n','**报告撰写时间：2026年6月12日**\n','> 注：学号、正式项目名、完整成员名单等未知信息以【待补充/待确认】标注。\n']
for it in items:
    if it[0]=='h': md.append('\n## '+it[1]+'\n')
    elif it[0]=='p': md.append(it[1]+'\n')
    elif it[0]=='c': md.append('```python\n'+it[1]+'\n```\n')
    elif it[0]=='f': md.append(f'![{it[2]}]({Path(it[1]).as_posix()})\n')
    elif it[0]=='t': md.append('[表格见PDF排版]\n')
MD.write_text('\n'.join(md),encoding='utf-8')

pages=[]
def newpage():
    img=Image.new('RGB',A4,'white'); return img,ImageDraw.Draw(img),M
img,d,y=newpage()
def add_text(text,size=24,bold=False,color=INK,indent=0,code=False):
    global img,d,y
    f=ft(size,bold,code)
    for line in wrap(d,text,f,W-indent):
        if y>A4[1]-130:
            pages.append(img); img,d,y=newpage(); f=ft(size,bold,code)
        d.text((M+indent,y),line,font=f,fill=color); y+=size+11
    y+=8

def add_heading(text):
    global y
    if y>A4[1]-220:
        pages.append(img); globals()['img'],globals()['d'],globals()['y']=newpage()
    globals()['d'].rounded_rectangle((M,y,M+12,y+43),radius=5,fill=TEAL)
    add_text(text,31,True,TEAL,24)

def add_table(rows,widths):
    global img,d,y
    rh=60
    if y+rh*len(rows)>A4[1]-120:
        pages.append(img); img,d,y=newpage()
    x0=M
    for r,row in enumerate(rows):
        fill=(236,253,245) if r==0 else 'white'
        d.rectangle((x0,y,x0+sum(widths),y+rh),fill=fill,outline=(203,213,225))
        x=x0
        for c,cell in enumerate(row):
            d.line((x,y,x,y+rh),fill=(203,213,225))
            f=ft(20,r==0)
            yy=y+9
            for line in wrap(d,str(cell),f,widths[c]-16)[:2]:
                d.text((x+8,yy),line,font=f,fill=INK); yy+=24
            x+=widths[c]
        d.line((x0+sum(widths),y,x0+sum(widths),y+rh),fill=(203,213,225))
        y+=rh
    y+=18

for it in items:
    if it[0]=='h': add_heading(it[1])
    elif it[0]=='p': add_text(it[1])
    elif it[0]=='c':
        if y>A4[1]-430: pages.append(img); img,d,y=newpage()
        d.rounded_rectangle((M,y,M+W,y+310),radius=12,fill=(17,24,39))
        yy=y+22
        for line in it[1].split('\n')[:10]:
            d.text((M+24,yy),line[:95],font=ft(19,code=True),fill=(229,231,235)); yy+=28
        y+=335
    elif it[0]=='f':
        fig=Image.open(it[1]).convert('RGB'); scale=min(W/fig.width,520/fig.height); sz=(int(fig.width*scale),int(fig.height*scale))
        if y+sz[1]+90>A4[1]-110: pages.append(img); img,d,y=newpage()
        fig=fig.resize(sz); img.paste(fig,(M+(W-sz[0])//2,y)); y+=sz[1]+10; add_text(it[2],21,False,MUTED)
    elif it[0]=='t': add_table(it[1],it[2])
pages.append(img)
for i,p in enumerate(pages,1): ImageDraw.Draw(p).text((A4[0]//2-28,A4[1]-70),f'- {i} -',font=ft(18),fill=MUTED)
pages[0].save(PDF,save_all=True,append_images=pages[1:],resolution=150.0)
print(json.dumps({'pdf':str(PDF),'markdown':str(MD),'pages':len(pages)},ensure_ascii=False,indent=2))
