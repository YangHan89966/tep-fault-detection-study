from pathlib import Path
import csv
import json

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT=Path(__file__).resolve().parents[1]
S=json.loads((ROOT/'results/SUMMARY.json').read_text())
L={x['metric']:x for x in S['metrics']}
V=list(csv.DictReader((ROOT/'results/ALL41_MEASUREMENT_DIFFERENCE_DECOMPOSITION.csv').open(encoding='utf-8-sig')))
D=Document()
sec=D.sections[0]
sec.page_width=Cm(21); sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.8); sec.bottom_margin=Cm(1.8)
sec.left_margin=Cm(1.9); sec.right_margin=Cm(1.9)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
    st=D.styles[name]
    st.font.name='Noto Serif CJK SC'
    st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Noto Serif CJK SC')
    st.font.color.rgb=RGBColor(0,0,0)
    st.paragraph_format.space_after=Pt(6)
    st.paragraph_format.line_spacing=1.15
D.styles['Normal'].font.size=Pt(10)
D.styles['Title'].font.size=Pt(20)
D.styles['Heading 1'].font.size=Pt(14)
D.styles['Heading 2'].font.size=Pt(11)
D.styles['Subtitle'].font.size=Pt(10)
for st in D.styles:
    for border in list(st._element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
TEXT=[]

def p(text,style=None,bold=False):
    para=D.add_paragraph(style=style)
    para.add_run(text).bold=bold
    TEXT.append(text)
    return para

def heading(text):
    p(text,'Heading 1')

def note(text):
    q=p(text)
    for run in q.runs: run.font.size=Pt(8.5)

def table(headers,rows,widths):
    t=D.add_table(rows=1,cols=len(headers))
    t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    for col,w in zip(t.columns,widths): col.width=Cm(w)
    for i,(c,txt,w) in enumerate(zip(t.rows[0].cells,headers,widths)):
        c.text=txt;c.width=Cm(w)
    for row in rows:
        cells=t.add_row().cells
        for c,txt,w in zip(cells,row,widths):c.text=str(txt);c.width=Cm(w)
    for ri,row in enumerate(t.rows):
        for ci,cell in enumerate(row.cells):
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr()
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'233D56' if ri==0 else ('F3F6F8' if ri%2==0 else 'FFFFFF'));pr.append(shade)
            borders=OxmlElement('w:tcBorders')
            for side in ['top','left','bottom','right']:
                b=OxmlElement('w:'+side);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
            pr.append(borders)
            margins=OxmlElement('w:tcMar')
            for side in ['top','left','bottom','right']:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),'85');el.set(qn('w:type'),'dxa');margins.append(el)
            pr.append(margins)
            for para in cell.paragraphs:
                para.paragraph_format.space_after=Pt(0)
                para.paragraph_format.line_spacing=1.05
                para.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size=Pt(8.6)
                    run.font.bold=(ri==0)
                    run.font.color.rgb=RGBColor(255,255,255) if ri==0 else RGBColor(0,0,0)
    repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
    TEXT.append(' | '.join(headers))
    TEXT.extend(' | '.join(str(x) for x in row) for row in rows)
    D.add_paragraph().paragraph_format.space_after=Pt(0)
    return t

def ci(name):
    x=L[name]
    return f"{x['estimate']:.6f}", f"{x['ci95_low']:.6f} 至 {x['ci95_high']:.6f}"

def newpage():D.add_page_break();TEXT.append('\n')

p('Fault 9 记录方差差距分析','Title')
p('ρ＝0 与 post-update 的差距拆解   2026年11月3日','Subtitle')
p('第一条分析已完成。400 条正常轨迹支持补充记录时序机制：3 秒内两个液位的去噪分量变化很小，但流量已经响应控制动作；XMEAS17 是固定 Fault 9 得分变化的主要贡献者。分析仪通道在两种记录中完全相同。',bold=True)
p('这项结果修正的是机制解释的范围。既有记录、监测器、检测率和阈值未被修改；本轮属于对已分析确认面板的事后机制分析。')
heading('同一固定监测器的直接替换结果')
rows=[]
for label,name in [('原始 post-update','variance_post_update'),('只将连续测量 1–22 换成 CI','variance_post_continuous_to_CI'),
                   ('只将分析仪 23–41 换成 CI','variance_post_analyzers_to_CI'),('只将 XMEAS17 换成 CI','variance_post_XMEAS17_to_CI'),
                   ('将十个流量通道换成 CI','variance_post_all10flows_to_CI'),('全部 41 个测量换成 CI','variance_post_all41_to_CI'),
                   ('同一时刻独立噪声 ρ＝0','variance_rho0')]:
    a,b=ci(name);rows.append([label,a,b])
table(['记录或替换方式','正常得分方差','配对运行重抽样的 95% 区间'],rows,[7.0,3.0,7.2])
note('CI 表示 controller-input。方差为 400×960 个原始有符号得分的 pooled sample variance，ddof＝1，不是 CUSUM 累积统计量方差。所有组合使用相同的历史仿射评分与原始尺度。区间为 20,000 次整条运行重抽样的边际描述性百分位区间。')
p('19 个分析仪通道共 7,296,000 对数值逐位相同，采样事件也相同；11 个操纵量同样逐位相同。因此，只替换连续测量就等于完整 CI 替换。原文“必须替换全部 41 个测量”应改成“替换全部测量足以恢复”，并报告本轮连续组已足够的结果。')
p('只换 XMEAS17 可把方差从 3.363 降到 2.173，但这同时改变记录时序和噪声配对，不能把其降低量直接当作 3.363−2.200 的因果贡献。下一步用共同噪声构造直接拆解这个差值。')

newpage()
heading('为什么两条液位的结论不能推广到流量')
p('从原始 Philox 随机调用记录恢复两次测量的加性噪声，再由带噪记录扣除噪声，得到去噪记录分量。对每个通道，测量差＝噪声差＋去噪分量差；其方差保留两倍协方差项。去噪分量仍包含原测量噪声经控制器传入过程的影响，不能称为完全不受噪声影响的物理过程。')
p('下表展示两个液位与两个出料流量，以各通道名义噪声标准差的平方为单位归一化，便于不同量纲比较。ddof＝0 与现行 Table S1 相同。十个流量及全部 41 通道的完整结果、原始量纲数值均保存在结果 CSV。')
rows=[]
selected=[11,14,13,16]
for j in selected:
    x=V[j]
    rows.append([x['variable'],f"{float(x['variance_diff_noise_normalized']):.5f}",f"{float(x['variance_noise_normalized']):.5f}",
                 f"{float(x['variance_signal_normalized']):.5f}",f"{float(x['twice_cov_normalized']):.5f}",f"{float(x['signal_share_percent']):.4f}%"])
table(['测量','总差方差','噪声差项','去噪分量项','两倍协方差','去噪项占比'],rows,[2.3,2.8,2.8,3.0,2.9,3.4])
note('每行总差方差＝噪声差方差＋去噪分量差方差＋2×协方差。占比是带符号协方差核算中的方差项比例，并非相互独立的因果贡献百分比。')
p('XMEAS12、15 两个液位的去噪差方差占比只有 0.0051% 与 0.0184%，与原 Table S1 一致；XMEAS14、17 两个出料流量则分别达到 95.28% 与 89.20%。因此，原结论对两液位成立，但不能扩展为所有测量在 3 秒内都不变。')
p('XMEAS14 的归一化去噪变化大于 XMEAS17；XMEAS17 在本监测器中的重要性由固定得分与协方差分析确定，而不是因为它在全部变量中变化最大。')
heading('源代码与记录共同支持的路径')
p('XMEAS15 → 控制器 8 → XMV8 → 阀位状态 → XMEAS17。阀 8 的时间常数为 5 秒，因此测量评价间相隔 3 秒时，流量可以已有明显响应。XMEAS17 又被控制器 11 用于生成 XMV11，这进一步影响测量与操纵量的联合协方差。')
p('本面板中，XMEAS17 去噪记录差与 CI 时点 XMEAS15 加性噪声的相关系数为 0.8833（95% 区间 0.8826–0.8840），与该路径一致。该关联是在既有闭环轨迹上观察到的，不等于独立干预控制器的实验。')

newpage()
heading('直接拆解 1.162997 的差距')
p('ρ＝0 由 CI 时点的去噪记录分量加独立测量噪声构成。先保持这一份独立噪声与所有操纵量不变，只把去噪分量换成实际 post-update 时点；随后再将噪声换回实际 post-update 噪声。这给出可精确相加的两步分解。')
table(['步骤','得分方差','相对上一步变化'],[
    ['同一 CI 时刻的 ρ＝0','2.200029','起点'],
    ['post-update 去噪分量＋同一份独立噪声','3.344703','＋1.144674'],
    ['实际 post-update 记录','3.363026','＋0.018323'],
    ['完整端点差','—','＋1.162997']],[9.0,4.1,4.1])
p('第一步的 1.144674（95% 区间 1.131978–1.157362）占端点差的 98.42%。第二步为 0.018323（0.006004–0.030728）。这是指定共同噪声与替换顺序下的统计分解；不能推广为所有记录方式或真实工厂中的固定因果比例。')
p('将第一步按对称协方差规则分配，XMEAS17 为 1.007282，其他连续测量合计为 0.137391，分析仪为 0。XMEAS17 约占该去噪分量变化效应的 88.0%，约占完整端点差的 86.6%。这些分配保留交互协方差，不与下面的另一种分解相加。')
heading('差距主要体现为测量与操纵量的协方差变化')
p('对线性得分的协方差矩阵作直接端点分解，以下六项精确相加为 1.162997。C 为连续测量，A 为分析仪，U 为操纵量。')
table(['协方差块','post-update 减 ρ＝0 的贡献'],[
    ['连续组内部 C','−0.002099'],['分析仪组内部 A','−0.000218'],['操纵量组内部 U','0.000000'],
    ['连续与分析仪 C–A','＋0.004591'],['连续与操纵量 C–U','＋1.156857'],['分析仪与操纵量 A–U','＋0.003867']],[10.5,6.7])
p('连续组得分方差从 5.816509 到 5.814410，几乎不变；2Cov(C,U) 则从 −8.178520 到 −7.021664，负协方差提供的抵消减弱 1.156857，占完整差值约 99.47%。因此更准确的解释是：记录时序改变了连续测量与操纵量的联合结构，使固定残差中的协方差抵消减弱。')

newpage()
heading('本次结果对应的最小改稿范围')
p('正文 2.2 第一处：将“每次模型评价都给测量加入新噪声”限定为连续测量 XMEAS1–22，并另外说明分析仪的采样保持。此面板所有分析仪的两个记录快照相同。')
p('正文 2.2 第二处：将“3 秒内过程变化可忽略”限定为 XMEAS12 和 XMEAS15 两个液位。随后补充流量的控制响应，引用本轮扩展方差分解。')
p('正文 5.2：分开解释两个比较。实际 post-update→CI 包含记录时序与噪声配对变化；同一 CI 时间的 ρ＝0→1 只操纵共享噪声分量。补充 XMEAS15→XMV8→XMEAS17 的记录时序路径，并将“all 41 measurements had to be replaced”改为证据实际支持的表述。')
p('SI：保留原 Table S1 的两液位结果和原样本标签，新增 400 条正常运行的扩展表及共同噪声分解表。不能把原 4/8 条面板静默替换成这 400 条面板，也不能把本轮重新命名为新增独立确认试验。')
p('讨论中的外推：ρ 是同一物理记录时点的理想化噪声共享参数，不能据此断言真实工厂必定位于两种实际记录方式之间。可表述为实际数据还可能受记录时点、滤波和压缩影响。')
heading('计算口径与核验')
p('数据为独立随机流确认面板的全部 400 条正常评价轨迹，每条 48 小时、960 条记录，共 384,000 条记录。模型为历史固定 H_D01_moment_A01_signed 仿射映射，保留全部 52 个输入与原始标准化、方向、截距。')
p('评分端点与旧存档逐点完全一致，最大误差为 0；噪声调度在全部 384,000 个记录锚点、最终计数和记录时间通过校验。每个连续通道的 native 噪声事件还用保存的累计随机数计数独立核对。测量分解最大绝对误差为 3.64×10⁻¹³；端点六块分解与五项时序噪声分解的加总误差均小于 4×10⁻¹⁵。')
p('不确定性使用 20,000 次配对整条运行重抽样，固定种子 2026110301；每次重新计算 pooled covariance，包含运行均值间差异。区间是条件于固定历史模型和既有噪声实现的边际描述性区间，不包含重新训练或重新校准的不确定性。')
p('本轮没有新过程仿真、训练、选方向、改阈值或故障性能重评。仅完成截图第 1 条及其直接需要的机制更正；Fault 15 替换和检出率–误报率曲线属于后续独立事项。')
heading('复核资料')
note('输入：TEP_P1_Normal_Evaluation_Data_v1_20261026.zip；TEP_P1_F9_Graded_Coupling_20261030.zip；P1_CCE_submission_v4.2_20261103.zip。算法依据：耦合包 PROTOCOL.json、code/recover_noise.py、code/run_coupling.py、code/score_core.py；控制路径依据：engine/vendor/src/tep/controllers.py、constants.py、python_backend.py 和 engine/code/dual_core.py、horizon.py。')
note('完整结果包包含每条运行的充分统计量、全部 41 通道方差分解、分组替换结果、两种精确加总分解、bootstrap、计算脚本、独立审核及英文最小改稿文本。结果文件可由所附汇总代码直接重算，无需重新下载原始轨迹；从头恢复需提供上面两个原始数据包。')

D.core_properties.title='Fault 9 记录方差差距分析'
D.core_properties.subject='rho0 and post-update variance decomposition using archived400 normal trajectories'
D.core_properties.author=''
D.core_properties.last_modified_by=''
out=ROOT/'TEP_P1_F9_Recording_Variance_Report_20261103.docx'
D.save(out)
(ROOT/'REPORT_ZH.txt').write_text('\n\n'.join(TEXT)+'\n',encoding='utf-8')
print(out)
