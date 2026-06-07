# Workshop 改造计划 — Stable Is Not Grounded → BlackboxNLP

> 目标档位:**重打包 + 加一个跨语言 witness**(已确认)。
> 投稿:BlackboxNLP @ EMNLP 2026(沿用 `outline.md` 的 claim 纪律)。
> 核心动作:把通用 shortcut-learning 语气**重新锚定到"句法结构 vs 表层线索"**,让 stipulated graph 由**语法/类型学**自然供给——这同时 (a) 向主页研究方向对齐,(b) 补上论文 §7 自己承认的 SC-grounded 自然 witness 缺口,(c) 落在 BlackboxNLP 主场(LM 的句法评测)。

源稿:`paper1_v2_stable_not_grounded.md`(§1–§8)、`paper1.tex`(~5200 词,注释自陈会超 8 页)。
**先确认 2026 CFP**:页数(预期 ACL 8 页正文 + 无限 refs)、archival vs 非archival、deadline。今天 2026-06-07,CFP 可能尚未发布。

---

## 0. 一句话定位(放进 intro 第一段)

本文是三段式 program 的**第一条腿——Certification**:判定一个 stable-correct 行为是被**句法结构**(支配一致的中心词、承载推理的依存)licensed,还是被一个只与答案相关的**表层线索**。Localization(Isotrace)与 Quantification(CIY)是后续腿。

破 circularity 的关键:**语法是一个天然存在、非投资者裁量的 stipulated graph**——"线性距离/邻近名词与主谓一致无关,相关的是句法主语"不是从模型推断的因果事实,是类型学给的(non-circular by kind)。

---

## 1. 删 / 压缩清单(腾篇幅给语言 witness)

| 当前章节 | 处理 | 理由 |
|---|---|---|
| §3.4 The cut recurses | **删**,并入一句 scope note | recursion 是 companion paper 的活 |
| §4.2 gate-shaped boundary | **删** | 注释已点名先砍这里;workshop 不需要 gate lattice |
| §4.3 one recursive consequence | **压成 1 句**放进 §4 scope | eval-contingent ranking 改由跨语言实验**经验地**展示,不必再抽象论证 |
| §4 的 non-identifiability/ill-posed 形式语气 | 收成 2 句 scope condition | 去掉"undecidable/ill-posed"哲学,只留"两侧不可观测时 level-3 declined" |
| §8 two-paper split | **压成结尾 1 段** | 只保留"companion/thesis 处理形式不可判定性"一句指针 |
| §6.1 合成 `can X fly?` | **降级**为 sanity check / 非空性证明,不再做 headline | 改由语言 witness 当 headline |

净效果:省出约 1.5–2 页,正好换给新 §6 语言 witness。

---

## 2. 改 / 重新措辞(framing 对齐,不动结论)

- **§1 intro**:把"学生用 always 启发式"的人类类比,替换/并列为**句法版**——"模型用'与最近名词一致'的启发式:在大多数句子里对、改写不动它、稳健,却没学会句法"。直接引出 attractor。
- **§2 related**:McCoy(HANS)已是 *Diagnosing **Syntactic** Heuristics*;再补一条 agreement 评测线(Linzen et al. 2016;Gulordava et al. 2018;Marvin & Linzen 2018;Mueller et al. 2020 跨语言)。把"结构 vs 表层"立成本文谱系。
- **§6.4 HANS**:从"overlap-irrelevance"**重新措辞为句法启发式**(constituent/subsequence heuristic),与新实验同属一条线;数字不变。
- **retitle**(候选,放语言结构进标题):
  1. *Structure or Surface: Certifying Grounding with Linguistic Structure as the Stipulated Graph*
  2. *Stable Is Not Grounded: Syntax as a Non-Circular Stipulated Graph for Shortcut Certification*
  3. (保守)保留主标题 + 副标题加 "via Linguistic Structure"

---

## 3. 目标章节结构(8 页)

```
1  Intro            结构 vs 表层 + 三态层级 + program 第一腿(改 framing)
2  Related          扰动floor + 因果instrument + 句法评测谱系(补 agreement 线)
3  The Hierarchy    保留 §3.1–3.3;删 §3.4
4  When Licensed    保留 §4.1 两侧可观测;§4.2/4.3 → 1 段 scope note
5  Protocol         保留(基本不动)
6  Demonstration
   6.1 sanity: 合成 can-X-fly(降级,半页,non-emptiness)
   6.2 ★ NEW: 跨语言主谓一致 attractor witness(中心,见 §4)
   6.3 HANS(重新措辞为句法启发式,SC-spurious 自然单向)
   6.4 BoolQ boundary-bites(两侧暗,declines,保留)
7  Limits           收紧(SC-grounded 缺口现已被 6.2 部分补上 → 改写这条)
8  Conclusion       1 段 program/companion 指针
```

---

## 4. ★ 新实验:跨语言主谓一致 attractor witness

### 4.1 现象与映射
主谓一致 attractor 句是结构 vs 表层的经典探针。**结构**(语法 stipulate):动词与**句法主语**(中心词)一致,无视线性距离与中间干扰名词;**表层线索**(可审计相关):与**最近名词**一致(linear proximity)。

| 论文 machinery | 跨语言一致性实现 |
|---|---|
| §3.2(a) 构造性无关 | 语法规定"线性邻近与一致无关";类型学供图 → non-circular by kind |
| §3.2(b) 可测纠缠 | 训练/评测语料里 `corr(邻近名词数, 一致标签)`,可审计 |
| do(class-3) sever | **attractor 句**:主谓间插入相反数的干扰名词(结构说忽略、表层说采用) |
| class-2 negative control | number-neutral 干扰词(不携带 proximity 线索)→ 不应掉点 |
| **SC-grounded**(论文缺口) | 在 aligned **和** attractor 两 slice 都高 = 跟踪主语 |
| **SC-spurious** | aligned 高、attractor 崩 = 骑邻近线索 |
| §4.3 eval-contingent ranking(改经验展示) | 同模型英语过、德语挂:语法切开 proximity 的频率不同 |

### 4.2 数据 / 模型(低成本,现成)
- **数据**:CLAMS / Marvin & Linzen(en);**Mueller et al. 2020 跨语言一致性套件**(en/de/fr/ru/he)——不从零造。先 **en + de** 做主张,fr/ru 作 robustness。
- **模型**:小 LM 即可(A40 / A100-MIG 足够);沿用已有 multiseed 流程出 bootstrap 95% CI。建议至少跨 2 架构(与 §6.2 "not an architecture artifact" 呼应)。
- **指标**:per-slice accuracy(aligned / attractor)、do(class-3) drop、class-2 control、跨语言 ranking 反转。

### 4.3 这个实验关掉什么
- 补上 §7 承认的**自然 SC-grounded witness**(现仅 HANS 单向 SC-spurious)。
- 把 §4.3 的"排名 eval-contingent"从抽象推论变成**跨语言经验事实**。
- 把整篇从"合成玩具 + 通用 shortcut"抬到"自然语言 + 句法结构",= 主页方向。

### 4.4 claim 纪律护栏(不得越界)
- 仍是 by-construction(grammar 供图),**没有**滑向"证明某部署模型依赖 shortcut"或"新因果框架"——守住 `outline.md` 的红线。
- attractor 掉点只在 class-2 control 完好时才读作 shortcut(强制对照,§3.2)。
- 语义/语法保持性校验门控(数、性、格的扰动要 answer-preserving)。

---

## 5. Limits(§7)需改写的两条
- "naturalistic SC-grounded witness remains open" → 改为"**部分由 §6.2 跨语言一致性闭合**(aligned+attractor 双高);剩余开口:更多语系 / 更大模型"。
- portability(zero-shot DeepSeek/Qwen)保留,但 reframe 为"shortcut 标记 trained-in 纠缠"的同一句法叙事。

---

## 6. 执行顺序(我建议的落地步骤)
1. **(本文档,done)** 计划定稿。
2. 改 framing:§1/§2 重写 + HANS 重新措辞 + retitle —— 不依赖新数据,可立即做。
3. 删减:§3.4 / §4.2 / §4.3 / §8 压缩 —— 立即做,先把篇幅压到 8 页内。
4. 起跨语言实验脚手架:`experiments/paper1/09_crosslingual_agreement/`(CLAMS/Mueller 数据 loader + attractor 切片 + sbatch),跑 en+de 双架构 multiseed。
5. 回填 §6.2 witness + 改 §7 limits + 更新 abstract/evidence_map。
6. 同步进 `paper1.tex`,本地不编译 → Overleaf ACL 模板出 PDF,核对 ≤8 页。

> 第 2、3 步与第 4 步可并行:framing/裁剪不阻塞实验。
> 待你确认本计划后,我建议从**第 2+3 步(改稿)**起手,同时给出第 4 步的实验脚手架。
