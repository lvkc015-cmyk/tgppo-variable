import math
import numpy as np


def _safe_tanh(x, s=1.0):
    return float(np.tanh(s * x))


def _ratio(num, den, cap=None):
    den = max(float(den), 1e-12)
    val = float(num) / den
    if cap is not None:
        return min(val, cap)
    return val


class RewardH1:
    """Baseline-normalized node efficiency + terminal bonus.

    Simple, scale-robust: penalizes nodes per step normalized by baseline nodes
    and bonuses at terminal based on speedup vs. baseline and problem status.
    """

    def __init__(self, logger=None, alpha=1.0, bonus_cap=3.0):
        self.logger = logger
        self.alpha = alpha
        self.bonus_cap = bonus_cap
        self.reset(1.0, 0.0, 1.0, "timelimit", 3600.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=3600.0, logger=None):
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)
        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 3600.0), 1.0)
        self.logger = logger or self.logger
        self.prev_nodes = 0

    def compute(self, model, done):
        nodes = int(model.getNNodes())
        dn = max(nodes - self.prev_nodes, 0)
        self.prev_nodes = nodes

        # Step penalty normalized by baseline nodes; bounded by tanh
        step_penalty = - _safe_tanh(dn / (self.B * 0.02 + 1.0), s=self.alpha)  # ~2% of baseline gives ~0.76 penalty
        if not done:
            if self.logger:
                self.logger.info(f"H1 step: nodes={nodes}, dn={dn}, penalty={step_penalty:.4f}")
            return step_penalty

        # Terminal bonus
        status = model.getStatus()
        gap = float(model.getGap())
        pdi = float(model.getPrimalDualIntegral())

        speedup = _ratio(self.B, max(nodes, 1), cap=self.bonus_cap)  # >1 if better than baseline
        if status == "optimal":
            bonus = 1.0 + 2.0 * speedup
        elif status in ("infeasible", "unbounded"):
            bonus = 0.5 + 1.5 * speedup
        elif status == "timelimit":
            # progress vs baseline
            gap_gain = _safe_tanh((self.baseline_gap - gap))
            pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
            bonus = 0.2 * speedup + 0.6 * gap_gain + 0.2 * pdi_gain
        else:
            bonus = 0.2 * speedup

        if self.logger:
            self.logger.info(f"H1 terminal: status={status}, nodes={nodes}, speedup={speedup:.3f}, bonus={bonus:.4f}")
        return float(bonus)


class RewardH2:
    """Log-scaled node efficiency + progress shaping.

    Adds: (1) log scaling to further damp huge baselines, (2) pace term comparing
    current nodes to a power-law expected curve, (3) gap/PDI improvement shaping.
    """

    def __init__(self, logger=None, scale=1.5, pace_power=0.7, w_nodes=0.5, w_pace=0.2, w_gap=0.2, w_pdi=0.1):
        self.logger = logger
        self.scale = scale
        self.pace_power = pace_power
        self.w_nodes = w_nodes
        self.w_pace = w_pace
        self.w_gap = w_gap
        self.w_pdi = w_pdi
        self.reset(1.0, 0.0, 1.0, "timelimit", 3600.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=3600.0, logger=None):
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        self.logB = math.log1p(self.B)
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)
        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 3600.0), 1.0)
        self.logger = logger or self.logger
        self.prev_nodes = 0
        self.prev_gap = float('inf')
        self.prev_pdi = 0.0

    def _node_efficiency(self, nodes):
        # 1 - log(nodes+1)/log(B+1) in [-inf,1]; map via tanh
        eff = 1.0 - (math.log1p(nodes) / self.logB)
        return _safe_tanh(eff, s=self.scale)

    def _pace_term(self, nodes, t):
        expected = self.B * (t ** self.pace_power)
        # positive when under expected nodes
        pace = (expected - nodes) / (expected + 1.0)
        return _safe_tanh(pace, s=self.scale)

    def compute(self, model, done):
        nodes = int(model.getNNodes())
        t = float(model.getSolvingTime()) / self.time_limit
        t = min(max(t, 0.0), 1.0)

        gap = float(model.getGap())
        if math.isinf(gap):
            gap = 1e6
        pdi = float(model.getPrimalDualIntegral())

        # Components
        r_nodes = self._node_efficiency(nodes)
        r_pace = self._pace_term(nodes, t)
        gap_impr = _safe_tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9)) if self.prev_gap < float('inf') else 0.0
        pdi_impr = _safe_tanh((self.prev_pdi - pdi) / self.baseline_pdi) if self.prev_pdi > 0 else 0.0

        self.prev_nodes = nodes
        self.prev_gap = gap
        self.prev_pdi = pdi

        if not done:
            reward = (self.w_nodes * r_nodes + self.w_pace * r_pace +
                      self.w_gap * gap_impr + self.w_pdi * pdi_impr)
            reward = float(np.clip(reward, -1.0, 1.0))
            if self.logger:
                self.logger.info(f"H2 step: nodes={nodes}, r_nodes={r_nodes:.3f}, r_pace={r_pace:.3f}, gap_impr={gap_impr:.3f}, pdi_impr={pdi_impr:.3f}, R={reward:.3f}")
            return reward

        # Terminal bonus: blend speedup with final quality
        status = model.getStatus()
        # 衡量搜索的节点相对baseline 用的节点数的比例, 用更少节点 → speedup > 1 → 好
        speedup = _ratio(self.B, max(nodes, 1), cap=4.0)
        gap_gain = _safe_tanh((self.baseline_gap - gap))
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)

        if status == "optimal":
            bonus = 1.0 + 2.5 * speedup
        elif status in ("infeasible", "unbounded"):
            bonus = 0.7 + 2.0 * speedup
        else:  # timelimit or other
            bonus = 0.4 * speedup + 0.4 * gap_gain + 0.2 * pdi_gain
        if self.logger:
            self.logger.info(f"H2 terminal: status={status}, speedup={speedup:.3f}, gap_gain={gap_gain:.3f}, pdi_gain={pdi_gain:.3f}, bonus={bonus:.3f}")
        return float(bonus)


class RewardH3:
    """Adaptive difficulty-aware reward.

    Uses a smooth mapping from baseline difficulty (log baseline nodes) to
    weights over components (nodes efficiency, progress, gap, PDI, time pace).
    Encourages anytime improvement on hard instances while still rewarding
    beating baseline on easy ones.
    """

    def __init__(self, logger=None, scale=2.0):
        self.logger = logger
        self.scale = scale
        self.reset(1.0, 0.0, 1.0, "timelimit", 600.0, logger)

    def reset(self, baseline_nodes, baseline_gap, baseline_pdi, solver_status, time_limit=600.0, logger=None):
        # 问题基线节点数，越大说明问题越难
        self.B = max(float(baseline_nodes or 1.0), 1.0)
        self.logB = math.log1p(self.B)
        self.baseline_gap = float(baseline_gap or 0.0)
        self.baseline_pdi = max(float(baseline_pdi or 1.0), 1.0)
        self.solver_status = solver_status or "timelimit"
        self.time_limit = max(float(time_limit or 600.0), 1.0)
        self.logger = logger or self.logger
        self.prev_nodes = 0
        self.prev_gap = float('inf')
        self.prev_pdi = 0.0
        self.prev_pb = None
        self.prev_db = None

        # Difficulty weight schedule via sigmoid over log baseline nodes
        # easy -> emphasize speedup; hard -> emphasize gap/PDI progress
        # map logB in [log(1), log(1e6)] roughly to [0,1]

        #归一化难度指数 [0,1]，用 log 节点数映射
        d = min(max((self.logB - math.log1p(1.0)) / (math.log1p(1e6) - math.log1p(1.0)), 0.0), 1.0)
        #权重 w_* 随难度调整
        #简单问题 (d≈0) → w_nodes 大，强调搜索效率。
        self.w_nodes = 0.55 * (1 - d) + 0.25 * d    # 0.55 -> 0.25
        #难问题 (d≈1) → w_gap 和 w_pdi 大，强调解质量
        self.w_gap = 0.10 * (1 - d) + 0.30 * d      # 0.10 -> 0.30
        self.w_pdi = 0.05 * (1 - d) + 0.20 * d      # 0.05 -> 0.20
        self.w_progress = 0.15 * (1 - d) + 0.15 * d # 0.15 -> 0.15
        self.w_pace = 0.15 * (1 - d) + 0.10 * d     # 0.15 -> 0.10
        #最后归一化使五个权重之和为 1
        s = self.w_nodes + self.w_gap + self.w_pdi + self.w_progress + self.w_pace
        self.w_nodes /= s; self.w_gap /= s; self.w_pdi /= s; self.w_progress /= s; self.w_pace /= s

    #节点效率奖励，节点数越少，eff 越大 → 奖励越高。
    def _node_eff(self, nodes):
        eff = 1.0 - (math.log1p(nodes) / self.logB)
        return _safe_tanh(eff, s=self.scale)

    #时间/进度奖励，难问题曲线更平滑，不强调早期加速；鼓励早期快找到解。
    def _pace(self, nodes, t):
        # piecewise expected curve: early aggressive, then conservative
        # exponent varies with difficulty; harder -> smaller exponent
        exponent = 0.9 - 0.5 * min(max(self.logB / math.log1p(1e6), 0.0), 1.0)
        expected = self.B * (t ** exponent)
        return _safe_tanh((expected - nodes) / (expected + 1.0), s=self.scale)

    def compute(self, model, done):
        nodes = int(model.getNNodes()) #当前 B&B 已扩展的节点数,越小越好（搜索更高效）
        #已用时间 / 时间上限 [0,1],表示进度条
        tfrac = min(max(model.getSolvingTime() / self.time_limit, 0.0), 1.0)
        gap = float(model.getGap()) #当前最优解与下界之间的差距百分比
        if math.isinf(gap):
            gap = 1e6

        #pdi越小越好,  原对偶间隙随时间变化的曲线下方围成的面积
        #举例：两个求解器可能都在 100 秒时找到了最优解。但求解器 A 在第 10 秒就找到了一个非常接近最优的解，而求解器 B 直到第 90 秒才找到。
        #求解器 A 的 PDI 更小，说明它在实际应用中更优秀
        pdi = float(model.getPrimalDualIntegral())
        #当前最好可行解(上界)
        pb = model.getPrimalbound()
        #当前松弛下界
        db = model.getDualbound()

        r_nodes = self._node_eff(nodes) #节点效率奖励，节点数越少，r_nodes 越大
        r_pace = self._pace(nodes, tfrac) #求解速度奖励,相同时间探索更多节点 → 奖励更高
        r_progress = 0.0   #进展奖励

        # gap improvement Gap 改善奖励
        if self.prev_gap < float('inf'):
            #计算当前步与上一步相比，原对偶间隙（Gap）缩小的比例。
            r_progress += _safe_tanh((self.prev_gap - gap) / max(abs(self.prev_gap), 1e-9), s=self.scale)
        # bound improvements  这是为了在 Gap 还没有整体大幅变动时，给智能体更细粒度的信号
        if self.prev_pb is not None and pb < self.prev_pb: #原问题上界（Primal Bound）下降
            r_progress += _safe_tanh((self.prev_pb - pb) / (abs(self.prev_pb) + 1e-9), s=self.scale)
        if self.prev_db is not None and db > self.prev_db: # 对偶下界（Dual Bound）上升
            r_progress += _safe_tanh((db - self.prev_db) / (abs(self.prev_db) + 1e-9), s=self.scale)
        # pdi decrease PDI 减小奖励
        if self.prev_pdi > 0: 
            r_progress += _safe_tanh((self.prev_pdi - pdi) / self.baseline_pdi, s=self.scale)

        # update prevs
        self.prev_gap = gap
        self.prev_pdi = pdi
        self.prev_pb = pb
        self.prev_db = db

        if not done: # 非终止时的奖励计算
            reward = (self.w_nodes * r_nodes + self.w_pace * r_pace +
                      self.w_gap * ( - _safe_tanh(gap / (self.baseline_gap + 1e-9), s=0.5) ) +
                      self.w_pdi * _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi) +
                      self.w_progress * r_progress)
            reward = float(np.clip(reward, -1.0, 1.0))
            # if self.logger:
            #     self.logger.info(f"H3 step: nodes={nodes}, r_nodes={r_nodes:.3f}, r_pace={r_pace:.3f}, r_prog={r_progress:.3f}, R={reward:.3f}")
            return reward

        # Terminal: blend speedup and quality with difficulty-aware weights
        status = model.getStatus() #求解状态
        ## 衡量搜索的节点相对baseline 用的节点数的比例, 用更少节点 → speedup > 1 → 好
        speedup = _ratio(self.B, max(nodes, 1), cap=5.0)
        gap_gain = _safe_tanh((self.baseline_gap - gap))
        pdi_gain = _safe_tanh((self.baseline_pdi - pdi) / self.baseline_pdi)
        if status == "optimal": #求解结束
            bonus = 1.0 + 3.0 * speedup
        elif status in ("infeasible", "unbounded"): #没找到可行解
            bonus = 0.8 + 2.0 * speedup
        elif status in ("timelimit", "nodelimit", "userinterrupt"):  #达到截止时间
            # bonus = 0.5 * speedup + 0.3 * gap_gain + 0.2 * pdi_gain
            bonus = -1.0 + 0.6 * max(0.0, gap_gain) + 0.4 * max(0.0, pdi_gain)
        else: #其他情况
            # bonus = 0.3 * speedup
            bonus = -1.0
        # if self.logger:
        #     self.logger.info(f"H3 terminal: status={status}, speedup={speedup:.3f}, gap_gain={gap_gain:.3f}, pdi_gain={pdi_gain:.3f}, bonus={bonus:.3f}")
        return float(np.clip(bonus, -5.0, 5.0))



#------------------------  工业进化版 ------------------------------------
class RewardH4:
    def __init__(self, logger=None, scale=1.0):
        self.logger = logger
        self.scale = scale
        # 初始默认重置
        self._internal_reset(1000, 1.0, 600.0)

    def reset(self, model, time_limit=600.0):
        # 1. 获取静态规模：非零元数量 (NZ) 所有约束矩阵里“非零系数”的总个数
        # nz = model.getNNonzeros()

        n_vars = len(model.getVars())
        n_conss = len(model.getConss())
        nz = n_vars + n_conss
        
        # 2. 获取初始解状态 (根节点 Gap)
        # first_gap = model.getGap()
        first_gap = 1.0
        # if math.isinf(first_gap) or first_gap > 1e6:
        #     first_gap = 1.0
            
        self._internal_reset(nz, first_gap, time_limit)

    def _internal_reset(self, nz, first_gap, time_limit):
        # 1. 难度指数 d：基于 NZ 规模自感知
        # 映射 NZ [1e3, 1e7] -> d [0, 1]
        self.d = min(max((math.log10(nz + 1) - 3.0) / 4.0, 0.0), 1.0)
        self.nz = nz
        
        # 2. 初始状态记录
        self.first_gap = max(first_gap, 1e-6)
        self.time_limit = max(time_limit, 1.0)
        
        # 3. 权重分配 (逻辑继承 H3，但 d 是自感知的)
        self.w_nodes = 0.55 * (1 - self.d) + 0.25 * self.d
        self.w_gap   = 0.10 * (1 - self.d) + 0.30 * self.d
        self.w_pdi   = 0.05 * (1 - self.d) + 0.20 * self.d
        self.w_progress = 0.15
       
        
        # 归一化权重
        s = self.w_nodes + self.w_gap + self.w_pdi + self.w_progress 
        self.w_nodes /= s; self.w_gap /= s; self.w_pdi /= s
        self.w_progress /= s; 

        # 4. 实时状态追踪
        self.prev_nodes = 0
        self.prev_gap = self.first_gap
        self.prev_pdi = 0.0
        self.prev_pb = None
        self.prev_db = None

    # def _dynamic_node_penalty(self, delta_nodes, delta_gap):
       
    #     if delta_nodes <= 0:
    #         return 0.0  # 没有生成新节点，无惩罚
            
    #     # 1. 计算当前步的 Gap 改善比例 (相对于初始 Gap)
    #     relative_improvement = max(0.0, delta_gap) / max(self.first_gap, 1e-6)
        
    #     # 2. 原谅机制 (Forgiveness)：
    #     # 如果 Gap 有改善，降低对产生节点的惩罚。
    #     # 放大系数 50.0 意味着：只要 Gap 相对初始值下降了 2% (0.02 * 50 = 1.0)，
    #     # tanh(1) 约等于 0.76，即免除 76% 的节点惩罚。
    #     forgiveness = _safe_tanh(relative_improvement * 50.0)
        
    #     # “无用节点”的比例
    #     wasted_ratio = 1.0 - forgiveness
        
    #     # 3. 动态单步预算 (Step Budget)：规模越小，单步能容忍的节点数越多
    #     nz_scale = math.log10(max(self.nz, 100))
    #     # 假设 NZ=100 (规模极小)，单步预算约 2500 节点
    #     # 假设 NZ=1,000,000 (规模极大)，单步预算约 833 节点
    #     step_budget = 5000.0 / nz_scale 
        
    #     excess_nodes = delta_nodes / step_budget
        
    #     # 4. 计算最终惩罚
    #     # 难度 d 越大 (问题越难)，stiffness 越小，惩罚越温和
    #     stiffness = 0.5 * (1.0 - self.d * 0.5) 
        
    #     # 只有“超额”且“无用”的节点才会带来惩罚
    #     penalty = -_safe_tanh(excess_nodes * wasted_ratio * stiffness, s=0.5)
        
    #     return penalty

    # def compute(self, model, done):
    #     nodes = int(model.getNNodes())
    #     # 时间进度
    #     tfrac = min(max(model.getSolvingTime() / self.time_limit, 0.0), 1.0)
        
    #     gap = float(model.getGap())
    #     # [OPT] 平滑处理无限大的 Gap：如果求解中途出现 inf，保持为上一步的 gap，防止进度奖励爆炸
    #     if math.isinf(gap): 
    #         gap = self.prev_gap 
        
    #     pdi = float(model.getPrimalDualIntegral())
    #     pb = model.getPrimalbound()
    #     db = model.getDualbound()

    #     # 计算当步的状态变化量
    #     delta_nodes = nodes - self.prev_nodes
    #     delta_gap = self.prev_gap - gap
    #     delta_pdi = max(0.0, pdi - self.prev_pdi)

    #     # 1. 各项组件计算
    #     r_nodes = self._dynamic_node_penalty(delta_nodes, delta_gap)
        
    #     # [OPT] Progress: 奖励相对于自身初始 Gap 的下降量
    #     r_progress = 0.0
    #     if self.prev_gap < float('inf'):
    #         r_progress += _safe_tanh((self.prev_gap - gap) / self.first_gap, s=self.scale)
        
    #     # [OPT] Bound 改善：使用稳定的 first_gap 替代 abs(prev_pb) 作为归一化分母
    #     stable_denom = max(self.first_gap, 1.0)
    #     if self.prev_pb is not None and pb < self.prev_pb:
    #         r_progress += 0.5 * _safe_tanh((self.prev_pb - pb) / stable_denom)
    #     if self.prev_db is not None and db > self.prev_db:
    #         r_progress += 0.5 * _safe_tanh((db - self.prev_db) / stable_denom)

    #     # [OPT] PDI 增量计算：只惩罚这一步新增的对偶积分，而非历史累计总和
    #     delta_pdi = max(0.0, pdi - self.prev_pdi)

    #     # 2. 组合奖励 (Step Reward)
    #     # 调整了 PDI 的惩罚力度，匹配 delta_pdi 的量级
    #     step_reward = (
    #         self.w_nodes * r_nodes +
    #         self.w_gap   * (-_safe_tanh(gap / self.first_gap, s=0.5)) +
    #         self.w_pdi   * (-_safe_tanh(delta_pdi / (self.first_gap * self.time_limit * 0.01 + 1e-6), s=0.5)) +
    #         self.w_progress * r_progress
    #     )
    #     step_reward = float(np.clip(step_reward, -1.0, 1.0))

    #     # [FIX] 核心 Bug 修复：在返回或处理 done 之前，必须更新所有 prev_* 状态！
    #     self.prev_nodes = nodes
    #     self.prev_gap = gap
    #     self.prev_pdi = pdi
    #     self.prev_pb = pb
    #     self.prev_db = db

    #     if not done:
    #         return step_reward

    #     # 3. 终止奖励 (Terminal Bonus)
    #     status = model.getStatus()
    #     time_saved_ratio = max(0.0, 1.0 - tfrac)
    #     gap_reduced_ratio = max(0.0, (self.first_gap - gap) / self.first_gap)
        
    #     if status == "optimal":
    #         bonus = 2.0 + 2.0 * time_saved_ratio
    #     elif status in ["infeasible", "unbounded"]:
    #         bonus = 2.0 + 2.0 * time_saved_ratio
    #     elif status in ["timelimit", "nodelimit"]:
    #         bonus = -1.0 + 1.5 * gap_reduced_ratio
    #     else:
    #         bonus = -1.0
            
    #     if self.logger:
    #          self.logger.info(f"H4 terminal: status={status}, step_reward={step_reward:.3f}, bonus={bonus:.3f}")
            
    #     # [FIX] 核心 Bug 修复：必须加上当步的常规奖励 step_reward，否则模型在最后一步会丢失进度反馈
    #     final_reward = float(np.clip(step_reward + bonus, -5.0, 5.0))
    #     return final_reward

    def _dynamic_node_penalty(self, delta_nodes, delta_gap):
            # 【修改】极其简单暴力的步数成本：每产生一个新节点，就扣除一点微小的固定体力。
            # 这逼迫模型必须用最少的节点解完树，而且不会因为树大而导致惩罚爆炸。
            base_penalty = -0.01 * delta_nodes
            
            # 原谅机制依然保留：如果 Gap 改善了，免除部分节点惩罚
            relative_improvement = max(0.0, delta_gap) / max(self.first_gap, 1e-6)
            forgiveness = _safe_tanh(relative_improvement * 50.0)
            
            return base_penalty * (1.0 - forgiveness)

    def compute(self, model, done):
        nodes = int(model.getNNodes())
        tfrac = min(max(model.getSolvingTime() / self.time_limit, 0.0), 1.0)
        
        gap = float(model.getGap())
        if math.isinf(gap): 
            gap = self.prev_gap 
        
        pdi = float(model.getPrimalDualIntegral())
        pb = model.getPrimalbound()
        db = model.getDualbound()

        delta_nodes = nodes - self.prev_nodes
        delta_gap = self.prev_gap - gap
        delta_pdi = max(0.0, pdi - self.prev_pdi)

        # 1. 各项组件计算
        r_nodes = self._dynamic_node_penalty(delta_nodes, delta_gap)
        
        # 进度奖励 (只奖励改善的部分，绝不惩罚现有的 Gap)
        r_progress = 0.0
        if self.prev_gap < float('inf'):
            r_progress += _safe_tanh((self.prev_gap - gap) / self.first_gap, s=self.scale)
        
        stable_denom = max(self.first_gap, 1.0)
        if self.prev_pb is not None and pb < self.prev_pb:
            r_progress += 0.5 * _safe_tanh((self.prev_pb - pb) / stable_denom)
        if self.prev_db is not None and db > self.prev_db:
            r_progress += 0.5 * _safe_tanh((db - self.prev_db) / stable_denom)

        # 2. 组合奖励 (Step Reward)
        # 【核心修改】删除了 w_gap 对绝对 gap 的持续惩罚！
        step_reward = (
            self.w_nodes * r_nodes +
            self.w_pdi   * (-_safe_tanh(delta_pdi / (self.first_gap * self.time_limit * 0.01 + 1e-6), s=0.5)) +
            self.w_progress * r_progress
        )
        step_reward = float(np.clip(step_reward, -1.0, 1.0))

        self.prev_nodes = nodes
        self.prev_gap = gap
        self.prev_pdi = pdi
        self.prev_pb = pb
        self.prev_db = db

        if not done:
            return step_reward

        # 3. 终止奖励 (Terminal Bonus)
        status = model.getStatus()
        time_saved_ratio = max(0.0, 1.0 - tfrac)
        gap_reduced_ratio = max(0.0, (self.first_gap - gap) / self.first_gap)
        
        if status == "optimal":
            # 【提升胜利果实】让成功解题的奖励足够大，覆盖掉之前的微小探索成本
            bonus = 5.0 + 3.0 * time_saved_ratio
        elif status in ["infeasible", "unbounded"]:
            bonus = 5.0 + 3.0 * time_saved_ratio
        elif status in ["timelimit", "nodelimit"]:
            # 【修改惩罚】时间或节点耗尽的惩罚变重，明确告诉模型这不合格
            bonus = -3.0 + 2.0 * gap_reduced_ratio
        else:
            bonus = -1.0
            
        if self.logger:
             self.logger.info(f"H4 terminal: status={status}, step_reward={step_reward:.3f}, bonus={bonus:.3f}")
            
        final_reward = float(np.clip(step_reward + bonus, -10.0, 10.0))
        return final_reward